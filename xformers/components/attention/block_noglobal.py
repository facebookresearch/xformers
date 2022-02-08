# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Simple non-ovelapping local block attention. To test how strong are the locality inductive bias in these tasks
"""


from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from functools import partial

from xformers.components.attention import Attention, AttentionConfig, register_attention


@dataclass
class BlockNoglobalAttentionConfig(AttentionConfig):
    block_size: int
    num_heads: int
    require_key_mask: bool

@register_attention("block_noglobal", BlockNoglobalAttentionConfig)
class BlockNoglobalAttention(Attention):
    def __init__(
        self,
        dropout: float,
        num_heads: int,
        block_size: int = 256, #
        *args, **kwargs
    ):
        super().__init__()
        self.block_size = block_size
        self.drop_attn = nn.Dropout(dropout)
        self.num_head = num_heads

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        *args, **kwargs
    ):
        # Notation: batch size: B, sequence length: L, number of blocks: nb, number of heads: nh
        # q, k, v: (B * nh, L, head_dim)
        bh = q.size(0)
        orig_seq_len = q.size(1)
        bsz = bh // self.num_head
        head_dim = q.size(-1)

        if key_padding_mask is None:
            key_padding_mask = torch.zeros(int(q.shape[0]/self.num_head), q.size(-2))
        key_padding_mask = key_padding_mask.to(q)

        # pad the input length to factors of bucket size
        def _pad_to_window_size(x, window_size):
            seq_len = x.size(-2)
            pad_len = (window_size - seq_len % window_size) % window_size
            return F.pad(x, (0,0,0,pad_len), value=0), pad_len
        q, _ = _pad_to_window_size(q, self.block_size)
        k, _ = _pad_to_window_size(k, self.block_size)
        v, _ = _pad_to_window_size(v, self.block_size)

        if key_padding_mask.shape[1] % self.block_size != 0:
            pad_len = (self.block_size - key_padding_mask.shape[1] % self.block_size) % self.block_size
            key_padding_mask = torch.cat([key_padding_mask, key_padding_mask.new_ones(key_padding_mask.size(0), pad_len).to(key_padding_mask)], dim=1)

        assert q.shape[1] % self.block_size == 0
        num_blocks = q.shape[1] // self.block_size
        b_q = blockify(num_blocks, q)
        b_k, b_v = map(partial(blockify, num_blocks), (k, v)) # (B * nh, nb, L // nb, head_dim)

        dots = torch.einsum('buie,buje->buij', b_q, b_k) * (head_dim ** -0.5) # (B * nh, nb, L // nb, L // nb)
        mask_value = -10000

        # this model does use global token markers (-1)
        q_mask = key_padding_mask != 1

        # 1 means not masking
        kv_mask = q_mask
        mq, mk = map(partial(blockify, num_blocks), (q_mask, kv_mask)) # (B, nb, L // nb)
        mask = mq[:, :, :, None] * mk[:, :, None, :] # (B, nb, L // nb, L // nb)

        dots = dots.view(bsz, self.num_head, num_blocks, self.block_size, self.block_size)
        dots.masked_fill_(~mask.unsqueeze(1), mask_value)

        # add relational bias
        seq_len = q.size(1)
        if attn_bias is not None:
            dots += attn_bias.unsqueeze(2)[:,:,:,:seq_len,:seq_len]

        block_attn_weights = dots.view(bsz*self.num_head, -1, self.block_size)
        all_attn_probs = block_attn_weights.softmax(dim=-1)
        all_attn_probs = self.drop_attn(all_attn_probs)

        # calculate block attention
        block_attn_probs = all_attn_probs[:, :, :block_attn_weights.shape[-1]]
        block_attn_probs = block_attn_probs.view(bsz*self.num_head, -1, self.block_size, self.block_size)
        out = block_attn_probs.matmul(b_v).view(bsz*self.num_head, -1, head_dim)

        out = out[:,:orig_seq_len]
        return out

def blockify(num_blocks, t, dim=1):
    shape = list(t.shape)
    shape[dim:dim+1] = [num_blocks, -1]
    return t.reshape(*shape)
