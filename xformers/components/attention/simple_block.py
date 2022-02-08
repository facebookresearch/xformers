# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from functools import partial, reduce
from inspect import isfunction
import torch.nn.functional as F

from xformers.components.attention import Attention, AttentionConfig, register_attention
from operator import mul

@dataclass
class BlockSelfAttentionConfig(AttentionConfig):
    window_size: int
    num_heads: int


@register_attention("simple_block", BlockSelfAttentionConfig)
class SimpleBlockAttention(Attention):
    def __init__(
        self,
        dropout: float,
        window_size: int,
        num_heads: int,
        *args,
        **kwargs
    ):
        """
        A straightforward implementation of disjoint block attentions using native pytorch operations and supports CPU inference.

        The sequence is split into equal-length chunks and the self-attention is calculated separately in each chunk
        """
        super().__init__()

        self.chunk_size = window_size
        self.drop_attn = nn.Dropout(dropout)
        self.num_head = num_heads

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        *args, **kwargs
    ):
        bh = q.size(0)
        orig_seq_len = q.size(1)
        bsz = bh // self.num_head
        head_dim = q.size(-1)

        assert key_padding_mask is not None
        key_padding_mask = key_padding_mask.to(q)

        # pad the input length to factors of chunk size
        def _pad_to_window_size(x, window_size):
            seq_len = x.size(-2)
            pad_len = (window_size - seq_len % window_size) % window_size
            return F.pad(x, (0,0,0,pad_len), value=0), pad_len

        q, _ = _pad_to_window_size(q, self.chunk_size)
        k, _ = _pad_to_window_size(k, self.chunk_size)
        v, _ = _pad_to_window_size(v, self.chunk_size)

        if key_padding_mask.shape[1] % self.chunk_size != 0:
            pad_len = (self.chunk_size - key_padding_mask.shape[1] % self.chunk_size) % self.chunk_size
            key_padding_mask = torch.cat([key_padding_mask, key_padding_mask.new_ones(key_padding_mask.size(0), pad_len).to(key_padding_mask)], dim=1)

        buckets = q.shape[1] // self.chunk_size
        b_q = bucket(buckets, q)
        b_k, b_v = map(partial(bucket, buckets), (k, v)) # BH * bct * n_b * D

        dots = torch.einsum('buie,buje->buij', b_q, b_k) * (head_dim ** -0.5)
        mask_value = -10000

        # The model does use global token markers (-1)
        q_mask = key_padding_mask != 1

        # 1 means not masking
        kv_mask = q_mask
        mq, mk = bucket(buckets, q_mask), bucket(buckets, kv_mask) # B * bkt * n_b
        expand_head_and_merge_into_batch = lambda x: merge_dims(0, 1, expand_dim(x.unsqueeze(1), 1, self.num_head))
        mq, mk = map(expand_head_and_merge_into_batch, (mq, mk)) # BH * bkt * n_b
        mask = mq[:, :, :, None] * mk[:, :, None, :]
        dots.masked_fill_(~mask, mask_value)
        del mask

        all_attn = dots.view(bsz*self.num_head, -1, self.chunk_size)
        all_attn_probs = all_attn.softmax(dim=-1)
        all_attn_probs = self.drop_attn(all_attn_probs)

        block_attn_probs = all_attn_probs[:, :, :all_attn.shape[-1]]
        block_attn_probs = block_attn_probs.view(bsz*self.num_head, -1, self.chunk_size, self.chunk_size)
        out = block_attn_probs.matmul(b_v).view(bsz*self.num_head, -1, head_dim)

        out = out[:,:orig_seq_len]
        return out

def expand_dim(t, dim, k):
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def merge_dims(ind_from, ind_to, tensor):
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tensor.reshape(*shape)

def bucket(buckets, t, dim=1):
    shape = list(t.shape)
    shape[dim:dim+1] = [buckets, -1]
    return t.reshape(*shape)

def unbucket(t, dim=1):
    shape = list(t.shape)
    shape[dim:dim+2] = [-1]
    return t.reshape(*shape)
