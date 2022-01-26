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
import torch.nn.functional as F
import math

from xformers.components.attention import (
    Attention,
    AttentionConfig,
    maybe_sparsify,
    register_attention,
    sparsify,
)

from xformers.components.attention.core import scaled_dot_product_attention


@dataclass
class BlockNativeAttentionConfig(AttentionConfig):
    block_size: int
    force_sparsity: Optional[bool]

@register_attention("block_native", BlockNativeAttentionConfig)
class BlockNativeAttention(Attention):
    def __init__(
        self,
        dropout: float,
        block_size: int = 256,
        force_sparsity: bool = False,
        *args, **kwargs
    ):
        super().__init__()
        self.attn_drop = nn.Dropout(dropout, inplace=False)
        self.block_size = block_size
        self.force_sparsity = force_sparsity
        self.attention_mask: Optional[torch.Tensor] = None

    def _get_block_mask(self, seq_len: torch.Size) -> torch.Tensor:
        repeat = seq_len // self.block_size
        block_mask = torch.ones(self.block_size, self.block_size)
        mask = torch.kron(torch.eye(repeat), block_mask).bool()
        mask = sparsify(mask) if self.force_sparsity else maybe_sparsify(mask)
        return mask


    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_mask: Optional[torch.Tensor] = None,
        *args, **kwargs
    ):

        orig_len = q.shape[1]
        # pad the input length to factors of block size
        def _pad_to_window_size(x, window_size):
            seq_len = x.size(-2)
            pad_len = (window_size - seq_len % window_size) % window_size
            return F.pad(x, (0,0,0,pad_len), value=0), pad_len
        q, _ = _pad_to_window_size(q, self.block_size)
        k, _ = _pad_to_window_size(k, self.block_size)
        v, _ = _pad_to_window_size(v, self.block_size)

        if self.attention_mask is None or self.attention_mask.shape[1] != q.shape[1]:
            self.attention_mask = self._get_block_mask(q.shape[1]).to(q.device)

        out = scaled_dot_product_attention(
            q=q.contiguous(), k=k.contiguous(), v=v.contiguous(), att_mask=self.attention_mask, dropout=self.attn_drop
        )[:,:orig_len]

        return out

