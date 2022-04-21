# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from xformers.components.attention import Attention, AttentionConfig, register_attention
from xformers.components.attention.core import scaled_dot_product_attention


@dataclass
class LinformerSelfAttentionConfig(AttentionConfig):
    seq_len: int  # dimension of the input sequence
    k: Optional[int]  # dimension of the internal space


@register_attention("linformer", LinformerSelfAttentionConfig)
class LinformerAttention(Attention):
    def __init__(
        self, dropout: float, seq_len: int, k: Optional[int] = None, *args, **kwargs
    ):
        """
        Linformer attention mechanism,
        from `Linformer: Self-Attention with Linear Complexity`_, Wang et al (2020).
        The original notation is kept as is.

        .. _`Linformer: Self-Attention with Linear Complexity` : https://arxiv.org/abs/2006.04768v2
        """
        super().__init__()

        if k is None:
            k = seq_len // 4

        self.k = k
        self.E = nn.Linear(seq_len, k, bias=False)
        self.F = nn.Linear(seq_len, k, bias=False)
        self.attn_drop = nn.Dropout(dropout, inplace=False)
        self.seq_len = seq_len

        # MHA related flags:
        # kq need to have the same dimension
        self.requires_same_k_q_dimensions = True

        # This attention does not support attention masks
        self.supports_attention_mask = False

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *args, **kwargs
    ):
        # Handle a smaller dimension than expected
        padding = 0
        if q.shape[1] < self.seq_len:
            padding = self.seq_len - q.shape[1]
            pad_dims = (0, 0, 0, padding)
            q = torch.nn.functional.pad(q, pad_dims)
            k = torch.nn.functional.pad(k, pad_dims)
            v = torch.nn.functional.pad(v, pad_dims)

        k_projected = self.E(k.transpose(-2, -1)).transpose(-2, -1)
        v_projected = self.F(v.transpose(-2, -1)).transpose(-2, -1)

        y = scaled_dot_product_attention(
            q=q, k=k_projected, v=v_projected, att_mask=None, dropout=self.attn_drop
        )

        y = self.attn_drop(y)

        return y[:, :-padding, :] if padding > 0 else y
