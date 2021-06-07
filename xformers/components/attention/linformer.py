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
        Linformer attention mechanism, from
        "
        Linformer: Self-Attention with Linear Complexity. ArXiv, 2048(2019)
        Wang, S., Li, B. Z., Khabsa, M., Fang, H., & Ma, H. (2020).
        "

        The paper's notation are kept wherever possible
        """
        super().__init__()

        if k is None:
            k = seq_len // 4

        self.k = k
        self.E = nn.Linear(seq_len, k, bias=False)
        self.F = nn.Linear(seq_len, k, bias=False)
        self.attn_drop = nn.Dropout(dropout, inplace=False)
        self.seq_len = seq_len

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *args, **kwargs
    ):
        k_projected = self.E(k.transpose(-2, -1)).transpose(-2, -1)
        v_projected = self.F(v.transpose(-2, -1)).transpose(-2, -1)

        y = scaled_dot_product_attention(
            q, k_projected, v_projected, att_mask=None, dropout=self.attn_drop
        )
        return y
