from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from xformers.components.attention import Attention, AttentionConfig, register_attention
from xformers.components.attention.core import scaled_dot_product_attention


@dataclass(init=False)
class LinformerSelfAttentionConfig(AttentionConfig):
    from_seq_dim: int  # dimension of the input sequence
    k: Optional[int]  # dimension of the internal space


@register_attention("linformer")
class LinformerAttention(Attention):
    def __init__(
        self,
        dropout: float,
        from_seq_dim: int,
        k: Optional[int] = None,
        *args,
        **kwargs
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
            k = from_seq_dim // 4

        self.E = nn.Linear(from_seq_dim, k, bias=False)
        self.F = nn.Linear(from_seq_dim, k, bias=False)
        self.attn_drop = nn.Dropout(dropout, inplace=True)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_mask: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ):

        # Project K and V, over the sequence length axis
        k_projected = self.E(k.transpose(-2, -1)).transpose(-2, -1)
        v_projected = self.F(v.transpose(-2, -1)).transpose(-2, -1)

        # Optimized, sparse-aware self-attend: (B x nh, S, hs) -> (B x nh, S, hs)
        y = scaled_dot_product_attention(
            q, k_projected, v_projected, att_mask=att_mask, dropout=self.attn_drop
        )
        return y

    @classmethod
    def from_config(cls, config: AttentionConfig) -> "Attention":
        return cls(**LinformerSelfAttentionConfig.as_patchy_dict(config))
