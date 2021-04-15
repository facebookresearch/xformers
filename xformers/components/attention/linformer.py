import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from xformers.components.attention import Attention, AttentionConfig, register_attention


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

        # Self-attend: (B, nh, S, hs) x (B, nh, hs, S) -> (B, nh, S, S)
        att = (q @ k_projected.transpose(-2, -1)) * (
            1.0 / math.sqrt(k_projected.size(-1))
        )

        # Optional masking
        if att_mask is not None:
            assert (
                att_mask.shape[-2] == att.shape[-2]
                and att_mask.shape[-1] == att.shape[-1]
            ), (
                "Linformer uses a projected sequence, the input mask needs to be adapted in consequence."
                + " Please use the `causal` constructor argument if this is the intended effect"
            )
            att += att_mask

        # Softmax to get the attention probabilities, then optional dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # Get to the predicted values, for all heads
        y = att @ v_projected  # (B, nh, S, S) x (B, nh, S, hs) -> (B, nh, S, hs)
        return y

    @classmethod
    def from_config(cls, config: AttentionConfig) -> "Attention":
        return cls(**LinformerSelfAttentionConfig.as_patchy_dict(config))
