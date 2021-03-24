import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from xformers.components.attention import Attention, AttentionConfig, register_attention


class LinformerSelfAttentionConfig(AttentionConfig):
    k: int  # dimension of the internal space


@register_attention("linformer")
class LinformerAttention(Attention):
    def __init__(
        self,
        dropout: float,
        causal: bool,
        dim_seq: int,
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
            k = dim_seq // 4

        self.E = nn.Linear(dim_seq, k, bias=False)
        self.F = nn.Linear(dim_seq, k, bias=False)
        self.attn_drop = nn.Dropout(dropout, inplace=True)

        if causal:
            mask = torch.tril(torch.ones(dim_seq, k), diagonal=0)
            mask[mask == 1] = -float("inf")

            # add the batch dimension and register the buffer in this nn.Module
            self.register_buffer("mask", mask.unsqueeze(0))
        else:
            self.mask = None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
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
        if input_mask is not None:
            assert (
                input_mask.shape[-2] == att.shape[-2]
                and input_mask.shape[-1] == att.shape[-1]
            ), (
                "Linformer uses a projected sequence, the input mask needs to be adapted in consequence."
                + "Please use the `causal` constructor argument if this is the intended effect"
            )
            att += input_mask.unsqueeze(0)

        if self.mask is not None:
            att += self.mask.unsqueeze(0)
        if input_mask is not None and self.mask is not None:
            logging.warning(
                "Getting both a mask and a positive causal setting, is that expected ?"
            )

        # Softmax to get the attention probabilities, then optional dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # Get to the predicted values, for all heads
        y = att @ v_projected  # (B, nh, S, S) x (B, nh, S, hs) -> (B, nh, S, hs)
        return y
