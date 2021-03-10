import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from xformers.components.attention import Attention, register_attention


@register_attention("scaled_dot_product")
class ScaledDotProduct(Attention):
    r"""
    Implementing the Scaled Dot-Product attention proposed in
    "Attention is all you need", Vaswani et al. https://arxiv.org/abs/1706.03762v5
    """

    def __init__(
        self,
        dropout=0.0,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.attn_drop = nn.Dropout(dropout, inplace=True)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
    ):
        # Self-attend: (B, nh, S, hs) x (B, nh, hs, S) -> (B, nh, S, S)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Optional masking
        if input_mask:
            # att = att.masked_fill(input_mask[:, :S, :S] == 0, float("-inf"))
            pass
        elif hasattr(self, "mask"):
            pass
            # FIXME @lefaudeux
            # att = att.masked_fill(self.mask[:, :S, :S] == 0, float("-inf"))

        # Softmax to get the attention probabilities, then optional dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # Get to the predicted values, for all heads
        y = att @ v  # (B, nh, S, S) x (B, nh, S, hs) -> (B, nh, S, hs)
        return y
