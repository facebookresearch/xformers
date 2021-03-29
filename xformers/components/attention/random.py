import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from xformers.components.attention import Attention, AttentionConfig, register_attention


class RandomAttentionConfig(AttentionConfig):
    r: float  # the ratio of keys that the query can attend to. 1.0 means dense attention
    constant_masking: bool  # whether the randomness is per query or defined at construction time


@register_attention("random")
class RandomAttention(Attention):
    def __init__(
        self,
        dropout: float,
        causal: bool,
        from_seq_dim: int,
        to_seq_dim: Optional[int] = None,
        r: float = 0.5,
        constant_masking: bool = True,
        *args,
        **kwargs
    ):
        """
        "Random" attention, as proposed for instance in _BigBird.
        Random means in that case means that each query can attend to a random set of keys.

        Args:
            r (float): the ratio in [0,1] of keys that the query can attend to
            constant_masking (bool): if true, keep the same random set for all queries.

        .. warning: the current implementation does not lead to saved memory, could be changed for biggest sizes
            (by using sparse matrices)

        _BigBird: "Zaheer, M., Guruganesh, G., Dubey, A., Ainslie, J., Alberti, C., Ontanon, S., Pham, P., Ravula,
         A., Wang, Q., Yang, L., & Ahmed, A. (2020). Big Bird: Transformers for Longer Sequences. ArXiv, NeurIPS."
        """
        super().__init__()

        if to_seq_dim is None:
            to_seq_dim = from_seq_dim

        self.from_seq_dim = from_seq_dim
        self.to_seq_dim = to_seq_dim

        self.attn_drop = nn.Dropout(dropout, inplace=True)

        if causal:
            mask = torch.tril(torch.ones(from_seq_dim, to_seq_dim), diagonal=0)
            mask[mask == 1] = -float("inf")

            # add the batch dimension and register the buffer in this nn.Module
            self.register_buffer("mask", mask.unsqueeze(0))
        else:
            self.mask = None

        self.r = r
        self.rand_mask = self._get_rand_mask()
        self.constant_masking = constant_masking

    def _get_rand_mask(self) -> torch.Tensor:
        return torch.FloatTensor(self.from_seq_dim, self.to_seq_dim).uniform_() < self.r

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ):

        # Self-attend: (B, nh, S, hs) x (B, nh, hs, S) -> (B, nh, S, S)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Rand masking
        if not self.constant_masking:
            self.rand_mask = self._get_rand_mask()

        att[:, :, ~self.rand_mask] = -float("inf")

        # Optional masking
        if input_mask is not None:
            att += input_mask.unsqueeze(0)

        # Self (causal) masking
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
        y = att @ v  # (B, nh, S, S) x (B, nh, S, hs) -> (B, nh, S, hs)
        return y
