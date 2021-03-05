import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from xformers.components.attention import Attention, AttentionConfig

from . import register_attention


class MultiHeadAttentionConfig(AttentionConfig):
    residual_dropout: float


# TODO: @lefaudeux Change that to be a wrapper for different attention mechanisms


@register_attention("multi_head_attention")
class MultiHeadAttention(Attention):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    credits A. Karpathy
    https://github.com/karpathy/minGPT/blob/master/mingpt/model.py

    See "Attention is all you need", Vaswani et al. https://arxiv.org/abs/1706.03762v5
    FIXME: @lefaudeux placeholder, to be improved

    # TODO: expose different head computation splits
    # TODO: expose different dimensions key/query ?
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        attention_dropout: float,
        residual_dropout: float,
        n_heads: int,
        causal: bool,
        *args,
        **kwargs
    ):
        super().__init__()

        assert (
            dim_out % n_heads == 0
        )  # static preset for now, each head works on 1/d the embeddings, could be relaxed
        assert n_heads > 0

        self.dim_k = dim_out // n_heads

        # key, query, value projections for all heads
        self.key = nn.Linear(dim_in, dim_in)
        self.query = nn.Linear(dim_in, dim_in)
        self.value = nn.Linear(dim_in, dim_in)

        # Regularization
        self.attn_drop = nn.Dropout(attention_dropout, inplace=True)
        self.resid_drop = nn.Dropout(residual_dropout, inplace=True)

        # Output projection
        self.proj = nn.Linear(dim_in, dim_in)

        # Optional causal mask to ensure that attention is only applied to the left in the input sequence
        if causal:
            self.register_buffer("mask", self.generate_mask(dim_in))

        self.n_heads = n_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: handle channels
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)

        B, S, E = x.size()  # Batch x Sequence x Embedding

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(x).view(B, S, self.n_heads, self.dim_k).transpose(1, 2)
        )  # (B, nh, S, hs)
        q = (
            self.query(x).view(B, S, self.n_heads, self.dim_k).transpose(1, 2)
        )  # (B, nh, S, hs)
        v = (
            self.value(x).view(B, S, self.n_heads, self.dim_k).transpose(1, 2)
        )  # (B, nh, S, hs)

        # Self-attend: (B, nh, S, hs) x (B, nh, hs, S) -> (B, nh, S, S)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Optional masking
        if hasattr(self, "mask"):
            pass
            # FIXME @lefaudeux
            # att = att.masked_fill(self.mask[:, :S, :S] == 0, float("-inf"))

        # Softmax to get the attention probabilities, then dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # Get to the predicted values, for all heads
        y = att @ v  # (B, nh, S, S) x (B, nh, S, hs) -> (B, nh, S, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, S, E)
        )  # re-assemble all head outputs side by side

        # Output projection
        y = self.resid_drop(self.proj(y))
        return y
