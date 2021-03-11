from typing import Optional

import torch
import torch.nn as nn
from attrdict import AttrDict

from xformers.components.attention import Attention


class MultiHeadDispatchConfig(AttrDict):
    residual_dropout: float
    dim_in: int
    dim_out: int
    n_heads: int
    attention: Optional[Attention]


class MultiHeadDispatch(nn.Module):
    """
    A vanilla multi-head masked self-attention dispatch mechanism, with a projection at the end,
    following the architecture proposed in
    "Attention is all you need", Vaswani et al. https://arxiv.org/abs/1706.03762v5

    The actual attention mechanism can vary, be it scaled dot product, local or other

    credits A. Karpathy
    https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(
        self,
        dim_model: int,
        residual_dropout: float,
        n_heads: int,
        attention: Attention,
        causal: bool = False,
        dim_seq: Optional[int] = None,
        dim_key: Optional[int] = None,
        dim_value: Optional[int] = None,
        *args,
        **kwargs,
    ):
        super().__init__()

        # TODO: Handle max sequence size / mask instead of fixed size
        # TODO: Expose the projection method,
        # see https://github.com/pytorch/text/blob/torchtext/nn/modules/multiheadattention.py#L5-L36, very clean

        assert (
            dim_model % n_heads == 0
        )  # static preset for now, each head works on 1/d the embeddings, could be relaxed
        assert n_heads > 0

        # Popular default is that all latent dimensions are the same
        dim_seq, dim_key, dim_value = map(
            lambda x: x if x else dim_model, (dim_seq, dim_key, dim_value)
        )

        self.n_heads = n_heads
        self.dim_k = dim_key // n_heads
        self.dim_value = dim_value
        self.dim_model = dim_model
        self.attention = attention

        # key, query, value projections for all heads
        self.key = nn.Linear(dim_model, dim_key, bias=False)  # NOTE: optional bias ?
        self.query = nn.Linear(dim_model, dim_key, bias=False)
        self.value = nn.Linear(dim_model, dim_value, bias=False)

        # Regularization
        self.resid_drop = nn.Dropout(residual_dropout, inplace=True)

        # Output projection
        self.proj = nn.Linear(dim_model, dim_model, bias=False)

        # Optional masking
        if causal:
            mask = torch.tril(torch.ones(dim_seq, dim_seq), diagonal=0)
            mask[mask == 1] = -float("inf")

            # add the batch dimension and register the buffer in this nn.Module
            self.register_buffer("mask", mask.unsqueeze(0))
        else:
            self.mask = None

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        def _check(t, name):
            assert (
                t.shape[2] % self.dim_k == 0
            ), f"the {name} embeddings need to be divisible by the number of heads"

        # Check the dimensions properly
        _check(query, "query")
        _check(value, "value")
        _check(key, "key")

        B, S, E = query.size()  # Batch x Sequence x Embedding (latent)

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # dimensions become (B, nh, S, hs)
        k = self.key(key).view(B, S, self.n_heads, self.dim_k).transpose(1, 2)
        q = self.query(query).view(B, S, self.n_heads, self.dim_k).transpose(1, 2)
        v = self.value(value).view(B, S, self.n_heads, self.dim_k).transpose(1, 2)

        # Self-attend: (B, nh, S, hs) x (B, nh, hs, S) -> (B, nh, S, S)
        y = self.attention(k, q, v, input_mask=self.mask)
        y = (
            y.transpose(1, 2).contiguous().view(B, S, E)
        )  # re-assemble all head outputs side by side

        # Output projection, dropout and good to go
        y = self.resid_drop(self.proj(y))
        return y

    @classmethod
    def from_config(cls, config: MultiHeadDispatchConfig):
        return cls(**config)
