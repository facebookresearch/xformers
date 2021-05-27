from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from xformers.components.attention import Attention
from xformers.utils import ExtensibleConfig


@dataclass(init=False)
class MultiHeadDispatchConfig(ExtensibleConfig):
    dim_model: int
    residual_dropout: float
    num_heads: int
    attention: Attention
    dim_key: Optional[int]
    dim_value: Optional[int]


# Move head forward and fold into batch dim. dimensions become (B * nh, S, hs)
def _fold_heads(t: torch.Tensor, B: int, S: int, H: int, Hs: int):
    return t.view(B, S, H, Hs).transpose(1, 2).flatten(start_dim=0, end_dim=1)


class MultiHeadDispatch(nn.Module):
    """
    A vanilla multi-head masked self-attention dispatch mechanism, with a projection at the end,
    following the architecture proposed in
    "Attention is all you need", Vaswani et al. https://arxiv.org/abs/1706.03762v5

    The actual attention mechanism can vary, be it scaled dot product, local or other
    """

    def __init__(
        self,
        dim_model: int,
        residual_dropout: float,
        num_heads: int,
        attention: Attention,
        dim_key: Optional[int] = None,
        dim_value: Optional[int] = None,
        *args,
        **kwargs,
    ):
        super().__init__()

        # TODO: Expose the projection method,
        # see https://github.com/pytorch/text/blob/torchtext/nn/modules/multiheadattention.py#L5-L36, very clean

        assert (
            dim_model % num_heads == 0
        )  # static preset for now, each head works on 1/d the embeddings, could be relaxed
        assert num_heads > 0

        # Popular default is that all latent dimensions are the same
        dim_key, dim_value = map(lambda x: x if x else dim_model, (dim_key, dim_value))

        self.num_heads = num_heads
        self.dim_k = dim_key // num_heads
        self.dim_value = dim_value
        self.dim_model = dim_model
        self.attention = attention

        # key, query, value projections for all heads
        self.project_key = nn.Linear(
            dim_model, dim_key, bias=False
        )  # NOTE: optional bias ?
        self.project_query = nn.Linear(dim_model, dim_key, bias=False)
        self.project_value = nn.Linear(dim_model, dim_value, bias=False)

        # Regularization
        self.resid_drop = nn.Dropout(residual_dropout, inplace=False)

        # Output projection
        self.proj = nn.Linear(dim_model, dim_model, bias=False)

    def _check(self, t, name):
        assert (
            t.shape[2] % self.dim_k == 0
        ), f"the {name} embeddings need to be divisible by the number of heads"

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        att_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # Check the dimensions properly
        self._check(query, "query")
        self._check(value, "value")
        self._check(key, "key")

        B, S, _ = query.size()  # Batch x Sequence x Embedding (latent)

        # Calculate query, key, values for all heads in batch
        k, q, v = (
            self.project_key(key),
            self.project_query(query),
            self.project_value(value),
        )

        k = _fold_heads(k, B, S, self.num_heads, self.dim_k)
        q = _fold_heads(q, B, S, self.num_heads, self.dim_k)
        v = _fold_heads(v, B, S, self.num_heads, self.dim_k)

        # Self-attend
        y = self.attention(k, q, v, att_mask=att_mask)

        # Re-assemble all head outputs side by side
        y = (
            y.view(B, self.num_heads, S, self.dim_k)
            .transpose(1, 2)
            .flatten(start_dim=2, end_dim=3)
        )

        # Output projection, dropout and good to go
        y = self.resid_drop(self.proj(y))

        # Return the same sequence size as the input
        return y

    @classmethod
    def from_config(cls, config: MultiHeadDispatchConfig):
        return cls(**MultiHeadDispatchConfig.as_patchy_dict(config))
