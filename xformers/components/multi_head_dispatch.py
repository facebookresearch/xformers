from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from xformers.components.attention import Attention


class InProjContainer(torch.nn.Module):
    """
    Inspired by https://github.com/pytorch/text/blob/master/torchtext/nn/modules/multiheadattention.py

    query_proj: a projection layer for query.
    key_proj: a projection layer for key.
    value_proj: a projection layer for value.
    """

    def __init__(
        self,
        query_proj: nn.Module,
        key_proj: Optional[nn.Module],
        value_proj: Optional[nn.Module],
    ):

        super().__init__()

        self.query_proj = query_proj

        # If no projection is passed for key and value, the projection from the Query (minus optional bias) is used
        bias_free_query_proj = nn.Linear(
            self.query_proj.in_features, self.query_proj.out_features, bias=False  # type: ignore
        )
        bias_free_query_proj.weights = self.query_proj.weight

        self.key_proj = key_proj if key_proj is not None else bias_free_query_proj
        self.value_proj = value_proj if value_proj is not None else bias_free_query_proj

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.query_proj(query), self.key_proj(key), self.value_proj(value)


@dataclass
class MultiHeadDispatchConfig:
    dim_model: int
    residual_dropout: float
    num_heads: int
    attention: Attention
    bias: bool
    dim_key: Optional[int]
    dim_value: Optional[int]
    in_proj_container: Optional[InProjContainer]
    use_separate_proj_weight: Optional[bool]
    out_proj: Optional[nn.Module]


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
        bias: bool = True,
        dim_key: Optional[int] = None,
        dim_value: Optional[int] = None,
        in_proj_container: Optional[InProjContainer] = None,
        use_separate_proj_weight: Optional[bool] = False,
        out_proj: Optional[nn.Module] = None,
        *args,
        **kwargs,
    ):
        super().__init__()

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
        # critical options are
        # - are we sharing weights ?
        # - are we adding biases, and if yes are they shared ?
        if attention.requires_input_projection:
            self.in_proj_container = (
                in_proj_container
                if in_proj_container is not None
                else InProjContainer(
                    query_proj=nn.Linear(
                        dim_model, dim_key, bias=bias
                    ),  # NOTE: optional bias ?
                    key_proj=nn.Linear(dim_model, dim_key, bias=False)
                    if use_separate_proj_weight
                    else None,
                    value_proj=nn.Linear(dim_model, dim_value, bias=False)
                    if use_separate_proj_weight
                    else None,
                )
            )

        # Regularization
        self.resid_drop = nn.Dropout(residual_dropout, inplace=False)

        # Output projection
        self.proj = out_proj if out_proj else nn.Linear(dim_model, dim_model, bias=bias)

    def _check(self, t, name):
        assert (
            t.shape[2] % self.dim_k == 0
        ), f"the {name} embeddings need to be divisible by the number of heads"

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        att_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Expected input dimensions are [batch size, sequence length, embed dim]
        Output dimensions are [batch size, sequence length, embed dim]
        """

        if key is None:
            key = query
        if value is None:
            value = query

        # Check the dimensions properly
        self._check(query, "query")
        self._check(value, "value")
        self._check(key, "key")

        B, S_Q, _ = query.size()  # Batch x Sequence x Embedding (latent)
        _, S_K, _ = key.size()  # K, Q's sequence length could differ

        # Calculate query, key, values for all heads in batch
        if self.attention.requires_input_projection:
            q, k, v = self.in_proj_container(query=query, key=key, value=value)
        else:
            k, q, v = key, query, value

        k = _fold_heads(k, B, S_K, self.num_heads, self.dim_k)
        q = _fold_heads(q, B, S_Q, self.num_heads, self.dim_k)
        v = _fold_heads(v, B, S_K, self.num_heads, self.dim_k)

        # Self-attend
        y = self.attention(q=q, k=k, v=v, att_mask=att_mask)

        # Re-assemble all head outputs side by side
        y = (
            y.view(B, self.num_heads, S_Q, self.dim_k)
            .transpose(1, 2)
            .flatten(start_dim=2, end_dim=3)
        )

        # Output projection, dropout and good to go
        y = self.resid_drop(self.proj(y))

        # Return the same sequence size as the input
        return y

    @classmethod
    def from_config(cls, config: MultiHeadDispatchConfig):
        # Generate the class inputs from the config
        fields = asdict(config)

        # Skip all Nones so that default values are used
        fields = {k: v for k, v in fields.items() if v is not None}

        return cls(**fields)
