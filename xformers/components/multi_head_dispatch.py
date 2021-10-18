# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from collections import namedtuple
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_

from xformers.components.attention import Attention

InProjParams = namedtuple("InProjParams", ["in_features", "out_features", "bias"])


class InProjContainer(nn.Module):
    """
    Handle all the input projections in one go, opportunistically fuse some operations.

    CREDITS: Inspired by https://github.com/pytorch/text/blob/master/torchtext/nn/modules/multiheadattention.py
    and the MultiHeadAttention implementation from PyTorch
    """

    def __init__(
        self,
        query_proj_params: InProjParams,
        key_proj_params: Optional[InProjParams],
        value_proj_params: Optional[InProjParams],
    ):

        super().__init__()

        assert (
            query_proj_params.in_features == query_proj_params.out_features
        ), "We assume in_features == out_features for queries, please provide your projection if this is not the case"

        # If nothing is specified for key and value, use the same as query
        if key_proj_params is None:
            key_proj_params = query_proj_params

        if value_proj_params is None:
            value_proj_params = query_proj_params

        # Catch a beneficial case, if Q,K,V dimensions are the same
        self.same_dimensions = (
            query_proj_params.in_features == key_proj_params.in_features
            and value_proj_params.in_features == key_proj_params.in_features
        )

        self.out_features = query_proj_params.out_features

        # - handle all the weights
        if self.same_dimensions:
            # We can use a single weight and bias buffer, which will speed up self attention
            self.in_proj_weight = nn.Parameter(
                torch.empty((3 * self.out_features, self.out_features))
            )
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)
        else:
            # The dimensions are different, use seperate buffers
            self.q_proj_weight = nn.Parameter(
                torch.empty((self.out_features, query_proj_params.in_features))
            )
            self.k_proj_weight = nn.Parameter(
                torch.empty((self.out_features, key_proj_params.in_features))
            )
            self.v_proj_weight = nn.Parameter(
                torch.empty((self.out_features, value_proj_params.in_features))
            )
            self.register_parameter("in_proj_weight", None)

        # - handle all the inputs
        if query_proj_params.bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * self.out_features))
        else:
            self.register_parameter("in_proj_bias", None)

        # - multi-head attention specific init for the weights and biases
        self._reset_parameters()

    def _reset_parameters(self):
        if self.in_proj_weight is not None:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.in_proj_weight is not None:
            if id(query) == id(key):
                # Self attention, get all the projected values at once
                # we compute everything transposed, so that q,k,v stay contiguous after splitting
                qkv = query @ self.in_proj_weight.transpose(-2, -1)

                if self.in_proj_bias is not None:
                    qkv += self.in_proj_bias

                q, k, v = map(
                    lambda x: x.contiguous(),
                    qkv.split(self.out_features, dim=-1),
                )
                return q, k, v

            else:
                # Not self attention
                # - bias free projection
                projections = self.in_proj_weight.split(self.out_features, dim=0)
                q, k, v = map(
                    lambda x, y: x @ y.transpose(1, 0), [query, key, value], projections
                )

                # - optionally add bias
                if self.in_proj_bias is not None:
                    biases = self.in_proj_bias.split(self.out_features, dim=0)
                    q, k, v = map(lambda x, y: x + y, [q, k, v], biases)

                return q, k, v

        # We have a weight per input, but share a bigger bias buffer
        assert (
            self.q_proj_weight is not None
            and self.k_proj_weight is not None
            and self.v_proj_weight is not None
        )

        # - bias free projection
        q, k, v = map(
            lambda x, y: x @ y.transpose(1, 0),
            [query, key, value],
            [self.q_proj_weight, self.k_proj_weight, self.v_proj_weight],
        )

        # - optionally add bias
        if self.in_proj_bias is not None:
            biases = self.in_proj_bias.split(self.out_features, dim=0)
            q, k, v = map(lambda x, y: x + y, [q, k, v], biases)

        return q, k, v


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


def _split_heads(t: torch.Tensor, B: int, S: int, H: int, Hs: int):
    return t.view(B, S, H, Hs).transpose(1, 2)


class MultiHeadDispatch(nn.Module):
    """
    A multi-head masked self-attention dispatch mechanism, with a projection at the end,
    following the architecture proposed in `Attention is all you need`_, Vaswani et al.

    The actual attention mechanism can vary, as well as the projections.
    This can be used to wrap the proposed attention mechanisms and make them multi-head aware,
    but it is optional.

    .. _`Attention is all you need`: https://arxiv.org/abs/1706.03762v5
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
                    query_proj_params=InProjParams(dim_model, dim_key, bias=bias),
                    key_proj_params=InProjParams(dim_model, dim_key, bias=bias)
                    if use_separate_proj_weight
                    else None,
                    value_proj_params=InProjParams(dim_model, dim_value, bias=bias)
                    if use_separate_proj_weight
                    else None,
                )
            )

        # Regularization
        self.resid_drop = nn.Dropout(residual_dropout, inplace=False)

        # Output projection
        self.proj = out_proj if out_proj else nn.Linear(dim_model, dim_model, bias=bias)
        if isinstance(self.proj, nn.Linear) and self.proj.bias is not None:
            constant_(self.proj.bias, 0.0)

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

        # Reshape k/q/v to either expose the heads, or fold the head dimension into the batch
        reshape_fn = (
            _split_heads if self.attention.requires_head_dimension else _fold_heads
        )

        k = reshape_fn(k, B, S_K, self.num_heads, self.dim_k)
        q = reshape_fn(q, B, S_Q, self.num_heads, self.dim_k)
        v = reshape_fn(v, B, S_K, self.num_heads, self.dim_k)

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
