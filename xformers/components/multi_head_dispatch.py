# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.init import constant_

from xformers.components.attention import Attention
from xformers.components.in_proj_container import InProjContainer, InProjParams
from xformers.components.positional_embedding import RotaryEmbedding


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
    use_rotary_embeddings: Optional[bool]
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
        use_rotary_embeddings: Optional[bool] = False,
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

        # Optional rotary embeddings
        self.rotary_embeddings = (
            RotaryEmbedding(self.dim_k) if use_rotary_embeddings else None
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
        key_padding_mask: Optional[torch.Tensor] = None,
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

        if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
            max_batch = max((query.shape[0], key.shape[0], value.shape[0]))
            query, key, value = map(
                lambda x: x.expand(max_batch, -1, -1), [query, key, value]
            )

        B, S_Q, _ = query.size()  # Batch x Sequence x Embedding (latent)
        _, S_K, _ = key.size()  # K, Q's sequence length could differ

        # Catch different query and key length but a causal attention
        if S_Q != S_K:
            assert (
                not self.attention.requires_same_k_q_dimensions
            ), "This attention mechanism requires query and key to have the same sequence (context) lengths"

            if hasattr(self.attention, "causal"):
                assert not self.attention.causal, (
                    "Causal attention is not supported when key and query have different sequence lengths.\n"
                    + "In that case causality is ill-determined. Please pad your sequences accordingly"
                )

        if self.attention.requires_skip_multi_head:
            return self.attention(
                query, key, value, att_mask=att_mask, key_padding_mask=key_padding_mask
            )

        # Calculate query, key, values for all heads in batch
        if self.attention.requires_input_projection:
            q, k, v = self.in_proj_container(query=query, key=key, value=value)
        else:
            k, q, v = key, query, value

        # Optional: rotary embedding, add relative positioning information
        if self.rotary_embeddings:
            # rotary requires the head dimension
            q = _split_heads(q, B, S_Q, self.num_heads, self.dim_k)
            k = _split_heads(k, B, S_K, self.num_heads, self.dim_k)
            v = _split_heads(v, B, S_K, self.num_heads, self.dim_k)

            q, k = self.rotary_embeddings(q=q, k=k)

            if not self.attention.requires_head_dimension:
                q, k, v = q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1)

        else:
            # Reshape k/q/v to either expose the heads, or fold the head dimension into the batch
            reshape_fn = (
                _split_heads if self.attention.requires_head_dimension else _fold_heads
            )

            q = reshape_fn(q, B, S_Q, self.num_heads, self.dim_k)
            k = reshape_fn(k, B, S_K, self.num_heads, self.dim_k)
            v = reshape_fn(v, B, S_K, self.num_heads, self.dim_k)

        # Self-attend
        y = self.attention(
            q=q, k=k, v=v, att_mask=att_mask, key_padding_mask=key_padding_mask
        )

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
