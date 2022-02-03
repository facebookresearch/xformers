# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn.init import (
    _calculate_fan_in_and_fan_out,
    _no_grad_uniform_,
    constant_,
    xavier_uniform_,
)


def small_init_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    r"""Fills the input `Tensor` with values according to the method
    described in `Transformer Without Tears`_, using a uniform distribution.

    This is a variation of the Xavier init. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-a, a)` where

    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + 4 * \text{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor

    .. _`Transformer Without Tears`: https://doi.org/10.5281/zenodo.3525484

    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + 4 * fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    return _no_grad_uniform_(tensor, -a, a)


@dataclass
class InProjParams:
    in_features: int
    out_features: int
    bias: bool
    small_init: bool = False


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
        self.q_p_params = query_proj_params
        self.k_p_params = key_proj_params
        self.v_p_params = value_proj_params

        # - handle all the weights
        # save the requested init method
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
            # 1/sqrt(2) init is empirically beneficial in that case
            self.in_proj_weight = self._init_weights(
                self.q_p_params, self.in_proj_weight, gain=1.0 / math.sqrt(2)
            )

        else:
            self.q_proj_weight = self._init_weights(
                self.q_p_params, self.q_proj_weight, gain=1.0
            )
            self.k_proj_weight = self._init_weights(
                self.k_p_params, self.k_proj_weight, gain=1.0
            )
            self.v_proj_weight = self._init_weights(
                self.v_p_params, self.v_proj_weight, gain=1.0
            )

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)

    @staticmethod
    def _init_weights(params: InProjParams, weights: torch.Tensor, gain: float):
        if params.small_init:
            return small_init_(weights, gain=gain)
        else:
            return xavier_uniform_(weights, gain=gain)

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
