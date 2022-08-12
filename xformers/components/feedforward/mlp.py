# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass

import torch
import torch.nn as nn

import xformers
from xformers.components import Activation, build_activation
from xformers.components.feedforward import Feedforward, FeedforwardConfig

if xformers._is_functorch_available:
    from xformers.components.nvfuser import (  # noqa
        NVFusedBiasActivationDropout,
    )

from . import register_feedforward


class LinearCustom(nn.Linear):
    """ """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        contain_bias: bool = True,
        use_bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, contain_bias, device, dtype)
        self.contain_bias = contain_bias
        self.use_bias = use_bias

        if self.use_bias:
            assert self.contain_bias

    def forward(self, input: torch.Tensor):
        if self.use_bias:
            return nn.functional.linear(input, self.weight, self.bias)
        else:
            return nn.functional.linear(input, self.weight, None)


@dataclass
class MlpConfig(FeedforwardConfig):
    hidden_layer_multiplier: int
    bias: bool


@register_feedforward("MLP", MlpConfig)
class MLP(Feedforward):
    def __init__(
        self,
        dim_model: int,
        dropout: float,
        activation: Activation,
        hidden_layer_multiplier: int,
        bias: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dropout = dropout
        self.activation = activation
        dim_mlp = hidden_layer_multiplier * dim_model

        self.functorch_mlp = False

        self.LL_1 = LinearCustom(
            in_features=dim_model,
            out_features=dim_mlp,
            contain_bias=bias,
            use_bias=bias,
        )
        self.LL_2 = LinearCustom(
            in_features=dim_mlp,
            out_features=dim_model,
            contain_bias=bias,
            use_bias=bias,
        )
        # check if functorch is applicable
        if xformers._is_functorch_available:
            self.init_functorch()
        else:
            self.BAD_1 = nn.Sequential(
                build_activation(activation), nn.Dropout(dropout)
            )
            self.BAD_2 = nn.Dropout(dropout)

    def init_functorch(self):
        # Catch unimported fused layer
        from xformers.components.nvfuser.bias_act_dropout import (  # noqa
            NVFusedBiasActivationDropout,
        )

        self.requires_cuda = True
        self.functorch_mlp = True

        self.LL_1.use_bias = False
        self.LL_2.use_bias = False
        self.BAD_1 = NVFusedBiasActivationDropout(
            p=self.dropout,
            activation=self.activation,
        )
        self.BAD_2 = NVFusedBiasActivationDropout(
            p=self.dropout,
            activation=None,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if xformers._is_functorch_available and not self.functorch_mlp:
            self.init_functorch()

        res = self.LL_1(inputs)
        res = self.BAD_1(res, self.LL_1.bias) if self.functorch_mlp else self.BAD_1(res)
        res = self.LL_2(res)
        res = self.BAD_2(res, self.LL_2.bias) if self.functorch_mlp else self.BAD_2(res)
        return res
