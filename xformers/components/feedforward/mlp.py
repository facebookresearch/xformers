# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from collections import OrderedDict
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
        self.bias = bias
        self.dim_model = dim_model
        self.dropout = dropout
        self.activation = activation
        self.dim_mlp = hidden_layer_multiplier * dim_model
        # check if fused Bias Activation Dropout is applicable
        if xformers._is_functorch_available:

            # Catch unimported fused layer
            from xformers.components.nvfuser.bias_act_dropout import (  # noqa
                NVFusedBiasActivationDropout,
            )

            self.requires_cuda = True
            self.mlp = nn.Sequential(
                nn.Linear(
                    in_features=dim_model, out_features=self.dim_mlp, bias=False
                ),  # bias is handled in the next layer
                NVFusedBiasActivationDropout(
                    p=dropout,
                    bias_shape=self.dim_mlp if bias else None,
                    activation=activation,
                ),
                nn.Linear(
                    in_features=self.dim_mlp, out_features=dim_model, bias=False
                ),  # bias is handled in the next layer
                NVFusedBiasActivationDropout(
                    p=dropout,
                    bias_shape=dim_model if bias else None,
                    activation=None,
                ),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(in_features=dim_model, out_features=self.dim_mlp, bias=bias),
                build_activation(activation),
                nn.Dropout(dropout),
                nn.Linear(in_features=self.dim_mlp, out_features=dim_model, bias=bias),
                nn.Dropout(dropout),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.mlp(inputs)

    def toFused(self):
        if not xformers._is_functorch_available:
            pass
        # Catch unimported fused layer
        from xformers.components.nvfuser.bias_act_dropout import (  # noqa
            NVFusedBiasActivationDropout,
        )

        mlp_weights = self.mlp.state_dict()
        rev_weights = OrderedDict(
            [
                k.replace(".3.", ".2.") if ".3." in k else (k, v)
                for k, v in mlp_weights.items()
            ]
        )
        self.requires_cuda = True
        self.mlp = nn.Sequential(
            nn.Linear(
                in_features=self.dim_model, out_features=self.dim_mlp, bias=False
            ),  # bias is handled in the next layer
            NVFusedBiasActivationDropout(
                p=self.dropout,
                bias_shape=self.dim_mlp if self.bias else None,
                activation=self.activation,
            ),
            nn.Linear(
                in_features=self.dim_model, out_features=self.dim_model, bias=False
            ),  # bias is handled in the next layer
            NVFusedBiasActivationDropout(
                p=self.dropout,
                bias_shape=self.dim_model if self.bias else None,
                activation=None,
            ),
        )
        self.mlp.cuda()
        self.mlp.load_state_dict(rev_weights)
