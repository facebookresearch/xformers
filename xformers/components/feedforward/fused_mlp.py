# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

from xformers import _is_triton_available
from xformers.components import Activation
from xformers.components.feedforward import (
    Feedforward,
    FeedforwardConfig,
    register_feedforward,
)

if _is_triton_available:
    try:
        from xformers.triton import FusedLinear

        @dataclass
        class FusedMlpConfig(FeedforwardConfig):
            hidden_layer_multiplier: int

        @register_feedforward("FusedMLP", FusedMlpConfig)
        class FusedMLP(Feedforward):
            """
            A MLP using fused linear layers.
            """

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

                dim_mlp = hidden_layer_multiplier * dim_model

                self.mlp = nn.Sequential(
                    FusedLinear(
                        in_features=dim_model,
                        out_features=dim_mlp,
                        bias=bias,
                        activation=activation,
                    ),
                    torch.nn.Dropout(p=dropout),
                    FusedLinear(
                        in_features=dim_mlp,
                        out_features=dim_model,
                        bias=bias,
                        activation=None,
                    ),
                    torch.nn.Dropout(p=dropout),
                )

                self.requires_cuda = True

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return self.mlp(inputs)

    except ImportError:
        logging.warning("Triton is not available, FusedMLP will not be enabled.")
