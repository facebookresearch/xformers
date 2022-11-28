# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

from xformers.components import Activation
from xformers.components.feedforward import (
    Feedforward,
    FeedforwardConfig,
    register_feedforward,
)

logger = logging.getLogger("xformers")


if torch.cuda.is_available():
    try:
        from xformers.triton import FusedDropoutBias

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
                    nn.Linear(
                        in_features=dim_model, out_features=dim_mlp, bias=False
                    ),  # bias is handled in the next layer
                    # pyre-ignore[16]: TODO(T101400990): Pyre did not recognize
                    # the `FusedLinear` import.
                    FusedDropoutBias(
                        p=dropout,
                        bias_shape=dim_mlp if bias else None,
                        activation=activation,
                    ),
                    nn.Linear(
                        in_features=dim_mlp, out_features=dim_model, bias=False
                    ),  # bias is handled in the next layer
                    # pyre-ignore[16]: TODO(T101400990): Pyre did not recognize
                    # the `FusedLinear` import.
                    FusedDropoutBias(
                        p=dropout,
                        bias_shape=dim_model if bias else None,
                        activation=None,
                    ),
                )
                self.requires_cuda = True

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return self.mlp(inputs)

    except ImportError:
        logger.warning("Triton is not available, FusedMLP will not be enabled.")
