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

if torch.cuda.is_available():
    try:
        from xformers.triton import FusedLinear

        @dataclass
        class FusedMlpConfig(FeedforwardConfig):
            hidden_layer_multiplier: int

        @register_feedforward("FusedMLP", FusedMlpConfig)
        class FusedMLP(Feedforward):
            """
            A MLP using fused linear layers.

            .. warning: This is not currently competitive with PyTorch in terms of training speed
            """

            def __init__(
                self,
                dim_model: int,
                dropout: float,
                activation: Activation,
                hidden_layer_multiplier: int,
                *args,
                **kwargs,
            ):
                super().__init__()

                self.mlp = nn.Sequential(
                    # pyre-ignore[16]: TODO(T101400990): Pyre did not recognize
                    # the `FusedLinear` import.
                    FusedLinear(
                        in_features=dim_model,
                        out_features=hidden_layer_multiplier * dim_model,
                        activation=activation,
                        bias=True,
                    ),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_layer_multiplier * dim_model, dim_model),
                    nn.Dropout(dropout),
                )
                self.requires_cuda = True

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return self.mlp(inputs)

    except ImportError:
        logging.warning("Triton is not available, FusedMLP will not be enabled.")
