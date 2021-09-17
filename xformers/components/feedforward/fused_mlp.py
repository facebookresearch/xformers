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

_use_triton = torch.cuda.is_available()
if _use_triton:
    try:
        from xformers.triton.fused_linear_layer import FusedLinear
    except ImportError:
        logging.warning("Triton is not available, FusedMLP will not be enabled.")
        _use_triton = False


"""
A MLP using fused linear layers + activation
"""

if _use_triton:

    @dataclass
    class FusedMlpConfig(FeedforwardConfig):
        hidden_layer_multiplier: int

    @register_feedforward("FusedMLP", FusedMlpConfig)
    class FusedMLP(Feedforward):
        def __init__(
            self,
            dim_model: int,
            dropout: float,
            activation: Activation,
            hidden_layer_multiplier: int,
            *args,
            **kwargs
        ):
            super().__init__()

            self.mlp = nn.Sequential(
                FusedLinear(dim_model, hidden_layer_multiplier * dim_model, activation),
                nn.Dropout(dropout),
                nn.Linear(hidden_layer_multiplier * dim_model, dim_model),
                nn.Dropout(dropout),
            )
            self.requires_cuda = True

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.mlp(inputs)
