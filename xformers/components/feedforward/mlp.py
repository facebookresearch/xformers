from dataclasses import dataclass

import torch
import torch.nn as nn

from xformers.components import Activation
from xformers.components.feedforward import Feedforward, FeedforwardConfig

from . import register_feedforward


@dataclass(init=False)
class MlpConfig(FeedforwardConfig):
    hidden_layer_multiplier: int


@register_feedforward("MLP")
class MLP(Feedforward):
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

        activation_layer: nn.Module = {
            Activation.ReLU: nn.ReLU,
            Activation.GeLU: nn.GELU,
        }[activation]()

        self.mlp = nn.Sequential(
            nn.Linear(dim_model, hidden_layer_multiplier * dim_model),
            activation_layer,
            nn.Linear(hidden_layer_multiplier * dim_model, dim_model),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.mlp(inputs)

    @classmethod
    def from_config(cls, config: FeedforwardConfig) -> "Feedforward":
        return cls(**MlpConfig.as_patchy_dict(config))
