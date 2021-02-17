import torch
import torch.nn as nn

from xformers.components.feedforward import Activations, Feedforward, FeedforwardConfig


class MLP(Feedforward):
    def __init__(
        self,
        dim_latent: int,
        dropout: float,
        activation: Activations,
        hidden_layer_multiplier: int,
    ):
        super().__init__()

        activation = {Activations.ReLU: nn.ReLU, Activations.GeLU: nn.GELU}[
            activation
        ]()

        self.mlp = nn.Sequential(
            nn.Linear(dim_latent, hidden_layer_multiplier * dim_latent),
            activation,
            nn.Linear(hidden_layer_multiplier * dim_latent, dim_latent),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.mlp(inputs)

    @classmethod
    def from_config(self, config: FeedforwardConfig) -> "MLP":
        # TODO: @lefaudeux
        pass
