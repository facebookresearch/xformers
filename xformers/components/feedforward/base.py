from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch.nn as nn

from xformers.utils import ExtensibleConfig


class Activations(str, Enum):
    GeLU = "gelu"
    ReLU = "relu"


@dataclass(init=False)
class FeedforwardConfig(ExtensibleConfig):
    name: str
    dim_latent: int
    dropout: float
    activation: Activations
    hidden_layer_multiplier: int


# Define the common interface, every feedforward block needs to derive from it
class Feedforward(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def __init__(
        self,
        dim_latent: Optional[int] = None,
        dropout: Optional[float] = None,
        activation: Optional[Activations] = None,
        *args,
        **kwargs
    ):
        super().__init__()

    @classmethod
    def from_config(cls, config: FeedforwardConfig) -> "Feedforward":
        return cls(**FeedforwardConfig.as_patchy_dict(config))
