from abc import ABCMeta, abstractmethod
from dataclasses import asdict, dataclass
from typing import Optional

import torch.nn as nn

from xformers.components import Activation


@dataclass
class FeedforwardConfig:
    name: str
    dim_model: int
    dropout: float
    activation: Activation


# Define the common interface, every feedforward block needs to derive from it
class Feedforward(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def __init__(
        self,
        dim_model: Optional[int] = None,
        dropout: Optional[float] = None,
        activation: Optional[Activation] = None,
        *args,
        **kwargs,
    ):
        super().__init__()

    @classmethod
    def from_config(cls, config: FeedforwardConfig) -> "Feedforward":
        # Generate the class inputs from the config
        fields = asdict(config)

        # Skip all Nones so that default values are used
        fields = {k: v for k, v in fields.items() if v is not None}

        return cls(**fields)
