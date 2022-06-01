# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from abc import ABCMeta, abstractmethod
from dataclasses import asdict, dataclass
from typing import Optional, Type, TypeVar

import torch.nn as nn

from xformers.components import Activation

Self = TypeVar("Self", bound="Feedforward")


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

        # This feedforward requires a CUDA accelerator
        self.requires_cuda = False

        # This feedforward requires a context length which is squared, often due to 2D pooling
        self.requires_squared_context = False

    @classmethod
    def from_config(cls: Type[Self], config: FeedforwardConfig) -> Self:
        # Generate the class inputs from the config
        fields = asdict(config)

        # Skip all Nones so that default values are used
        fields = {k: v for k, v in fields.items() if v is not None}

        return cls(**fields)
