from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import torch.nn as nn

from xformers.utils import ExtensibleConfig


@dataclass(init=False)
class PositionEmbeddingConfig(ExtensibleConfig):
    name: str
    dim_model: int
    seq_len: int


class PositionEmbedding(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @classmethod
    def from_config(cls, config: PositionEmbeddingConfig) -> "PositionEmbedding":
        return cls(**PositionEmbeddingConfig.as_patchy_dict(config))
