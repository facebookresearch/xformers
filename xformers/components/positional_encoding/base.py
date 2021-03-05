from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch.nn as nn


@dataclass
class PositionEncodingConfig:
    name: str
    dim_model: int
    seq_len: int


class PositionEncoding(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def __init__(
        self, dim_model: Optional[int] = None, seq_len: Optional[int] = None
    ) -> None:
        super().__init__()

    @classmethod
    @abstractmethod
    def from_config(cls, config: PositionEncodingConfig) -> "PositionEncoding":
        raise NotImplementedError
