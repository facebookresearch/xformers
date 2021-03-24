from abc import ABCMeta, abstractmethod
from typing import Optional

import torch.nn as nn
from attrdict import AttrDict


class PositionEncodingConfig(AttrDict):
    name: str
    dim_model: int
    seq_len: int


class PositionEncoding(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def __init__(
        self,
        dim_model: Optional[int] = None,
        seq_len: Optional[int] = None,
        *args,
        **kwargs
    ) -> None:
        super().__init__()

    @classmethod
    def from_config(cls, config: PositionEncodingConfig) -> "PositionEncoding":
        return cls(**config)
