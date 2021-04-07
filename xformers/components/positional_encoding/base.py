from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch.nn as nn

from xformers.utils import ExtensibleConfig


@dataclass(init=False)
class PositionEncodingConfig(ExtensibleConfig):
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
        return cls(**PositionEncodingConfig.as_patchy_dict(config))
