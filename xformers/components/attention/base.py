from abc import ABCMeta, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from attrdict import AttrDict


class AttentionConfig(AttrDict):
    name: str
    dropout: float
    causal: bool


# Define the common interface, every attention block needs to derive from it
class Attention(nn.Module, metaclass=ABCMeta):
    r"""The base Attention mechanism, which is typically a sub-part of the multi-head attention"""

    @abstractmethod
    def __init__(
        self,
        dropout: Optional[float] = None,
        causal: Optional[bool] = None,
        *args,
        **kwargs
    ):
        super().__init__()

    @classmethod
    def from_config(cls, config: AttentionConfig) -> "Attention":
        # NOTE: This will make sure that default values set in the constructors are used
        return cls(**config)

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def generate_mask(size: int) -> torch.Tensor:
        # FIXME
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask
