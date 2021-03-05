from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class AttentionConfig(dict):
    name: str
    n_heads: int
    dim_in: int
    dim_out: int
    attention_dropout: float
    residual_dropout: float
    causal: bool


# Define the common interface, every attention block needs to derive from it
class Attention(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def __init__(
        self,
        dim_in: Optional[int] = None,
        dim_out: Optional[int] = None,
        attention_dropout: Optional[float] = None,
        residual_dropout: Optional[float] = None,
        n_heads: Optional[int] = None,
        causal: Optional[bool] = None,
        *args,
        **kwargs
    ):
        super().__init__()

    @classmethod
    def from_config(cls, config: AttentionConfig) -> "Attention":
        return cls(
            config.dim_in,
            config.dim_out,
            config.attention_dropout,
            config.residual_dropout,
            config.n_heads,
            config.causal,
        )

    @staticmethod
    def generate_mask(size: int):
        # FIXME
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask
