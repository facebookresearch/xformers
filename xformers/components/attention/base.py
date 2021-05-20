from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from xformers.utils import ExtensibleConfig


@dataclass(init=False)
class AttentionConfig(ExtensibleConfig):
    """Parameters required for all Attentions.
    Can accept and store extra parameters.
    """

    name: str  # the registered name for this attention mechanism
    dropout: float  # dropout probability


# Define the common interface, every attention block needs to derive from it
class Attention(nn.Module, metaclass=ABCMeta):
    r"""The base Attention mechanism, which is typically a sub-part of the multi-head attention"""

    _causal_mask: Optional[torch.Tensor] = None

    @abstractmethod
    def __init__(self, dropout: Optional[float] = None, *args, **kwargs):
        super().__init__()

    @classmethod
    def from_config(cls, config: AttentionConfig) -> "Attention":
        return cls(**AttentionConfig.as_patchy_dict(config))

    @abstractmethod
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError

    def _get_causal_mask(self, seq_len: int, to_seq_len: int) -> torch.Tensor:
        # Cache a mask so that multiple instances would reuse the same
        if not self._causal_mask:
            self._causal_mask = torch.tril(torch.ones(seq_len, to_seq_len), diagonal=0)
            self._causal_mask[self._causal_mask == 1] = -float("inf")
            self._causal_mask.unsqueeze_(0)  # batch dimension

        return self._causal_mask
