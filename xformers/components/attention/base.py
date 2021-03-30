from abc import ABCMeta, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from attrdict import AttrDict


class AttentionConfig(AttrDict):
    name: str  # the registered name for this attention mechanism
    from_seq_dim: int  # the dimension of the input sequence
    to_seq_dim: Optional[int]  # the (optional) dimension of the output sequence
    dropout: float  # dropout probability
    causal: bool  # apply a causal mask


# Define the common interface, every attention block needs to derive from it
class Attention(nn.Module, metaclass=ABCMeta):
    r"""The base Attention mechanism, which is typically a sub-part of the multi-head attention"""

    _causal_mask: Optional[torch.Tensor] = None

    @abstractmethod
    def __init__(
        self,
        dropout: Optional[float] = None,
        causal: Optional[bool] = None,
        from_seq_dim: Optional[int] = None,
        to_seq_dim: Optional[int] = None,
        *args,
        **kwargs
    ):
        super().__init__()

    @classmethod
    def from_config(cls, config: AttentionConfig) -> "Attention":
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

    def _get_causal_mask(self, from_seq_dim: int, to_seq_dim: int) -> torch.Tensor:
        # Cache a mask so that multiple instances would reuse the same
        if (
            not self._causal_mask
            or self._causal_mask.shape[0] != from_seq_dim
            or self._causal_mask.shape[1] != to_seq_dim
        ):
            self._causal_mask = torch.tril(
                torch.ones(from_seq_dim, to_seq_dim), diagonal=0
            )
            self._causal_mask[self._causal_mask == 1] = -float("inf")
            self._causal_mask.unsqueeze_(0)  # batch dimension
        return self._causal_mask
