# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from abc import ABCMeta, abstractmethod
from dataclasses import asdict, dataclass
from typing import Optional, Type, TypeVar

import torch
import torch.nn as nn


@dataclass
class AttentionConfig:
    """Parameters required for all Attentions.
    Can accept and store extra parameters.
    """

    name: str  # the registered name for this attention mechanism
    dropout: float  # dropout probability


Self = TypeVar("Self", bound="Attention")


# Define the common interface, every attention block needs to derive from it
class Attention(nn.Module, metaclass=ABCMeta):
    r"""The base Attention mechanism, which is typically a sub-part of the multi-head attention"""

    _causal_mask: Optional[torch.Tensor] = None

    @abstractmethod
    def __init__(self, dropout: Optional[float] = None, *args, **kwargs):
        super().__init__()
        self.requires_input_projection = True
        self.requires_head_dimension = False
        self.accepts_att_mask = True
        # key padding mask and attention mask must be passed in as separate arguments instead of a merged attention mask
        self.requires_separate_masks = False

    @classmethod
    def from_config(cls: Type[Self], config: AttentionConfig) -> Self:
        # Generate the class inputs from the config
        fields = asdict(config)

        # Skip all Nones so that default values are used
        fields = {k: v for k, v in fields.items() if v is not None}

        return cls(**fields)

    @abstractmethod
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError

    def _get_causal_mask(self, seq_len: int, to_seq_len: int) -> torch.Tensor:
        # Cache a mask so that multiple instances would reuse the same
        causal_mask = self._causal_mask
        if not causal_mask:
            causal_mask = torch.tril(torch.ones(seq_len, to_seq_len), diagonal=0)
            causal_mask[self._causal_mask == 1] = -float("inf")
            causal_mask.unsqueeze_(0)  # batch dimension
            self._causal_mask = causal_mask

        return causal_mask
