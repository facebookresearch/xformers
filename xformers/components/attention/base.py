# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from abc import ABCMeta, abstractmethod
from dataclasses import asdict, dataclass
from typing import Optional, Type, TypeVar

import torch
import torch.nn as nn

from xformers.components.attention import AttentionMask


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

    _causal_mask: Optional[AttentionMask] = None

    @abstractmethod
    def __init__(self, dropout: Optional[float] = None, *args, **kwargs):
        super().__init__()

        # Requires the inputs to be projected
        self.requires_input_projection = True

        # Whether the head dimension needs to be present (if not it can be folded into the batch dimension)
        self.requires_head_dimension = False

        # key padding mask and attention mask must be passed in as separate arguments instead of a merged attention mask
        self.requires_separate_masks = False

        # Requires that K and Q have the same sequence length
        self.requires_same_k_q_dimensions = False

        # Whether the attention owns the single head/multihead mechanism
        # so that the MHA wrapper should skip it
        self.requires_skip_multi_head = False

        # This attention requires a context length which is squared, often due to 2D pooling
        self.requires_squared_context = False

        # Whether this attention mechanism supports attention masks
        self.supports_attention_mask = True
        self.supports_key_padding_mask = False

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

    @staticmethod
    def _maybe_pad_sequence(x: torch.Tensor, mask: torch.Tensor):
        """
        If the sequence is shorter than the mask, return a padded view
        """
        if x.shape[-2] != mask.shape[-1]:
            assert x.shape[-2] < mask.shape[-1], (
                "Sequence is bigger than the provided mask, cannot infer what to do with it."
                " Please update your attention mask"
            )

            pad_size = (0, 0, 0, mask.shape[-1] - x.shape[-2], 0, 0)
            return torch.nn.functional.pad(x, pad_size, mode="constant", value=0.0)

        return x
