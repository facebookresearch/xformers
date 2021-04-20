from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from xformers.components.attention import (
    _DENSITY_THRESHOLD,
    Attention,
    AttentionConfig,
    register_attention,
)
from xformers.components.attention.core import scaled_dot_product_attention


@dataclass(init=False)
class RandomAttentionConfig(AttentionConfig):
    r: float  # the ratio of keys that the query can attend to. 1.0 means dense attention
    constant_masking: bool  # whether the randomness is per query or defined at construction time


@register_attention("random")
class RandomAttention(Attention):
    def __init__(
        self,
        dropout: float,
        r: float = 0.01,
        constant_masking: bool = True,
        *args,
        **kwargs
    ):
        """
        "Random" attention, as proposed for instance in _BigBird.
        Random means in that case means that each query can attend to a random set of keys.
        This implementation is sparse-aware, meaning that the empty attention parts will not be represented in memory.

        Args:
            r (float): the ratio in [0,1] of keys that the query can attend to
            constant_masking (bool): if true, keep the same random set for all queries.

        _BigBird: "Zaheer, M., Guruganesh, G., Dubey, A., Ainslie, J., Alberti, C., Ontanon, S., Pham, P., Ravula,
         A., Wang, Q., Yang, L., & Ahmed, A. (2020). Big Bird: Transformers for Longer Sequences. ArXiv, NeurIPS."
        """
        super().__init__()

        self.attn_drop = nn.Dropout(dropout, inplace=True)

        self.r = r
        self.rand_mask: Optional[torch.Tensor] = None
        self.constant_masking = constant_masking

    def _get_rand_mask(self, shape: torch.Size) -> torch.Tensor:
        mask = torch.FloatTensor(shape[1], shape[1]).uniform_() < self.r
        mask = mask.unsqueeze(0).expand(
            shape[0], -1, -1
        )  # duplicate the mask over the batch dimension

        # Sparsity threshold, below that having a sparse matrix is more efficient
        if self.r < _DENSITY_THRESHOLD:
            mask = mask.to_sparse()

        return mask

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_mask: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ):

        # Rand masking
        if not self.constant_masking or self.rand_mask is None:
            self.rand_mask = self._get_rand_mask(q.shape).to(q.device)

        # Mask-aware attention
        mask = self.rand_mask if att_mask is None else self.rand_mask & att_mask

        return scaled_dot_product_attention(q, k, v, mask, dropout=self.attn_drop)

    @classmethod
    def from_config(cls, config: AttentionConfig) -> "Attention":
        return cls(**RandomAttentionConfig.as_patchy_dict(config))
