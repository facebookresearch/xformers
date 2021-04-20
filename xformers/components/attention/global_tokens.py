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
class GlobalAttentionConfig(AttentionConfig):
    attention_query_mask: torch.Tensor  # Mark the queries which have global attention


@register_attention("global")
class GlobalAttention(Attention):
    def __init__(
        self, dropout: float, attention_query_mask: torch.Tensor, *args, **kwargs
    ):
        """
        "Global" attention, as proposed for instance in _BigBird or _Longformer.

        Global means in that case that the queries positively labelled in the `attention_query_mask` can attend
        to all the other queries. The queries negatively labelled in the `attention_query_mask`cannot attend to
        any other query.

        This implementation is sparse-aware, meaning that the empty attention parts will not be represented in memory.

        Args:
            dropout (float): probability of an element to be zeroed
            attention_mask (torch.Tensor): if true, this query can attend to all the others

        _BigBird: "Zaheer, M., Guruganesh, G., Dubey, A., Ainslie, J., Alberti, C., Ontanon, S., Pham, P., Ravula,
         A., Wang, Q., Yang, L., & Ahmed, A. (2020). Big Bird: Transformers for Longer Sequences"

        _Longformer: "Beltagy, I., Peters, M. E., & Cohan, A. (2020). Longformer: The Long-Document Transformer"
        """
        super().__init__()

        assert attention_query_mask.dtype == torch.bool, "A boolean mask is expected"
        assert (
            attention_query_mask.shape[1] == 1
            and attention_query_mask.shape[0] > attention_query_mask.shape[1]
        ), "A N x 1 query mask is expected"

        self.attn_drop = nn.Dropout(dropout, inplace=False)
        self.attention_mask = attention_query_mask | attention_query_mask.transpose(
            1, 0
        )

        # Sparsity threshold, below which having a sparse matrix is more efficient
        if (
            torch.count_nonzero(self.attention_mask) / self.attention_mask.numel()
            < _DENSITY_THRESHOLD
        ):
            self.attention_mask = self.attention_mask.to_sparse()

    def _adapt_mask_to_batch(self, q: torch.Tensor):
        # Make sure that the mask is on the right device, and has the right dimensions
        if self.attention_mask.device != q.device:
            self.attention_mask = self.attention_mask.to(q.device)

        # Handle the batch dimension without duplicating memory.
        # Only needed in the sparse case for now,the dense case broadcasts
        if self.attention_mask.ndim != q.ndim and self.attention_mask.is_sparse:
            # FIXME: @lefaudeux this takes space in memory and is not really useful all things considered
            self.attention_mask = (
                self.attention_mask.to_dense().expand(q.shape[0], -1, -1).to_sparse()
            )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_mask: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ):
        self._adapt_mask_to_batch(q)

        # Mask-aware attention
        mask = (
            self.attention_mask if att_mask is None else self.attention_mask & att_mask
        )

        return scaled_dot_product_attention(
            q, k, v, att_mask=mask, dropout=self.attn_drop
        )

    @classmethod
    def from_config(cls, config: AttentionConfig) -> "Attention":
        return cls(**GlobalAttentionConfig.as_patchy_dict(config))
