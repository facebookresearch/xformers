from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from xformers.components.attention import (
    Attention,
    AttentionConfig,
    maybe_sparsify,
    register_attention,
)
from xformers.components.attention.attention_patterns import (
    causal_1d_pattern,
    global_token_pattern,
)
from xformers.components.attention.core import scaled_dot_product_attention


@dataclass(init=False)
class GlobalAttentionConfig(AttentionConfig):
    attention_query_mask: torch.Tensor  # Mark the queries which have global attention


@register_attention("global")
class GlobalAttention(Attention):
    def __init__(
        self,
        dropout: float,
        attention_query_mask: torch.Tensor,
        causal: bool = False,
        *args,
        **kwargs
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
        self.attention_mask = global_token_pattern(attention_query_mask[:, 0])

        if causal:
            self.attention_mask &= causal_1d_pattern(attention_query_mask.shape[1])

        self.attention_mask = maybe_sparsify(self.attention_mask)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_mask: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ):
        # Make sure that the mask is on the right device
        if self.attention_mask.device != q.device:
            self.attention_mask = self.attention_mask.to(q.device)

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
