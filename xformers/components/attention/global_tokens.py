# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from xformers.components.attention import (
    Attention,
    AttentionConfig,
    maybe_sparsify,
    register_attention,
    sparsify,
)
from xformers.components.attention.attention_patterns import (
    causal_1d_pattern,
    global_token_pattern,
)
from xformers.components.attention.core import scaled_dot_product_attention


@dataclass
class GlobalAttentionConfig(AttentionConfig):
    attention_query_mask: torch.Tensor  # Mark the queries which have global attention
    causal: Optional[bool]
    force_sparsity: Optional[bool]


@register_attention("global", GlobalAttentionConfig)
class GlobalAttention(Attention):
    def __init__(
        self,
        dropout: float,
        attention_query_mask: torch.Tensor,
        causal: bool = False,
        force_sparsity: bool = False,
        *_,
        **__,
    ):
        r"""
        Global attention, as proposed for instance in BigBird_ or Longformer_.

        Global means in that case that the queries positively labelled in the ```attention_query_mask``` can attend
        to all the other queries. The queries negatively labelled in the ```attention_query_mask``` cannot attend to
        any other query.

        This implementation is sparse-aware, meaning that the empty attention parts will not be represented in memory.

        Args:
            dropout (float): probability of an element to be zeroed
            attention_mask (torch.Tensor): if true, this query can attend to all the others

        """
        super().__init__()

        assert attention_query_mask.dtype == torch.bool, "A boolean mask is expected"
        assert (
            attention_query_mask.shape[1] == 1
            and attention_query_mask.shape[0] > attention_query_mask.shape[1]
        ), "A N x 1 query mask is expected"

        self.attn_drop = nn.Dropout(dropout, inplace=False)
        self.attention_mask = global_token_pattern(attention_query_mask[:, 0])
        self.force_sparsity = force_sparsity

        if causal:
            self.attention_mask &= causal_1d_pattern(attention_query_mask.shape[1])

        self.attention_mask = (
            sparsify(self.attention_mask)
            if self.force_sparsity
            else maybe_sparsify(self.attention_mask)
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_mask: Optional[torch.Tensor] = None,
        *_,
        **__,
    ):
        # Make sure that the mask is on the right device
        if self.attention_mask.device != q.device:
            self.attention_mask = self.attention_mask.to(q.device)

        # Mask-aware attention
        mask = (
            self.attention_mask if att_mask is None else self.attention_mask & att_mask
        )

        return scaled_dot_product_attention(
            q=q, k=k, v=v, att_mask=mask, dropout=self.attn_drop
        )
