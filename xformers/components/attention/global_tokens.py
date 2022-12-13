# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn

from xformers.components.attention import (
    Attention,
    AttentionConfig,
    AttentionMask,
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
            attention_query_mask (torch.Tensor): if true, this query can attend to all the others

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

        # Properties specific to this attention mechanism
        self.requires_same_k_q_dimensions = True
        self.supports_attention_mask = False
        self.supports_key_padding_mask = False

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_mask: Optional[Union[torch.Tensor, AttentionMask]] = None,
        *_,
        **__,
    ):
        # Make sure that the mask is on the right device
        if self.attention_mask.device != q.device:
            self.attention_mask = self.attention_mask.to(q.device)

        # Mask-aware attention
        if att_mask is not None:
            if att_mask.dtype == torch.bool and isinstance(
                self.attention_mask, AttentionMask
            ):
                if not isinstance(att_mask, AttentionMask):
                    att_mask = AttentionMask.from_bool(att_mask)
                mask = self.attention_mask + att_mask
            else:
                mask = self.attention_mask & att_mask
        else:
            mask = self.attention_mask

        # Handle q/k/v which would not fit the mask
        seq_len = q.shape[-2]
        q_, k_, v_ = map(lambda x: self._maybe_pad_sequence(x, mask), (q, k, v))

        # Normal attention with the global tokens mask
        att = scaled_dot_product_attention(
            q=q_, k=k_, v=v_, att_mask=mask, dropout=self.attn_drop
        )

        # Take into account an hypothetical padding
        return att[:, :seq_len, :]
