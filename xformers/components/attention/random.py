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
    random_pattern,
)
from xformers.components.attention.core import scaled_dot_product_attention


@dataclass
class RandomAttentionConfig(AttentionConfig):
    r: Optional[
        float
    ]  # the ratio of keys that the query can attend to. 1.0 means dense attention
    constant_masking: Optional[
        bool
    ]  # whether the randomness is per query or defined at construction time
    force_sparsity: Optional[bool]  # use sparsity in any case (potentially slower)


@register_attention("random", RandomAttentionConfig)
class RandomAttention(Attention):
    def __init__(
        self,
        dropout: float,
        causal: bool = False,
        r: float = 0.01,
        constant_masking: bool = True,
        force_sparsity: bool = False,
        *args,
        **kwargs,
    ):
        """
        "Random" attention, as proposed for instance in BigBird_.
        Random means in that case that each query can attend to a random set of keys.
        This implementation is sparse-aware, meaning that the empty attention parts will not be represented in memory.

        Args:
            r (float): the ratio in [0,1] of keys that the query can attend to
            constant_masking (bool): if true, keep the same random set for all queries.

        .. _BigBird: https://arxiv.org/pdf/2007.14062.pdf

        """
        super().__init__()

        self.attn_drop = nn.Dropout(dropout, inplace=False)
        self.causal = causal
        self.r = r
        self.rand_attention_mask: Optional[torch.Tensor] = None
        self.constant_masking = constant_masking
        self.force_sparsity = force_sparsity

        # Properties specific to this attention mechanism
        self.supports_attention_mask = True
        self.supports_key_padding_mask = False

        self.requires_same_k_q_dimensions = True

    def _get_rand_mask(self, shape: torch.Size) -> torch.Tensor:
        sparsity = 1 - self.r
        mask = random_pattern(shape[1], sparsity=sparsity)

        if self.causal:
            mask &= causal_1d_pattern(shape[1])

        mask = sparsify(mask) if self.force_sparsity else maybe_sparsify(mask)

        return mask

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_mask: Optional[Union[torch.Tensor, AttentionMask]] = None,
        *args,
        **kwargs,
    ):
        # Rand masking
        if not self.constant_masking or self.rand_attention_mask is None:
            self.rand_attention_mask = self._get_rand_mask(q.shape).to(q.device)

        # Mask-aware attention
        if att_mask is not None:
            if att_mask.dtype == torch.bool and isinstance(
                self.rand_attention_mask, AttentionMask
            ):
                mask = self.rand_attention_mask + AttentionMask.from_bool(att_mask)
            else:
                if isinstance(att_mask, AttentionMask):
                    # Needed because & op not defined for SparseCS with AttentionMask
                    att_mask = att_mask.to_bool()
                mask = self.rand_attention_mask & att_mask
        else:
            mask = self.rand_attention_mask

        # Handle q/k/v which would not fit the mask
        seq_len = q.shape[-2]
        q_, k_, v_ = map(lambda x: self._maybe_pad_sequence(x, mask), (q, k, v))

        # Normal attention with the random mask
        att = scaled_dot_product_attention(
            q=q_, k=k_, v=v_, att_mask=mask, dropout=self.attn_drop
        )

        # Take into account an hypothetical padding
        return att[:, :seq_len, :]
