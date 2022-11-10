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
    local_1d_pattern,
)
from xformers.components.attention.core import scaled_dot_product_attention


@dataclass
class LocalAttentionConfig(AttentionConfig):
    causal: Optional[bool] = None
    window_size: Optional[int] = None
    force_sparsity: Optional[bool] = None


@register_attention("local", LocalAttentionConfig)
class LocalAttention(Attention):
    def __init__(
        self,
        dropout: float = 0.0,
        causal: bool = False,
        window_size: int = 5,
        force_sparsity: bool = False,
        *args,
        **kwargs,
    ):

        r"""
        An implementation of a sliding window attention, as proposed in RoutingTransformer_, LongFormer_ or BigBird_


        Args:
            dropout (float): the probability of an output to be randomly dropped at training time
            causal (bool): apply a causal mask, in that the attention cannot be applied to the future
            window_size (int): the overall window size for local attention.
                Odd number is expected if the mask is not causal, as the window size will be evenly
                distributed on both sides of each query


        .. _RoutingTransformer: https://arxiv.org/pdf/2003.05997.pdf

        .. _BigBird: https://arxiv.org/pdf/2007.14062.pdf

        .. _Longformer: https://arxiv.org/pdf/2004.05150.pdf

        """
        super().__init__()

        self.attn_drop = nn.Dropout(dropout, inplace=False)
        self.causal = causal
        self.force_sparsity = force_sparsity

        if not self.causal:
            assert (
                window_size % 2 == 1
            ), "The window size is assumed to be odd (counts self-attention + 2 wings)"

        self.window_size = window_size
        self.attention_mask: Optional[torch.Tensor] = None
        self.requires_same_k_q_dimensions = True

        # Properties specific to this attention mechanism
        self.supports_attention_mask = True
        self.supports_key_padding_mask = False

    def _get_local_mask(self, shape: torch.Size) -> torch.Tensor:
        window_size = self.window_size * 2 + 1 if self.causal else self.window_size
        mask = local_1d_pattern(shape[1], window_size)

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
        # Local window attention masking
        if self.attention_mask is None or self.attention_mask.shape[1] != q.shape[1]:
            self.attention_mask = self._get_local_mask(q.shape).to(q.device)

        # Take into account the optional user mask
        if att_mask is None:
            mask = self.attention_mask
        else:
            if isinstance(att_mask, AttentionMask):
                # Needed because & op not defined for SparseCS with AttentionMask
                att_mask = att_mask.to_bool()
            mask = self.attention_mask & att_mask

        return scaled_dot_product_attention(
            q=q, k=k, v=v, att_mask=mask, dropout=self.attn_drop
        )
