# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import nn

from xformers.components.attention import (
    Attention,
    AttentionConfig,
    AttentionMask,
    register_attention,
)
from xformers.components.attention.core import scaled_dot_product_attention

logger = logging.getLogger("xformers")


@dataclass
class ScaledDotProductConfig(AttentionConfig):
    causal: Optional[bool]
    seq_len: Optional[int]
    to_seq_len: Optional[int]


@register_attention("scaled_dot_product", ScaledDotProductConfig)
class ScaledDotProduct(Attention):
    r"""
    Implementing the Scaled Dot-Product attention proposed in
    `Attention is all you need`_, Vaswani et al.

    .. _`Attention is all you need`: https://arxiv.org/abs/1706.03762v5
    """

    mask: Optional[AttentionMask]

    def __init__(
        self,
        dropout: float = 0.0,
        causal: bool = False,
        seq_len: Optional[int] = None,
        to_seq_len: Optional[int] = None,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.attn_drop = nn.Dropout(dropout, inplace=False)
        self.causal = causal
        self.seq_len = seq_len

        if causal and seq_len is not None:
            self.mask = AttentionMask.make_causal(seq_len, to_seq_len)
        else:
            self.mask = None

        # Properties specific to this attention mechanism
        self.supports_attention_mask = True
        self.supports_key_padding_mask = False

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_mask: Optional[Union[AttentionMask, torch.Tensor]] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        att_mask    A 2D or 3D mask which ignores attention at certain positions.

                    - If the mask is boolean, a value of True will keep the value,
                        while a value of False will mask the value.

                        Key padding masks (dimension: batch x sequence length) and attention masks
                        (dimension: sequence length x sequence length OR batch x sequence length x sequence length)
                        can be combined and passed in here. Method maybe_merge_masks provided in the utils can be
                        used for that merging.

                    - If the mask has the float type, then an additive mask is expected (masked values are -inf)

        """

        # Convenience, create an attention mask if a tensor was passed
        if att_mask is not None and isinstance(att_mask, torch.Tensor):
            # By default we don't know of the causality, and a check would be expensive
            att_mask = (
                AttentionMask.from_bool(att_mask)
                if att_mask.dtype == torch.bool
                else AttentionMask(att_mask, is_causal=False)
            )

        # Handle a possibly deferred causal mask handling
        mask = self.mask
        if self.causal and self.mask is None:
            mask = AttentionMask.make_causal(
                seq_len=q.shape[-2],
                to_seq_len=q.shape[-2],
                device=q.device,
                dtype=q.dtype,
            )

        # Merge the optional causal mask and the user-provided mask
        if mask is not None:
            mask = mask.to(dtype=q.dtype, device=q.device)

            att_mask = att_mask + mask if att_mask is not None else mask

        # Try to handle a case where the sequence is smaller than the mask
        if (
            att_mask is not None
            and q.shape[-2] == k.shape[-2]
            and q.shape[-2] < att_mask.shape[1]
        ):
            if isinstance(att_mask, AttentionMask):
                att_mask = att_mask.make_crop(seq_len=q.shape[-2])
            else:
                logger.error(
                    "Mismatching sparse attention mask and sequence length."
                    + " Please pad the inputs or adjust the attention mask"
                )
                raise NotImplementedError

        # Attend: (B x nh, S, hs) x (B x nh, hs, S) -> (B x nh, S, S)
        y = scaled_dot_product_attention(
            q=q, k=k, v=v, att_mask=att_mask, dropout=self.attn_drop
        )
        return y
