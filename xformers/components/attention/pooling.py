# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from xformers.components.attention import Attention, AttentionConfig, register_attention


@dataclass
class PoolingAttentionConfig(AttentionConfig):
    pool_size: int  # dimension of the input sequence
    stride: Optional[int]  # dimension of the internal space
    padding: Optional[int]


@register_attention("pooling", PoolingAttentionConfig)
class Pooling(Attention):
    def __init__(
        self,
        pool_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        *_,
        **__,
    ):
        """
        Pooling token mixing mechanism, as proposed in
        `Metaformer is actually what you need for vision`_, Yu et al (2021).

        The original notation is kept as is.

        .. _`Metaformer is actually what you need for vision` : https://arxiv.org/pdf/2111.11418v1.pdf
        """
        super().__init__()

        padding = padding if padding is not None else pool_size // 2
        self.pool = nn.AvgPool2d(
            pool_size,
            stride=stride,
            padding=pool_size // 2,
            count_include_pad=False,
        )

        # MHA related flags:
        # kq need to have the same dimension
        self.requires_same_k_q_dimensions = False

        # This attention does not support attention masks
        self.supports_attention_mask = False

        # This "attention" (token mixing) skips the multihead attention altogether
        self.requires_skip_multi_head = True
        self.requires_input_projection = False

        # This operator does not really handle q,k,v
        self.requires_same_k_q_dimensions = True

        # This attention requires the 2d structure out of the context,
        # implictly assumed to be a squared length
        self.requires_squared_context = True

    def forward(self, q: torch.Tensor, *_, **__):
        # Expose the 2D token structure
        B, HW, C = q.shape
        H = int(math.sqrt(HW))
        assert H * H == HW

        q = q.transpose(-2, -1).reshape(B, C, H, H)

        # 2D pool
        x_pool = self.pool(q) - q  # compensate for the residual path

        # Get back to B HW C
        return x_pool.flatten(2, 3).transpose(-2, -1)
