# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from xformers.components.attention import Attention, AttentionConfig, register_attention


@dataclass
class VisualAttentionConfig(AttentionConfig):
    dim_model: int  # dimension of the input sequence


class LKA(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3
        )
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


@register_attention("visual", VisualAttentionConfig)
class Visual(Attention):
    def __init__(
        self,
        dim_model: int,
        *_,
        **__,
    ):
        """
        Large kernel attention mechanism, as proposed in `Visual Attention Network`_, Guo et al (2022).
        The original notation is tentatively kept as is. See https://github.com/Visual-Attention-Network
        for the reference implementation

        .. Note: compared to the paper, this block contains the LKA (Large Kernel Attention)
            and the prior and posterior transformations (Conv2d and activation)

        .. _`Visual Attention Network` : https://arxiv.org/pdf/2202.09741.pdf
        """
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(dim_model, dim_model, 1),
            nn.GELU(),
            LKA(dim_model),
            nn.Conv2d(dim_model, dim_model, 1),
        )

        # MHA related flags:
        self.requires_same_k_q_dimensions = (
            True  # This mechanism only really supports self attention
        )
        self.supports_attention_mask = False
        self.requires_skip_multi_head = (
            True  # This mechanism skips the multihead attention altogether
        )
        self.requires_squared_context = (
            True  # Recovering the 2D structure from context assumes squared content
        )

        self.requires_input_projection = (
            False  # This mechanism does not require that the MHA projects inputs
        )

    def forward(self, q: torch.Tensor, *_, **__):
        # Expose the 2D token structure
        B, HW, C = q.shape
        H = int(math.sqrt(HW))
        assert H * H == HW

        x = q.transpose(-2, -1).reshape(B, C, H, H)

        # Large kernel attention
        residual = x.clone()
        x = self.block(x)
        x = x + residual

        # Get back to B HW C
        return x.flatten(2, 3).transpose(-2, -1)
