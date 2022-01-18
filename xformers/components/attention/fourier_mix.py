# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch

from xformers.components.attention import Attention, AttentionConfig, register_attention


@register_attention("fourier_mix", AttentionConfig)
class FourierMix(Attention):
    def __init__(self, *_, **__):
        """
        FFT-based pseudo-attention mechanism, from
        "
        "FNet: Mixing Tokens with Fourier Transforms"
        Lee-Thorp et al., 2021, https://arxiv.org/pdf/2105.03824.pdf
        """
        super().__init__()
        self.requires_input_projection = False

    def forward(self, q: torch.Tensor, *_, **__):
        return torch.fft.fft2(q).real
