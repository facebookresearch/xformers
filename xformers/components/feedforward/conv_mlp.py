# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


# CREDITS: Largely reusing the code from the reference VAN implementation
# see https://github.com/Visual-Attention-Network

import math
from dataclasses import dataclass
from typing import Optional

import torch.nn as nn

from xformers.components import Activation, build_activation
from xformers.components.feedforward import Feedforward, FeedforwardConfig

from . import register_feedforward


@dataclass
class ConvMlpConfig(FeedforwardConfig):
    hidden_layer_multiplier: int
    dim_model: int
    dim_model_out: Optional[int]
    act_layer: Activation
    dropout: float


@register_feedforward("Conv2DFeedforward", ConvMlpConfig)
class Conv2DFeedforward(Feedforward):
    """
    A Convolutional feed-forward network, as proposed in VAN_ (Vision Attention Network, Guo et al.)

    .. _VAN: https://arxiv.org/pdf/2202.09741.pdf
    """

    def __init__(
        self,
        dim_model: int,
        hidden_layer_multiplier: int = 1,
        dim_model_out: Optional[int] = None,
        activation: Activation = Activation.GeLU,
        dropout=0.0,
        *args,
        **kwargs,
    ):
        super().__init__()
        out_features = dim_model_out or dim_model
        hidden_features = hidden_layer_multiplier * dim_model

        self.conv_mlp = nn.Sequential(
            nn.Conv2d(dim_model, hidden_features, 1),
            nn.Conv2d(
                hidden_features,
                hidden_features,
                3,
                1,
                1,
                bias=True,
                groups=hidden_features,
            ),
            build_activation(activation),
            nn.Conv2d(hidden_features, out_features, 1),
            nn.Dropout(dropout),
        )

        # This feedforward requires a context length which is squared, often due to 2D pooling
        self.requires_squared_context = True

    def init_weights(self, **kwargs):
        # Follow the original init, but also make it possible to initialize from the outside
        def init_module(m: nn.Module):
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()

        self.apply(init_module)

    def forward(self, x):
        # The conv layers expect NCHW, we have NLC by default
        B, L, C = x.shape
        HW = int(math.sqrt(x.shape[-2]))
        assert HW**2 == L, "Conv2DFeedforward requires squared context lengths"

        x = x.reshape((B, HW, HW, C)).swapdims(1, -1)

        # The actual FW, including the 2d convolutions
        x = self.conv_mlp(x)

        # back to NLC
        x = x.transpose(1, -1)
        return x.flatten(1, 2)
