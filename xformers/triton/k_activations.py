# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional

import triton
import triton.language as tl

from xformers.components import Activation

_kAlpha = math.sqrt(2.0 / math.pi)


def get_triton_activation_index(activation: Optional[Activation]) -> int:
    return (
        {
            Activation.ReLU: 1,
            Activation.LeakyReLU: 2,
            Activation.GeLU: 3,
            Activation.SquaredReLU: 4,
            Activation.SmeLU: 5,
            Activation.StarReLU: 6,
        }[activation]
        if activation is not None
        else 0
    )


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def cosh(x):
    exp_x = tl.exp(x)
    return (exp_x + 1.0 / exp_x) * 0.5


# a Triton implementation of the most used activations
# See for instance http://arxiv.org/abs/1606.08415 for an overview

# ReLU
@triton.jit
def relu(x):
    """
    ReLU_ activation function

    .. _ReLU: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
    """
    return tl.where(x >= 0, x, 0.0)


@triton.jit
def relu_grad(x):
    # ReLU is different from other activations
    # in that it does not require the input to retrospectively compute its gradient
    # here the input is the downstream gradient, and we return the upstream gradient directly
    return tl.where(x >= 0, 1.0, 0.0)


@triton.jit
def squared_relu(x):
    """
    Squared ReLU activation, as proposed in the Primer_ paper.

    .. _Primer: https://arxiv.org/abs/2109.08668
    """
    x_sq = x * x
    return tl.where(x > 0.0, x_sq, 0.0)


@triton.jit
def squared_relu_grad(x):
    return tl.where(x >= 0.0, 2 * x, 0.0)


@triton.jit
def star_relu(x):
    """
    Star ReLU activation, as proposed in the "MetaFormer Baselines for Vision"_ paper.

    .. _ "MetaFormer Baselines for Vision": https://arxiv.org/pdf/2210.13452.pdf
    """
    x_sq = x * x
    return 0.8944 * tl.where(x > 0.0, x_sq, 0.0) - 0.4472


@triton.jit
def star_relu_grad(x):
    return tl.where(x >= 0.0, 1.7888 * x, 0.0)


# Leaky ReLU
@triton.jit
def leaky_relu(x):
    """
    LeakyReLU_ activation

    .. _LeakyReLU: https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html
    """
    return tl.where(x >= 0.0, x, 0.01 * x)


@triton.jit
def leaky_relu_grad(x):
    return tl.where(x >= 0.0, 1.0, 0.01)


@triton.jit
def gelu(x):
    """
    GeLU_ activation - Gaussian error linear unit

    .. _GeLU: https://arxiv.org/pdf/1606.08415.pdf
    """
    return 0.5 * x * (1 + tanh(_kAlpha * (x + 0.044715 * x * x * x)))


@triton.jit
def gelu_grad(x):
    # CREDITS: Fast implementation proposed in
    # https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/fused_bias_gelu.py#L30
    tanh_out = tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    return 0.5 * x * (
        (1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)
    ) + 0.5 * (1 + tanh_out)


@triton.jit
def smelu(x):
    """
    SmeLU_ activation -  Smooth ReLU with beta=2.0

    .. _SmeLU: https://arxiv.org/pdf/2202.06499.pdf
    """
    beta = 2.0

    relu = tl.where(x >= beta, x, 0.0)
    return tl.where(tl.abs(x) <= beta, (x + beta) * (x + beta) / (4.0 * beta), relu)


@triton.jit
def smelu_grad(x):
    beta = 2.0

    relu_grad = tl.where(x >= beta, 1.0, 0.0)
    return tl.where(tl.abs(x) <= beta, (beta + x) / (2.0 * beta), relu_grad)
