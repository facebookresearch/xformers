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


def get_triton_activation_kernel(activation: Optional[Activation]):
    return (
        {
            Activation.ReLU: relu,
            Activation.LeakyReLU: leaky_relu,
            Activation.GeLU: gelu,
            Activation.SquaredReLU: squared_relu,
        }[activation]
        if activation
        else None
    )


def get_triton_activation_bwd_kernel(activation: Optional[Activation]):
    return (
        {
            Activation.ReLU: relu_grad,
            Activation.LeakyReLU: leaky_relu_grad,
            Activation.GeLU: gelu_grad,
            Activation.SquaredReLU: squared_relu_grad,
        }[activation]
        if activation
        else None
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
    return tl.where(x >= 0, 1.0, 0.0)


@triton.jit
def squared_relu(x):
    """
    Squared ReLU activation, as proposed in the Primer_ paper.

    .. _Primer: https://arxiv.org/abs/2109.08668
    """
    x_ = relu(x)
    return x_ * x_


@triton.jit
def squared_relu_grad(x):
    return tl.where(x >= 0, 2.0 * x, 0.0)


# Leaky ReLU
@triton.jit
def leaky_relu(x):
    """
    LeakyReLU_ activation

    .. _LeakyReLU: https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html
    """
    scale = 0.01 + 0.0
    return tl.where(x >= 0, x, scale * x)


@triton.jit
def leaky_relu_grad(x):
    return tl.where(x >= 0, 1.0, 0.01)


@triton.jit
def gelu(x):
    """
    GeLU_ activation - Gaussian error linear unit

    .. _GeLU: https://arxiv.org/pdf/1606.08415.pdf
    """
    return 0.5 * x * (1 + tanh(_kAlpha * (x + 0.044715 * x * x * x)))


@triton.jit
def gelu_grad(x):
    # Normal computation, just try to maximize reuse
    x_3 = x * x * x
    _a = 0.0356774 * x_3 + _kAlpha * x

    # (hoping that a division is cheaper than an exponential..)
    exp_a = tl.exp(_a)
    exp_m_a = 1.0 / exp_a

    _cos_h = exp_a + exp_m_a
    _tan_h = (exp_a - exp_m_a) / _cos_h
    _cos_h *= 0.5
    return 0.5 + 0.5 * _tan_h + (0.0535161 * x_3 + 0.398942 * x) / (_cos_h * _cos_h)
