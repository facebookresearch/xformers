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
_kErf = 1 / math.sqrt(math.pi)
_k2Erf = 2 / math.sqrt(math.pi)
_k1OverSqrt2 = 1 / math.sqrt(2.0)


def get_triton_activation_kernel(activation: Optional[Activation]):
    return (
        {
            Activation.ReLU: relu,
            Activation.LeakyReLU: leaky_relu,
            Activation.GeLU: gelu_quick,
        }[activation]
        if activation
        else None
    )


def get_triton_activation_bwd_kernel(activation: Optional[Activation]):
    return (
        {
            Activation.ReLU: relu_grad,
            Activation.LeakyReLU: leaky_relu_grad,
            Activation.GeLU: gelu_quick_grad,
        }[activation]
        if activation
        else None
    )


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def erf(x):
    # Let's use the Taylor series expansion for now
    res = 2 * x * _kErf

    x_acc = x * x * x  # x3
    res -= (2.0 / 3.0 * x_acc) * _kErf

    x_acc = x_acc * x * x  # x5
    res += x_acc / 5.0 * _kErf

    x_acc = x_acc * x * x  # x7
    res -= x_acc / 21.0 * _kErf

    x_acc = x_acc * x * x  # x9
    res += x_acc / 108.0 * _kErf

    # x_acc = x_acc * x * x  # x11
    # res -= x_acc / 660.0 * _kErf
    return res


@triton.jit
def erf_grad(x):
    return _k2Erf * tl.exp(-x * x)


@triton.jit
def cosh(x):
    exp_x = tl.exp(x)
    return (exp_x + 1.0 / exp_x) * 0.5


# a Triton implementation of the most used activations
# See for instance http://arxiv.org/abs/1606.08415 for an overview

# ReLU
@triton.jit
def relu(x):
    return tl.where(x >= 0, x, 0.0)


@triton.jit
def relu_grad(x):
    # NOTE: +0.0 are temporary hacks, to force Triton
    # to consider these as variables (else 'constant' and fp32 is assumed)
    return tl.where(x >= 0, 1.0 + 0.0, 0.0 + 0.0)


# Leaky ReLU
@triton.jit
def leaky_relu(x):
    scale = 0.01 + 0.0
    return tl.where(x >= 0, x, scale * x)


@triton.jit
def leaky_relu_grad(x):
    return tl.where(x >= 0, 1.0 + 0.0, 0.01 + 0.0)


# GeLU - Gaussian error linear unit (https://arxiv.org/pdf/1606.08415.pdf)


@triton.jit
def gelu_quick(x):
    x = x.to(tl.float32)
    return x * tl.sigmoid(1.702 * x)


@triton.jit
def gelu_quick_grad(x):
    # (e^(1.702 x) (1.702 x + e^(1.702 x) + 1))/(e^(1.702 x) + 1)^2
    x = x.to(tl.float32)
    _exp = tl.exp(x)
    _x = 1.702 * x
    _denom = (_exp + 1) * (_exp + 1)
    return _exp * (_x + _exp + 1.0) / _denom


@triton.jit
def gelu_accurate(x):
    x = x.to(tl.float32)
    return 0.5 * x * (1 + erf(x * _k1OverSqrt2))


@triton.jit
def gelu_accurate_grad(x):
    # Normal computation, just try to maximize reuse
    x_3 = x * x * x
    _a = 0.0356774 * x_3 + _kAlpha * x

    # (hoping that a division is cheaper than an exponential..)
    exp_a = tl.exp(_a)
    exp_m_a = 1.0 / exp_a

    _cos_h = (exp_a + exp_m_a) * 0.5
    _tan_h = (exp_a - exp_m_a) / (exp_a + exp_m_a)
    return 0.5 + 0.5 * _tan_h + (0.0535161 * x_3 + 0.398942 * x) / (_cos_h * _cos_h)
