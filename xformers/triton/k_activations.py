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
            Activation.SmeLU: smelu,
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
            Activation.SmeLU: smelu_grad,
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
    zero = 0.0
    return tl.where(x >= 0, x, zero.to(x.dtype))


@triton.jit
def relu_grad(x):
    # ReLU is different from other activations
    # in that it does not require the input to retrospectively compute its gradient
    # here the input is the downstream gradient, and we return the upstream gradient directly
    zero = 0.0
    one = 1.0
    return tl.where(x >= 0, one.to(x.dtype), zero.to(x.dtype))


@triton.jit
def squared_relu(x):
    """
    Squared ReLU activation, as proposed in the Primer_ paper.

    .. _Primer: https://arxiv.org/abs/2109.08668
    """
    x_ = relu(x)
    return (x_ * x_).to(x.dtype)


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
    scale = scale.to(x.dtype)
    return tl.where(x >= 0, x, scale * x)


@triton.jit
def leaky_relu_grad(x):
    min_grad = 0.01
    max_grad = 1

    min_grad = min_grad.to(x.dtype)
    max_grad = max_grad.to(x.dtype)

    return tl.where(x >= 0, max_grad, min_grad)


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
    zero = 0.0
    four = 4.0
    two = 2.0
    beta = two.to(x.dtype)

    output = (x + beta) * (x + beta) / (four.to(x.dtype) * beta)
    relu = tl.where(x >= beta, x, zero.to(x.dtype))
    return tl.where(tl.abs(x) <= beta, output, relu)


@triton.jit
def smelu_grad(x):
    zero = 0.0
    one = 1.0
    two = 2.0
    beta = two.to(x.dtype)

    grad = (beta + x) / (two.to(x.dtype) * beta)
    relu_grad = tl.where(x >= beta, one.to(x.dtype), zero.to(x.dtype))
    return tl.where(tl.abs(x) <= beta, grad, relu_grad)
