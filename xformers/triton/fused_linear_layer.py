# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import custom_bwd, custom_fwd

from xformers.components.activations import Activation
from xformers.triton.activations import (
    get_triton_activation_bwd_kernel,
    get_triton_activation_kernel,
)
from xformers.triton.k_fused_matmul import fused_matmul, fused_matmul_backward

# The following activations require their inputs to be saved to be able to compute their gradients
_requires_bwd_inputs = [
    Activation.GeLU,
    Activation.SquaredReLU,
]


class _fused_linear_triton(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx,
        x,
        weight,
        bias,
        activation,
        act_grad_kernel,
        save_activation_inputs,
        trainable_weight,
        trainable_bias,
    ):

        # Kick the fused Triton kernel, handling bias and activation in one go
        y, activation_inputs = fused_matmul(
            x, weight, bias, activation, save_activation_inputs
        )

        ctx.activation_grad_kernel = act_grad_kernel
        ctx.trainable_weight = trainable_weight
        ctx.trainable_bias = trainable_bias
        ctx.save_activation_inputs = save_activation_inputs

        # Micro-optimization: saving these is not always needed (?)
        if x.requires_grad or ctx.trainable_weight or ctx.trainable_bias:
            if ctx.trainable_weight:
                ctx.save_for_backward(weight, activation_inputs, x)
            else:
                ctx.save_for_backward(weight, None, None)

        return y

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, grad_out: torch.Tensor) -> Any:  # type: ignore
        """
        Compute the derivative with respect to x, other tensors were not trainable inputs.
        """
        (weight, activation_inputs, inputs) = ctx.saved_tensors

        # Kick the fused Triton kernel, handling transpose and activation gradient in one go
        grad_input, grad_weight, grad_bias = fused_matmul_backward(
            grad_out=grad_out,
            inputs=inputs,
            weight=weight,
            trainable_weight=ctx.trainable_weight,
            trainable_bias=ctx.trainable_bias,
            activation_inputs=activation_inputs,
            activation_grad=ctx.activation_grad_kernel,
            activation_grad_req_inputs=ctx.save_activation_inputs,
        )

        return grad_input, grad_weight, grad_bias, None, None, None, None, None


class FusedLinear(nn.Module):
    """
    Handle a linear transform, like torch.nn.Linear_, and a given activation, in a single kernel.
    The whole transform: is :math:`y = activation(xA^T + b)`.

    This is typically significantly faster than PyTorch while using fp16 and non-sigmoid activations,
    as of September 2021.

    .. _torch.nn.Linear: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        activation: Optional[Activation] = None,
        **_,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=True
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features), requires_grad=True)
            if bias
            else None
        )

        self._activation_kernel = get_triton_activation_kernel(activation)
        self._activation_grad_kernel = get_triton_activation_bwd_kernel(activation)
        self._save_activation_inputs = (
            activation in _requires_bwd_inputs if activation is not None else False
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return _fused_linear_triton.apply(
            x,
            self.weight,
            self.bias,
            self._activation_kernel,
            self._activation_grad_kernel,
            self._save_activation_inputs,
            self.weight.requires_grad,
            self.bias.requires_grad if self.bias is not None else False,
        )
