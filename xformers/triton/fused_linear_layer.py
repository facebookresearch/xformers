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
from xformers.triton.k_activations import get_triton_activation_index
from xformers.triton.k_fused_matmul_bw import fused_matmul_backward
from xformers.triton.k_fused_matmul_fw import fused_matmul


class _fused_linear_triton(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx,
        x,
        weight,
        bias,
        activation,
        trainable_weight,
        trainable_bias,
        save_activation_inputs,
    ):

        # Kick the fused Triton kernel, handling bias and activation in one go
        y, activation_inputs = fused_matmul(
            x, weight, bias, activation, save_activation_inputs
        )

        ctx.activation = activation
        ctx.trainable_weight = trainable_weight
        ctx.trainable_bias = trainable_bias

        # Micro-optimization: saving these is not always needed (?)
        if x.requires_grad or ctx.trainable_weight or ctx.trainable_bias:
            ctx.save_for_backward(weight, activation_inputs, x)

        return y

    @staticmethod
    @custom_bwd
    def backward(
        ctx: Any, grad_out: torch.Tensor
    ) -> Any:  # pragma: no cover  # this is covered, but called directly from C++
        """
        Compute the derivative with respect to x, other tensors were not trainable inputs.
        """
        (weight, activation_inputs, x) = ctx.saved_tensors

        grad_input, grad_weight, grad_bias = fused_matmul_backward(
            grad_out=grad_out,
            inputs=x,
            act_in=activation_inputs,
            weight=weight,
            trainable_weight=ctx.trainable_weight,
            trainable_bias=ctx.trainable_bias,
            activation_grad=ctx.activation,
        )

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


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

        self._activation_index = get_triton_activation_index(activation)
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
            self._activation_index,
            self.weight.requires_grad,
            self.bias.requires_grad if self.bias is not None else False,
            self.training and x.requires_grad and self._activation_index > 0,
        )
