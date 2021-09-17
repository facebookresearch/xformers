# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import custom_bwd, custom_fwd

from xformers.components.activations import Activation, requires_bwd_inputs
from xformers.triton.activations import (
    get_triton_activation_bwd_kernel,
    get_triton_activation_kernel,
)
from xformers.triton.fused_matmul import fused_matmul
from xformers.triton.fused_matmul_backward import fused_matmul_backward


class _fused_linear_triton(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, x, weight, bias, activation, a_grad, save_activation_inputs):
        ctx.activation_grad = a_grad

        # Kick the fused Triton kernel, handling bias and activation in one go
        y, extra_outputs = fused_matmul(
            x, weight, bias, activation, save_activation_inputs
        )
        ctx.save_for_backward(weight, extra_outputs)
        ctx.save_activation_inputs = save_activation_inputs

        return y

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, grad_out) -> Any:  # type: ignore
        """
        Compute the derivative with respect to x, other tensors were not trainable inputs.
        """
        (weight, extra_outputs) = ctx.saved_tensors

        # Kick the fused Triton kernel, handling transpose and activation gradient in one go
        grad_input = fused_matmul_backward(
            grad_out,
            weight,
            extra_outputs,
            ctx.activation_grad,
            ctx.save_activation_inputs,
        )

        return grad_input, None, None, None, None, None


class FusedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        activation: Optional[Activation] = None,
        **kwargs
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        self._activation_kernel = get_triton_activation_kernel(activation)
        self._activation_grad_kernel = get_triton_activation_bwd_kernel(activation)
        self._save_activation_inputs = (
            activation in requires_bwd_inputs if activation is not None else False
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        Fused linear layer implementation, using the Triton programming model.

        y = activation(x * weight + bias)
        """

        # Minor perf boost: don't save inputs if we're only doing inference
        save_activation_inputs = self._save_activation_inputs and x.requires_grad

        return _fused_linear_triton.apply(
            x,
            self.weight,
            self.bias,
            self._activation_kernel,
            self._activation_grad_kernel,
            save_activation_inputs,
        )
