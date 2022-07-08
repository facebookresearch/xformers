# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
from functorch.compile import memory_efficient_fusion

from xformers.components import Activation, build_activation
from xformers.components.nvfuser import Fused, FusedConfig, register_fused


@dataclass
class FusedBiasActivationDropoutConfig(FusedConfig):
    p: float
    activation: Activation
    bias_shape: Optional[int]


def _fn(
    activation: Activation, prob: float, x: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    if bias is not None:
        x = torch.add(x, bias)
    y = activation(x)
    return torch.nn.functional.dropout(y, prob) if prob > 0.0 else y


@register_fused("fused_bias_activation_dropout", FusedBiasActivationDropoutConfig)
class FusedBiasActivationDropout(Fused):
    """
    A layer which fuses the computation of Dropout(Activation(x))
    with AOTAutograd and nvFuser
    """

    def __init__(
        self,
        p: float,
        activation: Activation,
        bias_shape: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.p = float(p)

        allowed_activations = [Activation.ReLU, Activation.GeLU]

        assert (
            self.p < 1.0
        ), f"We don't want to drop all the values, most probably p={self.p} is not properly set"

        assert activation in [
            Activation.ReLU,
            Activation.GeLU,
        ], f"Activation provided is not one of {allowed_activations}"

        self.activation = activation
        self.pytorch_activation = build_activation(self.activation)

        self.bias = (
            torch.zeros(bias_shape, requires_grad=True)
            if bias_shape is not None
            else None
        )

    def init_weights(self, *args, **kwargs):
        with torch.no_grad():
            if self.bias is not None:
                self.bias.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convenience, catch a possible type or device mismatch
        if self.bias is not None:
            self.bias = self.bias.to(dtype=x.dtype, device=x.device)  # type: ignore

        # Train/inference
        p = self.p if self.training else 0.0

        # Catch a non-cuda setup, fallback to pytorch
        if not x.is_cuda or p == 0.0:
            x = x + self.bias if self.bias is not None else x
            x = self.pytorch_activation(x)
            return torch.nn.functional.dropout(x, p) if p > 0.0 else x

        # AOTAutograd, NVFuser backed path
        aot_fn = memory_efficient_fusion(fn=_fn, static_argnums=(0, 1))
        return aot_fn(self.pytorch_activation, p, x, self.bias)
