# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import logging
from typing import Any, Optional

import torch
import torch.nn as nn
from functorch.compile import memory_efficient_fusion

from xformers.components import Activation, build_activation


def _fn(
    x: torch.Tensor, bias: torch.Tensor, activation: Activation, prob: float
) -> torch.Tensor:
    if bias is not None:
        x = torch.add(x, bias)
    y = activation(x)
    return torch.nn.functional.dropout(y, prob) if prob > 0.0 else y


class NVFusedBiasActivationDropout(torch.nn.Module):
    """
    A layer which fuses the computation of Dropout(Activation(x) + Bias)
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

        self.bias_shape = bias_shape
        self.bias = None

    def init_weights(self, *args, **kwargs):
        with torch.no_grad():
            if self.bias is not None:
                self.bias.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Lazy creation of learnable bias, to match input device
        if self.bias_shape is not None and self.bias is None:
            nn.Parameter(torch.zeros(self.bias_shape, device=x.device))

        # Train/inference
        p = self.p if self.training else 0.0

        # Catch a non-cuda setup, fallback to pytorch
        if not x.is_cuda or p == 0.0:
            # TODO just call _fn??????
            x = x + self.bias if self.bias is not None else x
            x = self.pytorch_activation(x)
            return torch.nn.functional.dropout(x, p) if p > 0.0 else x

        # AOTAutograd, NVFuser backed path
        aot_fn = memory_efficient_fusion(_fn, static_argnums=(2, 3))
        return aot_fn(x, self.bias, self.pytorch_activation, p)
