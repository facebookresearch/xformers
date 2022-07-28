# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import Optional

import torch
import torch.nn as nn
from functorch.compile import memory_efficient_fusion


def _fn(
    x: torch.Tensor,
    bias: Optional[torch.nn.parameter.Parameter],
    prob: float,
    residual: torch.Tensor,
) -> torch.Tensor:
    a = torch.add(x, bias) if bias is not None else x
    b = torch.nn.functional.dropout(a, prob) if prob > 0.0 else a
    return torch.add(b, residual)


class NVFusedBiasDropoutRes(torch.nn.Module):
    """
    A layer which fuses the computation of Dropout(x + Bias) + Residual
    with AOTAutograd and nvFuser
    """

    def __init__(
        self,
        p: float,
        bias_shape: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.p = float(p)
        self.requires_residual = True

        self.bias = (
            nn.Parameter(torch.zeros(bias_shape)) if bias_shape is not None else None
        )

        assert (
            self.p < 1.0
        ), f"We don't want to drop all the values, most probably p={self.p} is not properly set"

    def init_weights(self, *args, **kwargs):
        with torch.no_grad():
            if self.bias is not None:
                self.bias.fill_(0.0)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        # Train/inference
        p = self.p if self.training else 0.0

        # Catch a non-cuda setup, fallback to pytorch
        if not x.is_cuda:
            return _fn(x, self.bias, p, residual)

        # AOTAutograd, NVFuser backed path
        aot_fn = memory_efficient_fusion(fn=_fn, static_argnums=(2))
        return aot_fn(x, self.bias, p, residual)
