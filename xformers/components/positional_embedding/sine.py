# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


# Silence Mypy errors in this file.
# type: ignore

from __future__ import annotations

import math
from typing import TypeVar

import torch
from pyre_extensions import Generic
from typing_extensions import Literal as L

from xformers.components.positional_embedding import (
    PositionEmbedding,
    PositionEmbeddingConfig,
    register_positional_embedding,
)

N = TypeVar("N", bound=int)
DimModel = TypeVar("DimModel", bound=int)


@register_positional_embedding("sine", PositionEmbeddingConfig)
class SinePositionalEmbedding(PositionEmbedding, Generic[DimModel]):
    def __init__(self, dim_model: DimModel, *args, **kwargs):
        super().__init__()
        self.dim_model: DimModel = dim_model

    def forward(
        self, x: torch.Tensor[torch.float, N, N]
    ) -> torch.Tensor[torch.float, N, N, DimModel]:
        seq_len = x.shape[1]
        pos = (
            torch.arange(0, seq_len, device=x.device, dtype=torch.float32)
            .unsqueeze(1)
            .repeat(1, self.dim_model)
        )
        dim = (
            torch.arange(0, self.dim_model, device=x.device, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(seq_len, 1)
        )
        div = torch.exp(-math.log(10000) * (2 * (dim // 2) / self.dim_model))
        pos *= div
        pos[:, 0::2] = torch.sin(pos[:, 0::2])
        pos[:, 1::2] = torch.cos(pos[:, 1::2])

        # pyre-ignore[9]: x was declared as N x N but is expected as N * N * 1.
        # Handle a non-existing embedding dimension
        output: torch.Tensor[torch.float32, N, N, L[1]] = (
            x.unsqueeze(-1) if x.ndim == 2 else x
        )

        return output + pos.unsqueeze(0)
