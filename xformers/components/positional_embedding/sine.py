# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import math

import torch

from xformers.components.positional_embedding import (
    PositionEmbedding,
    PositionEmbeddingConfig,
    register_positional_embedding,
)


@register_positional_embedding("sine", PositionEmbeddingConfig)
class SinePositionalEmbedding(PositionEmbedding):
    def __init__(self, dim_model: int, *args, **kwargs):
        super().__init__()
        self.dim_model = dim_model

    def forward(self, x: torch.Tensor):
        seq_len = x.shape[1]
        pos = (
            torch.arange(0.0, seq_len, device=x.device)
            .unsqueeze(1)
            .repeat(1, self.dim_model)
        )
        dim = (
            torch.arange(0.0, self.dim_model, device=x.device)
            .unsqueeze(0)
            .repeat(seq_len, 1)
        )
        div = torch.exp(-math.log(10000) * (2 * (dim // 2) / self.dim_model))
        pos *= div
        pos[:, 0::2] = torch.sin(pos[:, 0::2])
        pos[:, 1::2] = torch.cos(pos[:, 1::2])

        if x.ndim == 2:
            # Handle a non-existing embedding dimension
            x = x.unsqueeze(-1)

        return x + pos.unsqueeze(0)
