# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass

import torch

from xformers.components.positional_embedding import (
    PositionEmbedding,
    PositionEmbeddingConfig,
    register_positional_embedding,
)


@dataclass
class LearnablePositionalEmbeddingConfig(PositionEmbeddingConfig):
    name: str
    seq_len: int
    dim_model: int
    add_class_token: bool


@register_positional_embedding("learnable", LearnablePositionalEmbeddingConfig)
class LearnablePositionalEmbedding(PositionEmbedding):
    def __init__(
        self, seq_len: int, dim_model: int, add_class_token: bool = False, *_, **__
    ):
        super().__init__()

        # 0.02 is BERT initialization
        self.pos_emb = torch.nn.Parameter(
            torch.randn(1, seq_len + int(add_class_token), dim_model) * 0.02
        )

        self.class_token = (
            torch.nn.Parameter(torch.zeros(dim_model)) if add_class_token else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.class_token is not None:
            # Prepend class token
            clf_token = (
                torch.ones(x.shape[0], 1, self.pos_emb.shape[-1], device=x.device)
                * self.class_token
            )
            x = torch.cat([clf_token, x], dim=1)

        if x.ndim == 2:
            x = x.unsqueeze(-1)

        return x + self.pos_emb
