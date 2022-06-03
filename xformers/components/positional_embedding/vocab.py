# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from xformers.components.positional_embedding import (
    PositionEmbedding,
    PositionEmbeddingConfig,
    register_positional_embedding,
)


@dataclass
class VocabEmbeddingConfig(PositionEmbeddingConfig):
    vocab_size: int
    dropout: float


@register_positional_embedding("vocab", VocabEmbeddingConfig)
class VocabEmbedding(PositionEmbedding):
    def __init__(
        self,
        dim_model: int,
        seq_len: int,
        vocab_size: int,
        dropout: float = 0.0,
        *args,
        **kwargs
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim_model = dim_model

        self.dropout = torch.nn.Dropout(p=dropout)
        self.position_embeddings = nn.Embedding(seq_len, self.dim_model)
        self.word_embeddings = nn.Embedding(self.vocab_size, self.dim_model)

        self.position_ids: Optional[torch.Tensor] = None

        self.init_weights()

    def init_weights(self, gain: float = 1.0):
        torch.nn.init.normal_(self.position_embeddings.weight, std=0.02 * gain)
        torch.nn.init.normal_(self.word_embeddings.weight, std=0.02 * gain)

    def forward(self, x: torch.Tensor):
        position_ids = torch.arange(x.shape[1], dtype=torch.long, device=x.device)[
            None, :
        ].repeat(x.shape[0], 1)

        X_token = self.word_embeddings(x)
        X_pos = self.position_embeddings(position_ids)

        X = X_token + X_pos
        X = self.dropout(X)

        return X
