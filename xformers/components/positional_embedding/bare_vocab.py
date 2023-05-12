# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass

import torch.nn

from xformers.components.positional_embedding import (
    PositionEmbedding,
    register_positional_embedding,
)


@dataclass
class VocabEmbeddingConfig:
    name: str
    dim_model: int
    vocab_size: int
    dropout: float
    init_std: float


@register_positional_embedding("bare_vocab", VocabEmbeddingConfig)
class BareVocabEmbedding(PositionEmbedding):
    """Vocabulary embedding without positional information. Required for ALiBi-like positioning."""

    def __init__(
        self,
        dim_model: int,
        vocab_size: int,
        dropout: float = 0.0,
        init_std: float = 0.02,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim_model = dim_model
        self.init_std = init_std

        self.dropout = torch.nn.Dropout(p=dropout)
        self.word_embeddings = torch.nn.Embedding(self.vocab_size, self.dim_model)

        self.init_weights()

    def init_weights(self, gain: float = 1.0):
        torch.nn.init.normal_(self.word_embeddings.weight, std=self.init_std * gain)

    def forward(self, x: torch.Tensor):
        y = self.dropout(self.word_embeddings(x))
        return y
