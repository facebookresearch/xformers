from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from xformers.components.positional_embedding import (
    PositionEmbedding,
    PositionEmbeddingConfig,
    register_positional_embedding,
)


@dataclass(init=False)
class VocabEmbeddingConfig(PositionEmbeddingConfig):
    vocab_size: int
    dropout: float


@register_positional_embedding("vocab")
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
        torch.nn.init.normal_(self.position_embeddings.weight, std=0.02)

        self.position_ids: Optional[torch.Tensor] = None

        self.word_embeddings = nn.Embedding(self.vocab_size, self.dim_model)
        torch.nn.init.normal_(self.word_embeddings.weight, std=0.02)

    def forward(self, x: torch.Tensor):
        position_ids = torch.arange(x.shape[1], dtype=torch.long, device=x.device)[
            None, :
        ].repeat(x.shape[0], 1)

        X_token = self.word_embeddings(x)
        X_pos = self.position_embeddings(position_ids)

        X = X_token + X_pos
        X = self.dropout(X)

        return X

    @classmethod
    def from_config(cls, config: PositionEmbeddingConfig) -> "PositionEmbedding":
        return cls(**VocabEmbeddingConfig.as_patchy_dict(config))
