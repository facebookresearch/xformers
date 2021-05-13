import pytest
import torch

from xformers.components.positional_embedding import (
    POSITION_EMBEDDING_REGISTRY,
    PositionEmbeddingConfig,
    build_positional_embedding,
)

BATCH = 20
SEQ = 512

assert (
    POSITION_EMBEDDING_REGISTRY.keys()
), "Positional encoding layers should have been registered"


@pytest.mark.parametrize("encoding_name", POSITION_EMBEDDING_REGISTRY.keys())
@pytest.mark.parametrize("dropout", [0.0, 0.2])
def test_dimensions(encoding_name: str, dropout: float):
    test_config = {
        "name": encoding_name,
        "dim_model": 384,
        "vocab_size": 32,
        "dropout": dropout,
        "max_sequence_len": SEQ,
    }

    # dummy, just check construction and dimensions in the FW pass
    encoding = build_positional_embedding(PositionEmbeddingConfig(**test_config))
    inputs = (torch.rand(BATCH, SEQ) * 10).abs().to(torch.int)
    _ = encoding(inputs)
