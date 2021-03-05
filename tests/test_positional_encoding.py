import pytest
import torch

from xformers.components.positional_encoding import (
    POSITION_ENCODING_REGISTRY,
    PositionEncodingConfig,
    build_positional_encoding,
)

BATCH = 20
SEQ = 512
MODEL = 384

assert (
    POSITION_ENCODING_REGISTRY.keys()
), "Positional encoding layers should have been registered"


@pytest.mark.parametrize("encoding_name", POSITION_ENCODING_REGISTRY.keys())
def test_construction(encoding_name: str):
    test_config = {"name": encoding_name, "dim_model": MODEL, "seq_len": SEQ}

    # dummy, just check construction and dimensions in the FW pass
    encoding = build_positional_encoding(PositionEncodingConfig(**test_config))

    inputs = torch.rand(BATCH, SEQ, MODEL)
    _ = encoding(inputs)
