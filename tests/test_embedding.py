# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from xformers.components import PatchEmbeddingConfig, build_patch_embedding
from xformers.components.positional_embedding import (
    POSITION_EMBEDDING_REGISTRY,
    build_positional_embedding,
)

BATCH = 20
SEQ = 512
MODEL = 384
assert (
    POSITION_EMBEDDING_REGISTRY.keys()
), "Positional encoding layers should have been registered"


@pytest.mark.parametrize("encoding_name", POSITION_EMBEDDING_REGISTRY.keys())
@pytest.mark.parametrize("dropout", [0.0, 0.2])
def test_dimensions(encoding_name: str, dropout: float):
    test_config = {
        "name": encoding_name,
        "dim_model": MODEL,
        "vocab_size": 32,
        "dropout": dropout,
        "seq_len": SEQ,
    }

    # dummy, just check construction and dimensions in the FW pass
    encoding = build_positional_embedding(test_config)
    inputs = (torch.rand(BATCH, SEQ) * 10).abs().to(torch.int)
    _ = encoding(inputs)

    # Test that inputs having an embedding dimension would also work out
    if "name" == "sine":
        inputs = (torch.rand(BATCH, SEQ, MODEL) * 10).abs().to(torch.int)
        _ = encoding(inputs)


def test_patch_embedding():
    patch_embedding_config = {
        "in_channels": 3,
        "out_channels": 64,
        "kernel_size": 7,
        "stride": 4,
        "padding": 2,
    }

    # dummy, just check construction and dimensions in the FW pass
    patch_emb = build_patch_embedding(PatchEmbeddingConfig(**patch_embedding_config))

    # Check BHWC
    inputs = torch.rand(BATCH, 32 * 32, 3)
    out = patch_emb(inputs)
    assert out.shape[-1] == 64

    # Check BCHW
    inputs = torch.rand(BATCH, 3, 32, 32)
    out = patch_emb(inputs)
    assert out.shape[-1] == 64
