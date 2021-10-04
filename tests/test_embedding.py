# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

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
