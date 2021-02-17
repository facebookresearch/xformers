import pytest
import torch

from xformers.components.positional_encoding import PositionEncoding
from xformers.components.positional_encoding.sine import SinePositionEncoding

BATCH = 20
SEQ = 512
EMBD = 16

encodings = [SinePositionEncoding]  # TODO: list these automatically


@pytest.mark.parametrize("encoding_class", encodings)
def test_construction(encoding_class: PositionEncoding):
    test_config = {"dim_embd": EMBD, "seq_len": SEQ}

    # dummy, just check construction and dimensions in the FW pass
    encoding = encoding_class(**test_config)

    inputs = torch.rand(BATCH, SEQ, EMBD)
    _ = encoding(inputs)
