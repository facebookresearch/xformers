import pytest
import torch

from xformers.components import Activation
from xformers.components.feedforward import (
    FEEDFORWARD_REGISTRY,
    FeedforwardConfig,
    build_feedforward,
)

BATCH = 20
SEQ = 512
EMBD = 16
LATENT = 128
DROPOUT = 0.5


assert FEEDFORWARD_REGISTRY.keys(), "Feedforward layers should have been registered"


@pytest.mark.parametrize("feedforward_name", FEEDFORWARD_REGISTRY.keys())
@pytest.mark.parametrize("activation", [a.value for a in Activation])
def test_feedforward(feedforward_name: str, activation: Activation):
    test_config = {
        "name": feedforward_name,
        "dim_model": LATENT,
        "dropout": DROPOUT,
        "activation": activation,
        "hidden_layer_multiplier": 4,
    }

    # dummy, just check construction and dimensions in the FW pass
    ffw = build_feedforward(FeedforwardConfig(**test_config))

    inputs = torch.rand(BATCH, SEQ, LATENT)
    _ = ffw(inputs)
