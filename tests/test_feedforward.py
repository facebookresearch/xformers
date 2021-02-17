import pytest
import torch

from xformers.components.feedforward import Activations, Feedforward
from xformers.components.feedforward.mlp import MLP

BATCH = 20
SEQ = 512
EMBD = 16
LATENT = 128
DROPOUT = 0.5

feedforwards = [MLP]  # TODO: list these automatically


@pytest.mark.parametrize("feedforward_class", feedforwards)
@pytest.mark.parametrize("activation", [a.value for a in Activations])
def test_feedforward(feedforward_class: Feedforward, activation: Activations):
    test_config = {
        "dim_latent": LATENT,
        "dropout": DROPOUT,
        "activation": activation,
        "hidden_layer_multiplier": 4,
    }

    # dummy, just check construction and dimensions in the FW pass
    ffw = feedforward_class(**test_config)

    inputs = torch.rand(BATCH, SEQ, LATENT)
    _ = ffw(inputs)
