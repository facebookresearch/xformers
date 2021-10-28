# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from xformers.components import Activation
from xformers.components.feedforward import FEEDFORWARD_REGISTRY, build_feedforward

BATCH = 20
SEQ = 512
EMBD = 16
LATENT = 128
DROPOUT = 0.5

DEVICES = (
    [torch.device("cpu")] if not torch.cuda.is_available() else [torch.device("cuda")]
)

assert FEEDFORWARD_REGISTRY.keys(), "Feedforward layers should have been registered"


@pytest.mark.parametrize("feedforward_name", FEEDFORWARD_REGISTRY.keys())
@pytest.mark.parametrize("activation", [a.value for a in Activation])
@pytest.mark.parametrize("device", DEVICES)
def test_feedforward(
    feedforward_name: str, activation: Activation, device: torch.device
):
    test_config = {
        "name": feedforward_name,
        "dim_model": LATENT,
        "dropout": DROPOUT,
        "activation": activation,
        "hidden_layer_multiplier": 4,
    }

    # dummy, just check construction and dimensions in the FW pass
    ffw = build_feedforward(test_config)

    if ffw.requires_cuda and not device.type == "cuda":
        # pyre-fixme[29]: The library function `pytest.skip` is not supported by Pyre.
        pytest.skip("This MLP requires CUDA and current device does not match")

    inputs = torch.rand(BATCH, SEQ, LATENT, device=device)
    ffw = ffw.to(device)

    _ = ffw(inputs)
