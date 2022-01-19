# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from xformers.components import Activation
from xformers.components.feedforward import FEEDFORWARD_REGISTRY, build_feedforward
from xformers.components.feedforward.mixture_of_experts import GateConfig

BATCH = 4
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
        "number_of_experts": 4,  # MoE
        "gate_config": "top_2",  # MoE
    }

    # dummy, just check construction and dimensions in the FW pass
    ffw = build_feedforward(test_config)

    if ffw.requires_cuda and not device.type == "cuda":
        # pyre-fixme[29]: The library function `pytest.skip` is not supported by Pyre.
        pytest.skip("This MLP requires CUDA and current device does not match")

    inputs = torch.rand(BATCH, SEQ, LATENT, device=device)
    ffw = ffw.to(device)

    _ = ffw(inputs)


def get_expert():
    return torch.nn.Linear(LATENT, LATENT, bias=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="This test requires CUDA")
@pytest.mark.parametrize("gate", [g.value for g in GateConfig])
@pytest.mark.parametrize("number_of_local_experts", [None, 4])
@pytest.mark.parametrize("expert_constructor", [None, get_expert])
def test_moe(gate, number_of_local_experts, expert_constructor):
    test_config = {
        "name": "MixtureOfExperts",
        "dim_model": LATENT,
        "dropout": DROPOUT,
        "activation": Activation.ReLU,
        "hidden_layer_multiplier": 4,
        "number_of_experts": 4,
        "number_of_local_experts": number_of_local_experts,
        "gate_config": gate,
        "expert_constructor": expert_constructor,
    }

    # dummy, just check construction and dimensions in the FW pass
    ffw = build_feedforward(test_config)

    inputs = torch.rand(BATCH, SEQ, LATENT, device=torch.device("cuda"))
    ffw = ffw.to(torch.device("cuda"))

    outputs = ffw(inputs)
    loss = torch.sum(outputs)
    loss.backward()
