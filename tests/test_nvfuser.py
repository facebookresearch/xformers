# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch
import torch.nn as nn
from torch.cuda.amp.autocast_mode import autocast

from xformers import _is_functorch_available
from xformers.components import Activation, LayerNormStyle
from xformers.components.feedforward import build_feedforward

_gpu_available = torch.cuda.is_available()

if _is_functorch_available:
    from xformers.components.nvfuser import (
        NVFusedBiasActivationDropout,
        NVFusedBiasDropoutRes,
        NVFusedBiasDropoutResLayerNorm,
    )
    from xformers.components.nvfuser.utils import build_nvfused

# Testing odd (non-power-of-two for instance) shapes on purpose
SHAPES = [
    (384, 512),
    (8, 384, 128),
    (8, 784, 512),
    (4, 16, 384),
    (4, 16, 1024),
    (2, 16, 2048),
    (2, 16, 4096),
    (1, 16, 12288),
]

BATCH = 4
SEQ = 256
EMBD = 16
LATENT = 128
DEVICES = [torch.device("cuda")]


@pytest.mark.skipif(not _gpu_available, reason="GPU is not available")
@pytest.mark.skipif(not _is_functorch_available, reason="Functorch is not available")
@pytest.mark.parametrize(
    "fused_pattern",
    [
        NVFusedBiasActivationDropout,
        NVFusedBiasDropoutRes,
        NVFusedBiasDropoutResLayerNorm,
    ],
)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("activation", [Activation.ReLU, Activation.GeLU])
@pytest.mark.parametrize("p", [0, 0.1, 0.5])
@pytest.mark.parametrize("layer_norm_style", [LayerNormStyle.Pre, LayerNormStyle.Post])
def test_nvfused_pattern_parity(
    fused_pattern: nn.Module,
    shape: tuple,
    amp: bool,
    bias: bool,
    activation: Activation,
    p: float,
    layer_norm_style: LayerNormStyle,
):

    if (
        fused_pattern != NVFusedBiasDropoutResLayerNorm
        and layer_norm_style == LayerNormStyle.Post
    ):
        pytest.skip(
            "Layer norm style doesn't apply, the same relevant params already tested once."
        )

    torch.cuda.manual_seed_all(0)
    torch.random.manual_seed(0)
    x = torch.normal(0, 1, size=shape, device="cuda", requires_grad=True)
    x_cpu = x.clone().cpu()

    with autocast(enabled=amp):
        fused = build_nvfused(
            fused_pattern, shape, bias, activation, p, layer_norm_style
        )
        fused.train().cuda()
        nvfused_res = fused(x, x) if fused.requires_residual else fused(x)
        fused.cpu()
        torch_res = (
            fused(x_cpu, x_cpu).cuda()
            if fused.requires_residual
            else fused(x_cpu).cuda()
        )

        # Check if operation was actually fused
        assert isinstance(
            nvfused_res.grad_fn, torch.autograd.function.BackwardCFunction
        )

        if p == 0.0:
            # Check fused and unfused paths are the same
            assert torch.allclose(torch_res, nvfused_res, atol=1e-6, rtol=1e-2)


@pytest.mark.skipif(not _is_functorch_available, reason="Functorch is not available")
@pytest.mark.parametrize("activation", [Activation.ReLU, Activation.GeLU])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("p", [0, 0.1, 0.5])
def test_nvfused_mlp(activation: Activation, device: torch.device, p: float):
    test_config = {
        "name": "MLP",
        "dim_model": LATENT,
        "dropout": p,
        "activation": activation,
        "hidden_layer_multiplier": 4,
        "bias": False,
    }

    torch.random.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    mlp = build_feedforward(test_config)
    mlp_default = build_feedforward(test_config | {"nvfuser": False})

    if mlp.requires_cuda and not device.type == "cuda":
        # pyre-fixme[29]: The library function `pytest.skip` is not supported by Pyre.
        pytest.skip("This MLP requires CUDA and current device does not match")

    inputs = torch.rand(BATCH, SEQ, LATENT, device=device)
    mlp.train()

    # Check fused pattern w/ unfused default (switch happens within NVFusedBiasActivationDropout)
    mlp.cuda()
    fused_res = mlp(inputs)

    mlp.cpu()
    unfused_res = mlp(inputs.cpu())

    if p == 0.0:
        assert torch.allclose(unfused_res.cuda(), fused_res, atol=1e-6, rtol=1e-2)

    # Check fused pattern w/ unfused default (switch happens within MLP)
    mlp.cuda()
    mlp_default.cuda()
    fused_res = mlp(inputs)
    unfused_res = mlp_default(inputs)
