# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import logging
from collections import OrderedDict
from contextlib import nullcontext

import pytest
import torch
import torch.nn as nn
from torch.cuda.amp.autocast_mode import autocast

import xformers
from xformers.components import Activation, ResidualNormStyle

# Store original and possible flag setting
flag_orig = xformers._is_functorch_available
flag_new = True
xformers._is_functorch_available = True


_gpu_available = torch.cuda.is_available()

try:
    import xformers.components.feedforward as ff
    from xformers.components.nvfuser import (
        NVFusedBiasActivationDropout,
        NVFusedBiasDropoutRes,
        NVFusedBiasDropoutResLayerNorm,
    )
    from xformers.components.nvfuser.utils import build_nvfused
except ImportError as e:
    logging.warning(f"Functorch is not available to run test_nvfuser.py. \nError {e}")
    flag_new = False

xformers._is_functorch_available = flag_orig

FUSED_PATTERNS = (
    [
        NVFusedBiasActivationDropout,
        NVFusedBiasDropoutRes,
        NVFusedBiasDropoutResLayerNorm,
    ]
    if flag_new
    else []
)

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

ACTIVATIONS = [
    Activation.ReLU,
    Activation.GeLU,
    Activation.LeakyReLU,
    Activation.SquaredReLU,
    Activation.SmeLU,
]


@pytest.mark.skipif(not flag_new, reason="Functorch is not available")
@pytest.mark.skipif(not _gpu_available, reason="GPU is not available")
@pytest.mark.parametrize("fused_pattern", FUSED_PATTERNS)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("p", [0, 0.1, 0.5])
@pytest.mark.parametrize(
    "layer_norm_style", [None, ResidualNormStyle.Pre, ResidualNormStyle.Post]
)
def test_nvfused_pattern_parity(
    fused_pattern: nn.Module,
    shape: tuple,
    amp: bool,
    bias: bool,
    activation: Activation,
    p: float,
    layer_norm_style: ResidualNormStyle,
):
    # Enable global flag
    xformers._is_functorch_available = flag_new

    if (
        fused_pattern != NVFusedBiasDropoutResLayerNorm
        and layer_norm_style != ResidualNormStyle.Pre
    ):
        pytest.skip(
            "Layer norm style doesn't apply, the same relevant params already tested once."
        )

    torch.cuda.manual_seed_all(0)
    torch.random.manual_seed(0)
    x = torch.normal(0, 1, size=shape, device="cuda", requires_grad=True)
    x_cpu = x.clone().cpu()

    with autocast(enabled=amp), pytest.raises(
        ValueError
    ) if layer_norm_style is None else nullcontext():
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

    # Restore original flag configuration
    xformers._is_functorch_available = flag_orig


@pytest.mark.skipif(not flag_new, reason="Functorch is not available")
@pytest.mark.skipif(not _gpu_available, reason="GPU is not available")
@pytest.mark.parametrize("activation", ACTIVATIONS)
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
    # Enable global flag
    xformers._is_functorch_available = flag_new

    torch.random.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    mlp = ff.build_feedforward(test_config)
    # Creates non-fused default MLP
    xformers._is_functorch_available = False
    mlp_default = ff.build_feedforward(test_config)
    xformers._is_functorch_available = flag_new

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

    # Load same weight parameters into both models
    default_param_dict = OrderedDict(
        [
            ("mlp.2.weight", v) if k == "mlp.3.weight" else (k, v)
            for k, v in mlp_default.state_dict().items()
        ]
    )
    mlp.load_state_dict(default_param_dict)
    fused_res = mlp(inputs)
    unfused_res = mlp_default(inputs)

    if p == 0.0:
        assert torch.allclose(unfused_res, fused_res, atol=1e-6, rtol=1e-2)

    # Restore original flag configuration
    xformers._is_functorch_available = flag_orig
