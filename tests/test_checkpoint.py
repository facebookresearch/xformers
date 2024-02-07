# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from copy import deepcopy

import pytest
import torch
from torch import nn

import xformers.ops
from xformers import checkpoint, list_operators

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
disable_on_rocm = pytest.mark.skipif(
    not not torch.version.hip, reason="could not be done on ROCM"
)
_devices = ["cpu"]
cuda_cap = (0, 0)

if torch.cuda.is_available():
    _devices.append("cuda")
    cuda_cap = torch.cuda.get_device_capability(_devices[1])


def _relu_policy(func, *args, **kwargs):
    return func == torch.ops.aten.relu.default


def _all_policy(func, *args, **kwargs):
    return True


@disable_on_rocm
@pytest.mark.parametrize("policy_fn", [None, [], _relu_policy, _all_policy])
@pytest.mark.parametrize("input_requires_grad", [True, False])
@pytest.mark.parametrize("device", _devices)
@pytest.mark.parametrize("autocast", [True, False])
def test_checkpoint(policy_fn, input_requires_grad, device, autocast):
    module = nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
    ).to(device)

    # Run model with and without checkpointing and verify gradients are
    # equivalent, regardless of if inputs require grads or not.
    module_copy = deepcopy(module)

    inputs = torch.rand(32, 10, device=device)
    inputs_copy = inputs.clone()
    inputs.requires_grad_(input_requires_grad)
    inputs_copy.requires_grad_(input_requires_grad)
    out = inputs
    out_copy = inputs_copy
    with torch.autocast(device_type=device, enabled=autocast):
        for i in range(10):
            out = checkpoint(module, out, policy_fn=policy_fn)
            out_copy = module_copy(out_copy)

    assert torch.allclose(out, out_copy)
    out.sum().backward()
    out_copy.sum().backward()
    for p, p_copy in zip(module.parameters(), module_copy.parameters()):
        assert torch.allclose(p.grad, p_copy.grad)


@pytest.mark.parametrize("policy_fn", [None, [], _relu_policy, _all_policy])
@pytest.mark.parametrize("input_requires_grad", [True, False])
@pytest.mark.parametrize("grad_mode", [True, False])
def test_checkpoint_with_grad(policy_fn, input_requires_grad, grad_mode):
    module = nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
    )

    # Run model with and without checkpointing and verify gradients are
    # equivalent, regardless of if inputs require grads or not.
    module_copy = deepcopy(module)

    inputs = torch.rand(32, 10)
    inputs_copy = inputs.clone()
    inputs.requires_grad_(input_requires_grad)
    inputs_copy.requires_grad_(input_requires_grad)
    out = inputs
    out_copy = inputs_copy
    with torch.set_grad_enabled(grad_mode):
        for i in range(10):
            out = checkpoint(module, out, policy_fn=policy_fn)
            out_copy = module_copy(out_copy)

    assert torch.allclose(out, out_copy)


@cuda_only
@pytest.mark.parametrize("policy_fn", [None, [], _relu_policy, _all_policy])
@pytest.mark.parametrize("input_requires_grad", [True, False])
@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("autocast", [True, False])
@pytest.mark.parametrize(
    "op",
    [
        xformers.ops.MemoryEfficientAttentionFlashAttentionOp,
        (
            xformers.ops.MemoryEfficientAttentionCutlassOp
            if torch.version.cuda
            else xformers.ops.MemoryEfficientAttentionCkOp
        ),
    ],
)
def test_checkpoint_attention(policy_fn, input_requires_grad, device, autocast, op):
    if (
        op[0].CUDA_MINIMUM_COMPUTE_CAPABILITY > cuda_cap
        or op[1].CUDA_MINIMUM_COMPUTE_CAPABILITY > cuda_cap
    ):
        pytest.skip("skipping operator not supported in this arch")

    if (
        op is xformers.ops.MemoryEfficientAttentionFlashAttentionOp
        and torch.version.hip
    ):
        pytest.skip("FlashAttentionOp is not supported on ROCM!")

    if op is xformers.ops.MemoryEfficientAttentionCkOp and op[0].IS_CK_TILED:
        pytest.skip("Gradience is currently not supported by ck-tiled!")

    class Attn(nn.Module):
        def forward(self, x):
            out = xformers.ops.memory_efficient_attention(x, x, x, op=op)
            return out + x

    num_layers = 10
    dtype = torch.float32 if autocast else torch.float16
    modules = nn.Sequential(
        *[
            nn.Sequential(
                nn.Linear(10, 64),
                Attn(),
                nn.ReLU(),
                nn.Linear(64, 10),
                nn.ReLU(),
            )
            .to(device)
            .to(dtype)
            for _ in range(num_layers)
        ]
    )

    # Run model with and without checkpointing and verify gradients are
    # equivalent, regardless of if inputs require grads or not.
    modules_copy = deepcopy(modules)

    inputs = torch.rand(32, 128, 10, dtype=dtype, device=device)
    inputs_copy = inputs.clone()
    inputs.requires_grad_(input_requires_grad)
    inputs_copy.requires_grad_(input_requires_grad)
    out = inputs
    out_copy = inputs_copy
    with torch.autocast(device_type=device, enabled=autocast):
        for i in range(num_layers):
            out = checkpoint(modules[i], out, policy_fn=policy_fn)
            out_copy = modules_copy[i](out_copy)

    assert torch.allclose(out, out_copy)
    out.sum().backward()
    out_copy.sum().backward()
    for p, p_copy in zip(modules.parameters(), modules_copy.parameters()):
        assert torch.allclose(
            p.grad, p_copy.grad
        ), f"{(p.grad - p_copy.grad).abs().max()}"

    if input_requires_grad:
        assert torch.allclose(inputs.grad, inputs_copy.grad)


def test_list_operators():
    module = nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
    )
    inputs = torch.rand(32, 10)
    operators = list_operators(module, inputs)
    operators_str = [str(x) for x in operators]
    ref = [
        "aten.t.default",
        "aten.addmm.default",
        "aten.relu.default",
        "aten.detach.default",
        "aten.t.default",
        "aten.addmm.default",
        "aten.relu.default",
        "aten.detach.default",
    ]
    assert operators_str == ref
