# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from contextlib import nullcontext
from copy import deepcopy

import pytest
import torch

import xformers.ops
from torch import nn
from xformers.checkpoint import (
    _optimize_runtime_with_given_memory,
    checkpoint,
    get_optimal_checkpoint_policy,
    list_operators,
    selective_checkpoint_wrapper,
)

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
_devices = ["cpu"]
cuda_cap = (0, 0)

if torch.cuda.is_available():
    _devices.append("cuda")
    cuda_cap = torch.cuda.get_device_capability(_devices[1])


def _relu_policy(ctx, func, *args, **kwargs):
    return func == torch.ops.aten.relu.default


def _all_policy(ctx, func, *args, **kwargs):
    return True


@pytest.mark.parametrize("policy_fn", [None, [], _relu_policy, _all_policy])
@pytest.mark.parametrize("input_requires_grad", [True, False])
@pytest.mark.parametrize("device", _devices)
@pytest.mark.parametrize("autocast", [True, False])
def test_checkpoint(policy_fn, input_requires_grad, device, autocast):
    def build_module():
        return nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
        ).to(device)

    module = nn.ModuleList([build_module() for i in range(10)])

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
            out = checkpoint(module[i], out, policy_fn=policy_fn)
            out_copy = module_copy[i](out_copy)

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

    if op is xformers.ops.MemoryEfficientAttentionCkOp:
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


@pytest.mark.parametrize(
    "max_memory,optimal_soln",
    [
        (0, torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float64)),
        (100, torch.tensor([1, 0, 0, 0, 0, 1, 0, 1], dtype=torch.float64)),
        (120, torch.tensor([1, 0, 0, 1, 0, 0, 0, 0], dtype=torch.float64)),
        (200, torch.tensor([1, 0, 1, 0, 0, 1, 0, 1], dtype=torch.float64)),
        (220, torch.tensor([1, 0, 0, 1, 0, 1, 0, 1], dtype=torch.float64)),
        (320, torch.tensor([1, 0, 1, 1, 0, 1, 0, 1], dtype=torch.float64)),
        (420, torch.tensor([1, 1, 1, 1, 0, 1, 0, 1], dtype=torch.float64)),
    ],
)
def test_optimize_runtime_with_given_memory(max_memory, optimal_soln):
    data = [
        ("aten.copy_", 5, 0),
        ("aten.add", 5, 100),
        ("aten.div", 8, 100),
        ("aten.mm", 15, 120),
        ("aten.native_dropout", 15, 0),
        ("aten.linear", 9, 100),
        ("aten.t", 1, 0),
        ("aten.relu_", 5, 0),
    ]

    inplace_ops = [(0, 0), (7, 5)]
    view_like_ops = [6]
    rand_ops = [4]

    runtimes = torch.tensor([x[1] for x in data], dtype=torch.float64)
    memory = torch.tensor([x[2] for x in data], dtype=torch.float64)

    out = _optimize_runtime_with_given_memory(
        memory,
        runtimes,
        max_memory,
        view_like_ops,
        inplace_ops,
        rand_ops,
        force_store_random=False,
    )
    torch.testing.assert_close(optimal_soln, out)


def _get_model_blocks(num_layers, dtype, device, inplace, random, first_inplace):
    modules = []

    class Add_(torch.nn.Module):
        def forward(self, x):
            return x.add_(1)

    for _ in range(num_layers):
        mods = [
            nn.Linear(10, 10),
            nn.CELU(inplace=inplace),
        ]
        if first_inplace:
            mods.insert(0, Add_())
        if random:
            mods.append(nn.Dropout())
        mods.append(nn.Linear(10, 10))
        if random:
            mods.append(nn.Dropout())
        mods.append(nn.CELU(inplace=inplace))

        modules.append(nn.Sequential(*mods).to(device).to(dtype))
    return modules


class _Model(torch.nn.Module):
    def __init__(self, blocks, policy_fn):
        super().__init__()
        self.blocks = torch.nn.ModuleList(blocks)
        self.policy_fn = policy_fn

    def forward(self, x):
        for b in self.blocks:
            x = checkpoint(b, x, policy_fn=self.policy_fn)
        return x


@cuda_only
@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("memory_budget", [0, 0.03, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("random", [True, False])
@pytest.mark.parametrize("first_inplace", [False])
def test_optimal_checkpoint_policy(
    device, memory_budget, inplace, random, first_inplace
):
    if first_inplace and inplace:
        pytest.skip("This case is degenerate and doesn't work with vanilla PyTorch")
    torch.manual_seed(42)
    dtype = torch.float16
    modules = _get_model_blocks(
        3, dtype, device, inplace=inplace, random=random, first_inplace=first_inplace
    )
    inputs = torch.rand(32, 128, 10, dtype=dtype, device=device)

    policy_fn = get_optimal_checkpoint_policy(
        modules[0], inputs, memory_budget=memory_budget
    )
    model = _Model(modules, policy_fn)
    model_ref = torch.nn.Sequential(*deepcopy(modules))

    grad = torch.rand_like(inputs)

    torch.manual_seed(42)
    out = model(inputs.clone())
    out.backward(grad)

    torch.manual_seed(42)
    out_ref = model_ref(inputs.clone())
    out_ref.backward(grad)

    torch.testing.assert_close(out, out_ref)

    for p, p_ref in zip(model.parameters(), model_ref.parameters()):
        torch.testing.assert_close(p.grad, p_ref.grad)


@pytest.mark.skipif(True, reason="TODO[fmassa]: Broken on nightly")
@cuda_only
@pytest.mark.parametrize("no_grad", [False, True])
@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("memory_budget", [0, 0.1, 0.3, 1.0])
@pytest.mark.parametrize("inplace", [False])
@pytest.mark.parametrize("random", [False])
@torch._dynamo.config.patch(  # type: ignore
    "_experimental_support_context_fn_in_torch_utils_checkpoint", True
)
def test_selective_checkpoint_wrapper_compile(
    device, no_grad, memory_budget, inplace, random
):
    torch.manual_seed(42)
    dtype = torch.float16
    modules = _get_model_blocks(
        3, dtype, device, inplace=inplace, random=random, first_inplace=False
    )
    inputs = torch.rand(32, 128, 10, dtype=dtype, device=device)

    model = torch.nn.Sequential(
        *[selective_checkpoint_wrapper(b, memory_budget=memory_budget) for b in modules]
    )
    model = torch.compile(model)
    model_ref = torch.nn.Sequential(*deepcopy(modules))

    grad = torch.rand_like(inputs)

    context = torch.no_grad() if no_grad else nullcontext()

    with context:
        torch.manual_seed(42)
        out = model(inputs.clone())
        if not no_grad:
            out.backward(grad)

        torch.manual_seed(42)
        out_ref = model_ref(inputs.clone())
        if not no_grad:
            out_ref.backward(grad)

    atol = 3e-4
    rtol = 1e-3
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)

    if no_grad:
        return

    for p, p_ref in zip(model.parameters(), model_ref.parameters()):
        atol = 4e-4
        rtol = 2e-3
        torch.testing.assert_close(p.grad, p_ref.grad, atol=atol, rtol=rtol)
