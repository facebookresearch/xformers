# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from importlib import import_module

import pytest
import torch

checkpoint_module = import_module("xformers.checkpoint")


def _save_mm(ctx, func, *args, **kwargs):
    return func == torch.ops.aten.mm.default


def _wrapper(module, **kwargs):
    with pytest.warns(FutureWarning, match="selective_checkpoint_wrapper"):
        return checkpoint_module.selective_checkpoint_wrapper(module, **kwargs)


@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode])
def test_no_grad_warmup_does_not_cache_policy(context, monkeypatch):
    module = torch.nn.Linear(4, 4)
    wrapper = _wrapper(module, memory_budget=0.5)
    calls = []

    def select_policy(function, *args, memory_budget):
        assert function is module
        calls.append(memory_budget)
        return _save_mm

    monkeypatch.setattr(
        checkpoint_module, "_get_optimal_checkpoint_policy", select_policy
    )
    inputs = torch.randn(2, 4, requires_grad=True)
    with context():
        torch.testing.assert_close(wrapper(inputs), module(inputs))
        torch.testing.assert_close(wrapper(inputs), module(inputs))
    assert calls == []
    assert wrapper.policy_fn is None
    wrapper(inputs).sum().backward()
    assert wrapper.policy_fn is _save_mm
    wrapper(inputs).sum().backward()
    assert calls == [0.5]


@pytest.mark.parametrize("policy", [[], [torch.ops.aten.mm.default], _save_mm])
def test_explicit_policy_survives_no_grad_warmup(policy, monkeypatch):
    module = torch.nn.Linear(4, 4)
    wrapper = _wrapper(module, policy_fn=policy)

    def unexpected_solver(*args, **kwargs):
        pytest.fail("explicit policy must not invoke the budget optimizer")

    monkeypatch.setattr(
        checkpoint_module, "_get_optimal_checkpoint_policy", unexpected_solver
    )
    inputs = torch.randn(2, 4, requires_grad=True)
    with torch.no_grad():
        torch.testing.assert_close(wrapper(inputs), module(inputs))
    wrapper(inputs).sum().backward()
    assert wrapper.policy_fn is policy


@pytest.mark.skipif(
    not torch.cuda.is_available() or not checkpoint_module._scipy_is_available,
    reason="CUDA and SciPy are required for the actual budget optimizer",
)
@pytest.mark.parametrize("memory_budget", [0.0, 0.5, 1.0])
def test_no_grad_then_training_uses_real_budget_optimizer(memory_budget):
    torch.manual_seed(42)
    module = torch.nn.Sequential(
        torch.nn.Linear(16, 16), torch.nn.Sigmoid(), torch.nn.Linear(16, 16)
    ).cuda()
    reference = deepcopy(module)
    wrapper = _wrapper(module, memory_budget=memory_budget)
    inputs = torch.randn(8, 16, device="cuda", requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()
    with torch.no_grad():
        torch.testing.assert_close(wrapper(inputs), reference(reference_inputs))
    assert wrapper.policy_fn is None
    output = wrapper(inputs)
    reference_output = reference(reference_inputs)
    assert callable(wrapper.policy_fn)
    policy = wrapper.policy_fn
    output.sum().backward()
    reference_output.sum().backward()
    torch.testing.assert_close(output, reference_output)
    torch.testing.assert_close(inputs.grad, reference_inputs.grad)
    for parameter, reference_parameter in zip(
        module.parameters(), reference.parameters()
    ):
        torch.testing.assert_close(parameter.grad, reference_parameter.grad)
    wrapper(inputs).sum().backward()
    assert wrapper.policy_fn is policy
