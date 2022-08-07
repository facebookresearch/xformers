# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging

import pytest
import torch
from torch.cuda.amp.autocast_mode import autocast

from xformers.components import Activation, build_activation

_triton_available = torch.cuda.is_available()
if _triton_available:
    try:
        from triton.testing import assert_almost_equal

        from xformers.triton import FusedLinear
        from xformers.triton.k_activations import get_triton_activation_kernel
        from xformers.triton.k_fused_matmul_fw import fused_matmul
        from xformers.triton.utils import gpu_capabilities_older_than_70
    except ImportError:
        logging.warning(
            "Triton is not available, some optimizations will not be tested."
        )
        _triton_available = False

SHAPES = [(128, 256), (8, 384, 128), (8, 784, 512)]


@pytest.mark.skipif(not _triton_available, reason="Triton is not available")
@pytest.mark.skipif(
    not _triton_available or gpu_capabilities_older_than_70(),
    reason="Triton requires a SM70+ GPU",
)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16])
def test_fused_matmul(shape, dtype):
    """Check that the matrix multiply kernel and Pytorch's give the same results"""
    torch.random.manual_seed(0)

    # Force pytorch to use TF32 accumulators (same as Triton)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True  # type: ignore

    # Raw fused matrix multiply first, to catch gross errors
    a = torch.rand((shape[-2], shape[-1]), dtype=dtype, device="cuda")
    b = torch.rand((shape[-1], shape[-2]), dtype=dtype, device="cuda")

    # Test that not passing any bias is fine
    res_torch = a @ b
    res_triton, _ = fused_matmul(a, b.transpose(0, 1).contiguous(), None)

    assert_almost_equal(
        res_torch, res_triton, err_msg="Vanilla matmul is broken", decimal=1
    )

    # Now test with a real FMA
    c = -torch.rand((shape[-2],), dtype=dtype, device="cuda")
    res_torch = torch.addmm(c, a, b)
    res_triton, _ = fused_matmul(a, b.transpose(1, 0).contiguous(), c)

    assert_almost_equal(
        res_torch,
        res_triton,
        err_msg=f"Vanilla fused matmul is broken {torch.max(torch.abs(res_torch-res_triton)).item()}",
        decimal=1,
    )

    # Now check that adding an activation to the mix still produces valid results
    for activation in Activation:
        torch_activation = build_activation(activation.value)
        a /= a.numel()
        res_torch = torch_activation(torch.addmm(c, a, b))

        triton_activation = get_triton_activation_kernel(activation)
        res_triton, _ = fused_matmul(
            a, b.transpose(1, 0).contiguous(), c, triton_activation
        )

        assert_almost_equal(
            res_torch,
            res_triton,
            err_msg=f"Fused matmul broken with activation {activation}. Max diff: \
                {torch.max(torch.abs(res_torch - res_triton))}",
            decimal=1,
        )


@pytest.mark.skipif(
    not _triton_available or gpu_capabilities_older_than_70(),
    reason="Triton requires a SM70+ GPU",
)
@pytest.mark.parametrize("activation", [None] + [a.value for a in Activation])  # type: ignore
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("amp", [True])  # FIXME: @lefaudeux check the fp32 case
def test_fused_linear_parity(shape, activation: Activation, bias: bool, amp: bool):
    """Check that PyTorch and fused linear layers give the same result"""

    # Instantiate pytorch and fused layers, same initialization
    torch.random.manual_seed(0)

    # Force pytorch to use TF32 accumulators (same as Triton)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True  # type: ignore

    X = torch.normal(0, 1, size=shape, device="cuda")
    X.requires_grad_()

    torch_linear = torch.nn.Linear(shape[-1], shape[-1] // 2, bias=bias).to("cuda")
    torch_activation = build_activation(activation)
    torch_sequence = torch.nn.Sequential(torch_linear, torch_activation)

    torch.random.manual_seed(0)
    X_ = torch.normal(0, 1, size=shape, device="cuda")
    X_.requires_grad_()

    # pyre-ignore[16]: TODO(T101400990): Pyre did not recognize the
    # `FusedLinear` import.
    triton_fused_linear = FusedLinear(
        shape[-1], shape[-1] // 2, bias=bias, activation=activation
    ).to("cuda")

    # Now check parity
    torch_linear.train()
    triton_fused_linear.train()

    torch_linear.zero_grad()
    triton_fused_linear.zero_grad()

    assert_almost_equal(triton_fused_linear.weight, torch_linear.weight)
    assert_almost_equal(X, X_)

    with autocast(enabled=amp):
        y_torch = torch_sequence(X)
        y_triton = triton_fused_linear(X_)

        # Check that BW also gives the same result
        loss_torch = torch.norm(y_torch)
        loss_torch.backward()

        loss_triton = torch.norm(y_triton)
        loss_triton.backward()

        assert_almost_equal(X, X_, err_msg=f"{X} vs. {X_}")

        # Input grad being correct checks both the loss + some of the backward pass
        assert X.grad is not None and X_.grad is not None
        assert_almost_equal(X.grad, X_.grad, err_msg=f"{X.grad} vs. {X_.grad}")

        # Check that the linear layer bias are also properly trainable
        if bias:
            assert (
                triton_fused_linear.bias is not None
                and triton_fused_linear.bias.grad is not None
            )
            assert torch_linear.bias is not None and torch_linear.bias.grad is not None
            assert_almost_equal(
                torch_linear.bias.grad,
                triton_fused_linear.bias.grad,
                err_msg=f"{torch_linear.bias.grad} vs. {triton_fused_linear.bias.grad}",
            )

        # Check that the linear layer weights are also properly trainable
        assert (
            torch_linear.weight.grad is not None
            and triton_fused_linear.weight.grad is not None
        )
        assert_almost_equal(
            torch_linear.weight.grad,
            triton_fused_linear.weight.grad,
            err_msg=f"{torch_linear.weight.grad} vs. {triton_fused_linear.weight.grad}",
        )
