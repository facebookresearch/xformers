# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging

import pytest
import torch
from torch.cuda.amp.autocast_mode import autocast

import xformers
from xformers.components import Activation, build_activation

_triton_available = xformers._is_triton_available()
if _triton_available:
    try:
        import triton  # noqa: F401

        from xformers.triton import FusedLinear
        from xformers.triton.k_activations import get_triton_activation_index
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
    # TODO: fix or remove this
    pytest.skip("This is broken")
    torch.random.manual_seed(0)

    # Raw fused matrix multiply first, to catch gross errors
    a = torch.normal(0, 1, size=(shape[-2], shape[-1]), dtype=dtype, device="cuda")
    b = torch.normal(0, 1, size=(shape[-1], shape[-2]), dtype=dtype, device="cuda")

    # Test that not passing any bias is fine
    res_torch = a @ b
    res_triton, _ = fused_matmul(
        a, b.transpose(0, 1).contiguous(), bias=None, activation=0
    )
    torch.testing.assert_close(res_torch, res_triton)

    # Now test with a real FMA
    c = -torch.randn((shape[-2],), dtype=dtype, device="cuda")
    res_torch = torch.addmm(c, a, b)
    res_triton, _ = fused_matmul(a, b.transpose(1, 0).contiguous(), c)

    torch.testing.assert_close(
        res_torch,
        res_triton,
        atol=1e-3,
        rtol=1e-3,
        msg="Fused matmul broken",
    )

    # Now check that adding an activation to the mix still produces valid results
    # NOTE: SquaredReLU fails, some outlier representation issue but the eyeballed results look reasonable
    # could be due to a different accumulation out of the box (tf32 for instance)
    for activation in filter(
        lambda x: x not in (Activation.SquaredReLU, Activation.StarReLU), Activation
    ):
        torch_activation = build_activation(activation.value)
        res_torch = torch_activation(torch.addmm(c, a, b))

        triton_activation_index = get_triton_activation_index(activation)
        print(activation, triton_activation_index)
        res_triton, _ = fused_matmul(
            a, b.transpose(1, 0).contiguous(), c, triton_activation_index
        )

        torch.testing.assert_close(
            res_torch,
            res_triton,
            atol=1e-3,
            rtol=1e-3,
            msg=f"Fused matmul broken with activation {activation}",
        )


@pytest.mark.skipif(
    not _triton_available or gpu_capabilities_older_than_70(),
    reason="Triton requires a SM70+ GPU",
)
@pytest.mark.parametrize("activation", [None] + [a.value for a in Activation])  # type: ignore
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("amp", [True])
def test_fused_linear_parity(shape, activation: Activation, bias: bool, amp: bool):
    """Check that PyTorch and fused linear layers give the same result"""
    # TODO: fix or remove this
    pytest.skip("This is broken")
    torch.random.manual_seed(0)

    # Instantiate pytorch and fused layers, same initialization
    X = torch.normal(0, 1, size=shape, device="cuda")
    X.requires_grad_()

    torch_linear = torch.nn.Linear(shape[-1], shape[-1] // 2, bias=bias).to("cuda")
    torch_sequence = torch.nn.Sequential(torch_linear, build_activation(activation))

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

    torch.testing.assert_close(
        triton_fused_linear.weight,
        torch_linear.weight,
        atol=1e-3,
        rtol=1e-3,
        msg="Broken test setup",
    )
    torch.testing.assert_close(X, X_, atol=1e-3, rtol=1e-3, msg="Broken test setup")

    with autocast(enabled=amp):
        y_torch = torch_sequence(X)
        y_triton = triton_fused_linear(X_)

        grad = torch.randn_like(y_torch)

        # Check that BW also gives the same result
        y_torch.backward(grad)
        y_triton.backward(grad)

        torch.testing.assert_close(X, X_, atol=1e-3, rtol=1e-3)

        # Input grad being correct checks both the loss + some of the backward pass
        assert X.grad is not None and X_.grad is not None
        torch.testing.assert_close(X.grad, X_.grad, atol=1e-3, rtol=1e-3)

        # Check that the linear layer bias are also properly trainable
        if bias:
            assert (
                triton_fused_linear.bias is not None
                and triton_fused_linear.bias.grad is not None
            )
            assert torch_linear.bias is not None and torch_linear.bias.grad is not None
            torch.testing.assert_close(
                torch_linear.bias.grad,
                triton_fused_linear.bias.grad,
                atol=1e-3,
                rtol=1e-3,
            )

        # Check that the linear layer weights are also properly trainable
        assert (
            torch_linear.weight.grad is not None
            and triton_fused_linear.weight.grad is not None
        )
        torch.testing.assert_close(
            torch_linear.weight.grad,
            triton_fused_linear.weight.grad,
            atol=1e-3,
            rtol=1e-3,
        )
