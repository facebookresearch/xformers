import logging

import pytest
import torch
from torch.cuda.amp.autocast_mode import autocast

from xformers.components import Activation, build_activation

_triton_available = torch.cuda.is_available()
if _triton_available:
    try:
        from xformers.triton import FusedLinear
        from xformers.triton.activations import get_triton_activation_kernel
        from xformers.triton.fused_matmul import fused_matmul
        from xformers.triton.utils import gpu_capabilities_older_than_70

    except ImportError:
        logging.warning(
            "Triton is not available, some optimizations will not be tested."
        )
        _triton_available = False

SHAPES = [(8, 384, 128), (8, 784, 512)]


@pytest.mark.skipif(not _triton_available, reason="Triton is not available")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize(
    "dtype", [torch.float32]
)  # Triton use tensor cores, which return slightly different results to pytorch mm
def test_fused_matmul(shape, dtype):
    """ Check that the matrix multiply kernel and Pytorch's give the same results"""
    torch.random.manual_seed(0)

    # Raw fused matrix multiply first, to catch gross errors
    a = torch.rand((shape[1], shape[2]), dtype=dtype, device="cuda")
    b = torch.rand((shape[2], shape[1]), dtype=dtype, device="cuda")

    # Test that not passing any bias is fine
    res_torch = a @ b
    res_triton, _ = fused_matmul(a, b.transpose(0, 1), None)
    assert torch.allclose(res_torch, res_triton), "Vanilla matmul is broken"

    # Now test with a real FMA
    c = -torch.rand((shape[1],), dtype=dtype, device="cuda")
    res_torch = torch.addmm(c, a, b)
    res_triton, _ = fused_matmul(a, b.transpose(1, 0), c)

    assert torch.allclose(
        res_torch, res_triton
    ), f"Vanilla fused matmul is broken {torch.max(torch.abs(res_torch-res_triton)).item()}"

    # Now check that adding an activation to the mix still produces valid results
    for activation in Activation:
        torch_activation = build_activation(activation.value)
        res_torch = torch_activation(torch.addmm(c, a, b))

        triton_activation = get_triton_activation_kernel(activation)
        res_triton, _ = fused_matmul(a, b.transpose(1, 0), c, triton_activation)

        # FIXME: @lefaudeux
        # GeLUs are not well handled for now, we use an approximation
        # they're also slower than pytorch so not likely to be used
        # Issue tracked with https://github.com/fairinternal/xformers/issues/238
        tol = 1e-6 if activation != Activation.GeLU else 1e-2

        assert torch.allclose(
            res_torch, res_triton, atol=tol
        ), f"Fused matmul broken with activation {activation}. Max diff: {torch.max(torch.abs(res_torch - res_triton))}"


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
    X = torch.normal(0, 1, size=shape, device="cuda")
    X.requires_grad_()

    torch_linear = torch.nn.Linear(shape[2], shape[2] // 2, bias=bias).to("cuda")
    torch_activation = build_activation(activation)
    torch_sequence = torch.nn.Sequential(torch_linear, torch_activation)

    torch.random.manual_seed(0)
    X_ = torch.normal(0, 1, size=shape, device="cuda")
    X_.requires_grad_()

    triton_fused_linear = FusedLinear(
        shape[2], shape[2] // 2, bias=bias, activation=activation
    ).to("cuda")

    # Now check parity
    torch_linear.train()
    triton_fused_linear.train()

    torch_linear.zero_grad()
    triton_fused_linear.zero_grad()

    assert torch.allclose(
        triton_fused_linear.weight, torch_linear.weight
    ), "Broken test setup"
    assert torch.allclose(X, X_), "Broken test setup"

    with autocast(enabled=amp):
        tolerance = 1e-3 if not amp else 1e-2

        y_torch = torch_sequence(X)
        y_triton = triton_fused_linear(X_)

        # Check that BW also gives the same result
        loss_torch = torch.norm(y_torch)
        loss_torch.backward()

        loss_triton = torch.norm(y_triton)
        loss_triton.backward()

        assert torch.allclose(X, X_, atol=tolerance), f"{X[:,0,0]} vs. {X_[:,0,0]}"

        # Input grad being correct checks both the loss + some of the backward pass
        assert torch.allclose(
            X.grad, X_.grad, atol=tolerance
        ), f"{X.grad[:,0,0]} vs. {X_.grad[:,0,0]}"

        # Check that the linear layer bias are also properly trainable
        if bias:
            assert triton_fused_linear.bias is not None
            assert triton_fused_linear.bias.grad is not None
            assert torch.allclose(
                torch_linear.bias.grad, triton_fused_linear.bias.grad, atol=tolerance
            )

        # Check that the linear layer weights are also properly trainable
        assert torch.allclose(
            torch_linear.weight.grad,
            triton_fused_linear.weight.grad,
            atol=tolerance,
        )
