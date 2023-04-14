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

_gpu_available = torch.cuda.is_available()
_triton_available = xformers._is_triton_available()

if _triton_available:
    try:
        import triton  # noqa: F401

        from xformers.triton import dropout as triton_dropout
        from xformers.triton.dropout import FusedDropoutBias
        from xformers.triton.utils import gpu_capabilities_older_than_70

        _triton_available = True
    except ImportError:
        logging.warning(
            "Triton is not available, some optimizations will not be tested."
        )
        _triton_available = False

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


@pytest.mark.skipif(not _triton_available, reason="Triton is not available")
def test_dropout_cpu():
    triton_dropout = FusedDropoutBias(p=0.1, bias_shape=None)
    x = torch.normal(0, 1, size=(16, 16), device="cpu")
    _ = triton_dropout(x)

    # Check eval means no dropout
    triton_dropout.eval()
    y = triton_dropout(x)
    assert y.count_nonzero() == y.numel()

    triton_dropout.train()
    y = triton_dropout(x)
    assert y.count_nonzero() != y.numel()


@pytest.mark.skipif(not _gpu_available, reason="GPU is not available")
@pytest.mark.skipif(not _triton_available, reason="Triton is not available")
@pytest.mark.skipif(
    not _triton_available or gpu_capabilities_older_than_70(),
    reason="Triton requires a SM70+ GPU",
)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("p", [0, 0.1, 0.5])
def test_dropout(shape, amp, bias, p):
    """
    Check some basic dropout properties
    """
    torch.random.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    x = torch.normal(0, 1, size=shape, device="cuda", requires_grad=True)
    b = (
        torch.normal(0, 1, size=(shape[-1],), device="cuda", requires_grad=True)
        if bias
        else None
    )

    with autocast(enabled=amp):
        tol = 1e-2 if amp else 1e-5  # AMP rounding causes issues, 1e-5 is the default

        # Check that 0 means no dropout
        y = triton_dropout(x, p=0, bias=b)
        x_ref = (x + b if bias else x).to(y.dtype)
        assert torch.allclose(x_ref, y, rtol=tol), f"{x[x>y]}"

        # Check that 1 means drop all
        y = triton_dropout(x, p=1, bias=b)
        x_ref = (x + b if bias else x).to(y.dtype)
        assert torch.allclose(torch.zeros_like(y), y, rtol=tol)

        # Check that .99 means probably dropout
        y = triton_dropout(x, p=0.99, bias=b)
        x_ref = (x + b if bias else x).to(y.dtype)
        assert not torch.allclose(x_ref, y, rtol=tol)

        # Check that the drops are different for every row (could catch broken seeds per row)
        y = triton_dropout(x, p=0.5)

        y = y.flatten(0, 1) if y.ndim == 3 else y
        assert not torch.sum(torch.eq(y[0, :] == 0.0, y[1, :] == 0.0)) == y.shape[1]

        # Check that the drops are different over time, for the same line
        y_a = triton_dropout(x, p=0.5)
        y_b = triton_dropout(x, p=0.5)

        y_a = y_a.flatten(0, 1) if y_a.ndim == 3 else y_a
        y_b = y_b.flatten(0, 1) if y_b.ndim == 3 else y_b

        assert (
            not torch.sum(torch.eq(y_a[0, :] == 0.0, y_b[0, :] == 0.0)).item()
            == y.shape[1]
        )

        # Check that the drop probability is about right
        y = triton_dropout(x, p=p)
        drop_p = (y.numel() - y.count_nonzero()) / y.numel()
        assert abs(drop_p - p) < 0.02

        # Check that the same seeds lead to the same dropout
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        y_1 = triton_dropout(x, p=0.5)

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        y_2 = triton_dropout(x, p=0.5)

        torch.testing.assert_close(y_1, y_2)


@pytest.mark.skipif(not _gpu_available, reason="GPU is not available")
@pytest.mark.skipif(not _triton_available, reason="Triton is not available")
@pytest.mark.skipif(
    not _triton_available or gpu_capabilities_older_than_70(),
    reason="Triton requires a SM70+ GPU",
)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("activation", [a.value for a in Activation])
@pytest.mark.parametrize("p", [0, 0.01, 0.5])
def test_dropout_parity(shape, amp, bias, activation, p):
    """
    Check some basic dropout properties
    """

    torch.random.manual_seed(0)
    x = torch.normal(0, 1, size=shape, device="cuda", requires_grad=True)
    b = (
        torch.ones(size=(shape[-1],), device="cuda", requires_grad=True)
        if bias
        else None
    )

    torch.random.manual_seed(0)
    x_ = torch.normal(0, 1, size=shape, device="cuda", requires_grad=True)
    b_ = (
        torch.ones(size=(shape[-1],), device="cuda", requires_grad=True)
        if bias
        else None
    )

    with autocast(enabled=amp):
        torch_activation = build_activation(activation)
        res_torch = torch.nn.functional.dropout(
            torch_activation(x + b if b is not None else x), p=p
        )
        loss_torch = torch.sum(res_torch)

        res_triton = triton_dropout(x=x_, p=p, bias=b_, activation=activation)
        loss_triton = torch.sum(res_triton)

        if p < 0.01:
            # Check the FW pass
            assert torch.allclose(
                loss_torch, loss_triton, rtol=0.01
            ), f"{loss_torch} - {loss_triton}"

            # Check the gradients
            loss_torch.backward()
            loss_triton.backward()

            # - gradients wrt inputs
            assert torch.allclose(
                torch.norm(x.grad), torch.norm(x_.grad), rtol=0.01
            ), f"{x.grad}\n{x_.grad}"

            # - gradients wrt bias
            if bias:
                assert torch.allclose(
                    torch.norm(b.grad), torch.norm(b_.grad), rtol=0.01
                ), f"{b.grad.norm()} - {b_.grad.norm()}"
