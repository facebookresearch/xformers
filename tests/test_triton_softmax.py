# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging

import pytest
import torch
from torch.cuda.amp.autocast_mode import autocast

import xformers

try:
    from xformers.triton import log_softmax as triton_log_softmax
    from xformers.triton import softmax as triton_softmax

    _triton_available = xformers._is_triton_available()
except ImportError as e:
    logging.warning(
        f"Triton is not available, some optimizations will not be tested.\n{e}"
    )
    _triton_available = False

SHAPES = [
    (384, 384),
    (2, 384, 384),
    (1, 784, 784),
    (1, 1024, 1024),
    (1, 2048, 2048),
    (1, 3136, 3136),
    (1, 4096, 4096),
    (2, 2, 384, 384),
    (2, 2, 2, 384, 384),
]


@pytest.mark.skipif(not _triton_available, reason="Triton is not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("log", [False, True])
@pytest.mark.parametrize("masking", [True, False])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("contiguous", [True, False])
def test_softmax_parity(shape, amp, log, masking, causal, contiguous):
    """Check that PyTorch and Triton softmax give the same result"""
    torch.random.manual_seed(0)

    # Check the result of a FW pass
    X = torch.normal(0, 1, size=shape, device="cuda", requires_grad=False)

    if not contiguous:
        # Make sure that the buffer is not contiguous
        X = X.transpose(-2, -1).contiguous().transpose(-2, -1)

    X_ = X.clone()
    X.requires_grad = True
    X_.requires_grad = True

    seq = shape[-1]
    mask = torch.zeros((seq, seq)).cuda()
    if masking:
        mask[torch.rand((seq, seq)) > 0.8] = -float("inf")

    mask_triton = mask.clone() if masking else None

    if causal:
        mask[~torch.tril(torch.ones_like(mask)).bool()] = -float("inf")

    with autocast(enabled=amp):
        y_torch = (
            torch.log_softmax(X + mask, dim=-1)
            if log
            else torch.softmax(X + mask, dim=-1)
        )
        y_triton = (
            triton_log_softmax(X_, mask_triton, causal)
            if log
            else triton_softmax(X_, mask_triton, causal)
        )

        assert torch.allclose(y_torch, y_triton, equal_nan=True)

        # Check that BW also gives the same result
        loss_torch = torch.norm(y_torch.transpose(-2, -1) @ y_torch)
        loss_torch.backward()

        loss_triton = torch.norm(y_triton.transpose(-2, -1) @ y_triton)
        loss_triton.backward()

        assert torch.allclose(
            torch.norm(X.grad), torch.norm(X_.grad), equal_nan=True, atol=1e-5
        ), f"{torch.norm(X.grad)}, {torch.norm(X_.grad)}"


@pytest.mark.skipif(not _triton_available, reason="Triton is not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_softmax(dtype):
    b, s, d = 8, 64, 32

    a = torch.rand(b, s, d, device="cuda", dtype=dtype)
    triton_softmax(a)


@pytest.mark.skipif(not _triton_available, reason="Triton is not available")
@pytest.mark.parametrize("log", [False, True])
@pytest.mark.parametrize("masking", [True, False])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_softmax_parity_fallback(log, masking, causal, contiguous, device):
    """Check that the fallback paths are correct"""
    torch.random.manual_seed(0)

    shape = (16, 16)

    # Check the result of a FW pass
    X = torch.normal(0, 1, size=shape, device=device, requires_grad=False)

    if not contiguous:
        # Make sure that the buffer is not contiguous
        X = X.transpose(-2, -1).contiguous().transpose(-2, -1)

    X_ = X.clone()
    X.requires_grad = True
    X_.requires_grad = True

    seq = shape[1]
    mask = torch.zeros((seq, seq), device=device)
    if masking:
        mask[torch.rand((seq, seq), device=device) > 0.8] = -float("inf")

    mask_causal = torch.zeros_like(mask)
    if causal:
        mask_causal[~torch.tril(torch.ones_like(mask)).bool()] = -float("inf")

    y_torch = (
        torch.log_softmax(X + mask + mask_causal, dim=-1)
        if log
        else torch.softmax(X + mask + mask_causal, dim=-1)
    )
    y_triton = (
        triton_log_softmax(X_, mask, causal)
        if log
        else triton_softmax(X_, mask, causal)
    )

    assert torch.allclose(y_torch, y_triton, equal_nan=True)

    # Check that BW also gives the same result
    loss_torch = torch.norm(y_torch.transpose(-2, -1) @ y_torch)
    loss_torch.backward()

    loss_triton = torch.norm(y_triton.transpose(-2, -1) @ y_triton)
    loss_triton.backward()

    assert torch.allclose(
        torch.norm(X.grad), torch.norm(X_.grad), equal_nan=True, atol=1e-5
    ), f"{torch.norm(X.grad)}, {torch.norm(X_.grad)}"
