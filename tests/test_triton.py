import logging

import pytest
import torch
from torch.cuda.amp.autocast_mode import autocast

try:
    from xformers.triton.softmax import softmax as triton_softmax

    _triton_available = True
except ImportError:
    logging.warning("Triton is not available, some optimizations will not be tested.")
    _triton_available = False

# Testing odd shapes on purpose
SHAPES = [
    (8, 384, 128),
    (8, 784, 512),
    (4, 2048, 384),
    (4, 3136, 1024),
    (2, 1024, 2048),
    (2, 2048, 4096),
    (2, 4096, 4096),
]


@pytest.mark.skipif(not _triton_available, reason="Triton is not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("amp", [False, True])
def test_softmax_parity(shape, amp):
    """Check that PyTorch and Triton softmax give the same result"""

    # Check the result of a FW pass
    X = torch.normal(0, 1, size=shape, device="cuda", requires_grad=True)
    X_ = X.detach().clone()
    X_.requires_grad = True

    with autocast(enabled=amp):
        y_torch = torch.softmax(X, dim=-1)
        y_triton = triton_softmax(X_)
        assert torch.allclose(y_torch, y_triton)

        # Check that BW also gives the same result
        loss_torch = torch.norm(y_torch)
        loss_torch.backward()

        loss_triton = torch.norm(y_triton)
        loss_triton.backward()
        assert torch.allclose(X.grad, X_.grad)


@pytest.mark.skipif(not _triton_available, reason="Triton is not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_fp16(dtype):
    b, s, d = 8, 64, 32

    a = torch.rand(b, s, d, device="cuda", dtype=dtype)
    triton_softmax(a)
