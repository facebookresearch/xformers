# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import logging

import pytest
import torch
from torch.cuda.amp.autocast_mode import autocast

_triton_available = torch.cuda.is_available()

if _triton_available:
    try:
        from xformers.triton import dropout
        from xformers.triton.utils import gpu_capabilities_older_than_70

        _triton_available = True
    except ImportError:
        logging.warning(
            "Triton is not available, some optimizations will not be tested."
        )
        _triton_available = False

# Testing odd shapes on purpose
SHAPES = [
    (384, 128),
    (8, 384, 128),
    (8, 784, 512),
    (4, 2048, 384),
    (4, 3136, 1024),
    (2, 1024, 2048),
    (2, 2048, 4096),
    (2, 4096, 4096),
    (1, 2048, 12288),
]


@pytest.mark.skipif(not _triton_available, reason="Triton is not available")
@pytest.mark.skipif(
    not _triton_available or gpu_capabilities_older_than_70(),
    reason="Triton requires a SM70+ GPU",
)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("amp", [False, True])
def test_dropout(shape, amp):
    torch.random.manual_seed(0)

    x = torch.normal(0, 1, size=shape, device="cuda", requires_grad=True)

    with autocast(enabled=amp):
        tol = 1e-2 if amp else 1e-5  # AMP rounding causes issues, 1e-5 is the default

        # Check that 0 means no dropout
        y = dropout(x, p=0)
        assert torch.allclose(x.to(y.dtype), y, rtol=tol), f"{x[x>y]}"

        # Check that 1 means dropout for sure
        y = dropout(x, p=1)
        assert not torch.allclose(x.to(y.dtype), y, rtol=tol)

        # Check that the drops are different for every row (could catch broken seeds per row)
        y = dropout(x, p=0.5)

        y = y.flatten(0, 1) if y.ndim == 3 else y
        assert not torch.sum(torch.eq(y[0, :] == 0.0, y[1, :] == 0.0)) == y.shape[1]

        # Check that the drops are different over time, for the same line
        y_a = dropout(x, p=0.5)
        y_b = dropout(x, p=0.5)

        y_a = y_a.flatten(0, 1) if y_a.ndim == 3 else y_a
        y_b = y_b.flatten(0, 1) if y_b.ndim == 3 else y_b

        assert not torch.sum(torch.eq(y_a[0, :] == 0.0, y_b[0, :] == 0.0)) == y.shape[1]
