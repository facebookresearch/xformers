# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math

import pytest
import torch

try:
    from xformers.triton import mem_efficient_attention
    from xformers.triton.utils import gpu_capabilities_older_than_70

    _triton_available = True
except ImportError:
    logging.warning("Triton is not available, some optimizations will not be tested.")
    _triton_available = False


# Testing odd shapes on purpose
SHAPES = [
    # (384, 256),
    # (1, 384, 128),
    # (8, 384, 128),
    # (8, 784, 512),
    # (2, 2048, 384),
    # (4, 3136, 1024),
    (2, 1024, 2048),
]


def attention_pytorch(q, k, v):
    # attention matrix
    q = q / math.sqrt(q.size(-1))
    a = q @ k.transpose(-2, -1)

    # softmax
    a = torch.softmax(a, dim=-1)

    # retrieval
    return a @ v


@pytest.mark.skipif(not _triton_available, reason="Triton is not available")
@pytest.mark.skipif(
    not _triton_available or gpu_capabilities_older_than_70(),
    reason="Triton requires a SM70+ GPU",
)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_mem_efficient_attention_parity(shape, dtype):
    q = torch.rand(shape, dtype=dtype, device=torch.device("cuda"))
    k = torch.rand(shape, dtype=dtype, device=torch.device("cuda"))
    v = torch.rand(shape, dtype=dtype, device=torch.device("cuda"))

    res_pytorch = attention_pytorch(q, k, v)
    res_me = mem_efficient_attention.apply(q, k, v, None)

    assert torch.allclose(res_pytorch, res_me)
    # TODO: test different sequence lengths for q and k
    # TODO: check parity with normal attention
