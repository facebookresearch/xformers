# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import math

import pytest
import torch

from xformers.triton.k_ragged_qk_dotprod import mem_efficient_fw
from xformers.triton.utils import gpu_capabilities_older_than_70

_triton_available = True
# except ImportError as e:
#     err = f"Triton is not available, some optimizations will "f"not be tested.{e}"
#     logging.warning(
#         err
#     )
#     print(err)
#     _triton_available = False


# Testing odd shapes on purpose
SHAPES = [
    (1, 384, 128),
    (8, 384, 128),
    (8, 784, 512),
    (16, 1024, 1024),
    (2, 2048, 384),
    # (4, 3136, 1024), # FIXME
]


def attention_pytorch(q, k):
    # attention matrix
    q = q / math.sqrt(q.size(-1))
    a = q @ k.transpose(-2, -1)
    return a


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

    res_pytorch = attention_pytorch(q, k)
    res_me = mem_efficient_fw(q=q, k=k)

    assert torch.mean(torch.abs(res_pytorch - res_me)) < 0.2

    # assert torch.allclose(res_pytorch, res_me, rtol=1e-1) FIXME
    # TODO: test different sequence lengths for q and k
    # TODO: check parity with normal attention


@pytest.mark.skipif(not _triton_available, reason="Triton is not available")
@pytest.mark.skipif(
    not _triton_available or gpu_capabilities_older_than_70(),
    reason="Triton requires a SM70+ GPU",
)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_mem_efficient_attention_memory_use(shape, dtype):
    # FW a random bunch of data
    q = torch.rand(shape, dtype=dtype, device=torch.device("cuda"))
    k = torch.rand(shape, dtype=dtype, device=torch.device("cuda"))

    # Vanilla attention
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    _ = attention_pytorch(q, k)
    torch.cuda.synchronize()
    max_memory_torch = torch.cuda.max_memory_allocated() // 2 ** 20
    print(f"Dense - Peak memory use: {max_memory_torch}MB")

    # Mem efficient attention
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    mem_efficient_fw(
        q=q,
        k=k,
    )
    torch.cuda.synchronize()

    max_memory_me = torch.cuda.max_memory_allocated() // 2 ** 20
    print(f"Memory efficient - Peak memory use: {max_memory_me}MB")

    fudge_factor = 3.0
    assert max_memory_me <= max_memory_torch * fudge_factor


"""
pytest -vxs --tb=native tests/test_ragged_qk_dotprod.py
"""
