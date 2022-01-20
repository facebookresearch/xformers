# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import math

import pytest
import torch

from xformers.triton.k_ragged_qk_dotprod import ragged_qk_dotprod
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
# Shapes correspond to (B, M&N, L)
# TODO: test with M & N different from each other
SHAPES = [
    (1, 16, 16),
    (1, 384, 64),
    (8, 384, 128),
    (8, 784, 512),
    (16, 1024, 1024),
    (2, 2048, 384),
    (1, 18, 128),
]

DEBUG = True


def debug_plot_diff(a, b):
    if DEBUG:
        if a.ndim == 3:
            a = a[0]

        if b.ndim == 3:
            b = b[0]
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.heatmap(a.cpu() - b.cpu())
        plt.show()

    return torch.max(torch.abs(a - b))


def attention_pytorch(q, k):
    # attention matrix
    q = q / math.sqrt(q.size(-1))
    return q @ k.transpose(-2, -1)


@pytest.mark.skipif(not _triton_available, reason="Triton is not available")
@pytest.mark.skipif(
    not _triton_available or gpu_capabilities_older_than_70(),
    reason="Triton requires a SM70+ GPU",
)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_ragged_qk_parity(shape, dtype):
    q = torch.rand(shape, dtype=dtype, device=torch.device("cuda"))
    k = torch.rand(shape, dtype=dtype, device=torch.device("cuda"))

    res_pytorch = attention_pytorch(q, k)
    res_me = ragged_qk_dotprod(q=q, k=k)
    print(torch.abs(res_pytorch - res_me))

    assert torch.allclose(res_pytorch, res_me, rtol=1e-1), debug_plot_diff(
        res_pytorch, res_me
    )
    # TODO: test different sequence lengths for q and k


@pytest.mark.skipif(not _triton_available, reason="Triton is not available")
@pytest.mark.skipif(
    not _triton_available or gpu_capabilities_older_than_70(),
    reason="Triton requires a SM70+ GPU",
)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_ragged_qk_memory_use(shape, dtype):
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

    ragged_qk_dotprod(
        q=q,
        k=k,
    )
    torch.cuda.synchronize()

    max_memory_me = torch.cuda.max_memory_allocated() // 2 ** 20
    print(f"Memory efficient - Peak memory use: {max_memory_me}MB")

    fudge_factor = 3.0
    assert max_memory_torch == 0 or max_memory_me <= max_memory_torch * fudge_factor


"""
pytest -vxs --tb=native tests/test_ragged_qk_dotprod.py
"""
