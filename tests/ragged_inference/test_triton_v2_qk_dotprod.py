# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch

from xformers.helpers.test_utils import assert_eq, bf16_cuda
from xformers.triton.ragged_inference.triton_v2_qk_dotprod import qk_dotprod


def _make_seq(n_ctx: int, value: int, d_model: int):
    return torch.full([n_ctx, d_model], value, **bf16_cuda())


SHAPES = [
    (3, 7),
    (384, 128),
    (784, 512),
    (1024, 1024),
    (2048, 384),
]


def qk_dotprod_pytorch(q, k):
    # attention matrix
    return torch.einsum("bqd,bkd->bqk", q, k)


def qk_dotprod_single_head_pytorch(q, k):
    # attention matrix
    return torch.einsum("qd,kd->qk", q, k)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_qk_dotprod(shape, dtype):
    a = torch.randn(shape, dtype=dtype, device="cuda")
    b = torch.randn(shape, dtype=dtype, device="cuda")

    out = qk_dotprod(a, b)

    torch_out = qk_dotprod_single_head_pytorch(a, b)
    assert_eq(out, torch_out, rtol=0.05, atol=0.05)


def test_simple_qk_dotprod():
    dtype = torch.float32
    shape = (8, 8)

    # a = torch.zeros(shape, dtype=dtype, device="cuda")
    # a[0,0] = 1.0
    # b = torch.randn(shape, dtype=dtype, device="cuda")

    k = torch.zeros(shape, dtype=dtype, device="cuda")
    k[0, 0] = 1.0
    k[0, 1] = 1.0
    q = torch.randn(shape, dtype=dtype, device="cuda")

    print(f"{q=}")
    print(f"{k=}")
    out = qk_dotprod(q, k)

    torch_out = qk_dotprod_single_head_pytorch(q, k)
    assert_eq(out, torch_out, rtol=0.05, atol=0.05)


"""
pytest -vxs --tb=native tests/ragged_inference/test_triton_v2_qk_dotprod.py -k test_qk_dotprod
"""
