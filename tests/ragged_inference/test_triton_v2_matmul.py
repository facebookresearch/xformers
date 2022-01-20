# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch

from xformers.helpers.test_utils import assert_eq, bf16_cuda
from xformers.triton.ragged_inference.triton_v2_matmul import matmul


def _make_seq(n_ctx: int, value: int, d_model: int):
    return torch.full([n_ctx, d_model], value, **bf16_cuda())


SHAPES = [
    (384, 128),
    (784, 512),
    (1024, 1024),
    (2048, 384),
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_matmul(shape, dtype):
    a = torch.randn(shape, dtype=dtype, device="cuda")
    b = torch.randn(shape, dtype=dtype, device="cuda").T

    out = matmul(a, b)

    torch_out = torch.matmul(a, b)
    assert_eq(out, torch_out, rtol=0.05, atol=0.05)


"""
pytest -vxs --tb=native tests/ragged_inference/test_triton_v2_matmul.py -k test_matmul
"""
