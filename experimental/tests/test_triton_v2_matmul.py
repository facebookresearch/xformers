# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch
from ragged_inference.test_utils import assert_eq
from ragged_inference.triton_v2_matmul import matmul

SHAPES = [
    (3, 7),
    (384, 128),
    (784, 512),
    (1024, 1024),
    (2048, 384),
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_matmul(shape, dtype):
    a = torch.randn(shape, dtype=dtype, device="cuda")
    b = torch.randn(shape, dtype=dtype, device="cuda").T

    out = matmul(a, b)

    torch_out = torch.matmul(a, b)
    assert_eq(out, torch_out, rtol=0.01, atol=0.2)


"""
pytest -vxs --tb=native tests/ragged_inference/test_triton_v2_matmul.py -k test_matmul
"""
