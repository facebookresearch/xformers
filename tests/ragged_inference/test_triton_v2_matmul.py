# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch

from xformers.helpers.test_utils import assert_eq, bf16_cuda
from xformers.triton.ragged_inference.triton_v2_matmul import matmul


def _make_seq(n_ctx: int, value: int, d_model: int):
    return torch.full([n_ctx, d_model], value, **bf16_cuda())


def test_matmul():
    K = 128
    M = 16
    N = 8
    a = torch.randn(M, K, **bf16_cuda())
    b = torch.randn(K, N, **bf16_cuda())
    out = matmul(a, b)

    torch_out = torch.matmul(a, b)
    assert_eq(out, torch_out)


"""
pytest -vxs --tb=native tests/ragged_inference/test_triton_v2_matmul.py -k test_matmul
"""
