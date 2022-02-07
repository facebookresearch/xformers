# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

# from xformers.components.attention._sputnik_sparse import SparseCS
from xformers.components.attention.core import scaled_dot_product_attention
from xformers.sparse import (
    BlockSparseTensor,
    CausalTensor,
    SparseCOOTensor,
    SparseCSRTensor,
)
from xformers.testing import _create_tensor

_devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
_tensor_types = [BlockSparseTensor, SparseCSRTensor, SparseCOOTensor, CausalTensor]


@pytest.mark.parametrize("tensor_type", _tensor_types)
@pytest.mark.parametrize("device", _devices)
def test_core_attention(tensor_type, device):
    N, C, H, W, L = 8, 2, 64, 64, 32
    sparsity = 0.7
    dtype = torch.float32

    shape0 = (N, C, H, W)
    shape1 = (N, C, H, L)
    shape2 = (N, C, W, L)

    if tensor_type != BlockSparseTensor:
        shape0 = shape0[1:]
        shape1 = shape1[1:]
        shape2 = shape2[1:]

    mask_sparse = _create_tensor(
        tensor_type, device, dtype=torch.bool, shape=shape0, sparsity=sparsity
    )
    mask = mask_sparse.to_dense()

    query = torch.randn(shape1, dtype=dtype, device=device)
    key = torch.randn(shape2, dtype=dtype, device=device)
    value = torch.randn(shape2, dtype=dtype, device=device)

    # Check that the sparse and dense computations are equivalent
    r_sparse = scaled_dot_product_attention(query, key, value, mask_sparse)
    r_dense = scaled_dot_product_attention(query, key, value, mask)

    assert torch.allclose(r_sparse, r_dense, atol=1e-6)


def test_core_attention_mask_types():

    b, s, d = 8, 900, 32
    prob = 0.8  # make sure that we trigger the sparse kernels

    a = torch.rand(b, s, d)
    mask = torch.rand(b, s, s) > prob

    # mask of bools
    r_dense_bool = scaled_dot_product_attention(a, a, a, mask)
    r_sparse_bool = scaled_dot_product_attention(a, a, a, mask.to_sparse())
    assert torch.allclose(r_dense_bool, r_sparse_bool)

    # Test additive mask. Mask of 0's and -infs.
    float_mask_add = torch.zeros_like(mask, dtype=torch.float)
    float_mask_add = float_mask_add.masked_fill(mask, float("-inf"))

    r_dense_add = scaled_dot_product_attention(a, a, a, float_mask_add)
    r_sparse_add = scaled_dot_product_attention(a, a, a, float_mask_add.to_sparse())

    # Now properly handled
    assert torch.allclose(r_dense_add, r_sparse_add)


"""
@pytest.mark.parametrize("device", _devices)
def test_amp_attention_dense_no_mask(device):
    b, s, d = 8, 64, 32

    a = torch.rand(b, s, d, device=device)

    with torch.cuda.amp.autocast():
        r = scaled_dot_product_attention(a, a, a, att_mask=None)

    expected_device = torch.float16 if device == "cuda" else torch.float32
    assert r.dtype == expected_device


@pytest.mark.parametrize("device", _devices)
def test_amp_attention_dense(device):
    b, s, d = 8, 64, 32
    prob = 0.9

    a = torch.rand(b, s, d, device=device)
    m = torch.rand(s, s, device=device) > prob

    with torch.cuda.amp.autocast():
        r = scaled_dot_product_attention(a, a, a, m)

    expected_device = torch.float16 if device == "cuda" else torch.float32
    assert r.dtype == expected_device


@pytest.mark.parametrize("device", _devices)
def test_amp_attention_sparse(device):
    b, s, d = 8, 64, 32
    prob = 0.9

    a = torch.rand(b, s, d, device=device)
    m = torch.rand(s, s, device=device) > prob
    m = m.to_sparse()

    with torch.cuda.amp.autocast():
        r = scaled_dot_product_attention(a, a, a, m)

    expected_device = torch.float32
    assert r.dtype == expected_device


@pytest.mark.parametrize("device", _devices)
def test_amp_attention_sparsecs(device):
    b, s, d = 8, 64, 32
    prob = 0.9

    a = torch.rand(b, s, d, device=device)
    m = torch.rand(s, s, device=device) > prob
    m = SparseCS(m, device)

    with torch.cuda.amp.autocast():
        r = scaled_dot_product_attention(a, a, a, m)

    expected_device = torch.float32
    assert r.dtype == expected_device
"""
