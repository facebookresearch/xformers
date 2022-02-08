# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

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


def _generate_qkvmask(tensor_type, device, dtype, dtype_mask):
    N, C, H, W, L = 8, 2, 64, 64, 32
    sparsity = 0.7

    shape0 = (N, C, H, W)
    shape1 = (N, C, H, L)
    shape2 = (N, C, W, L)

    if tensor_type != BlockSparseTensor:
        shape0 = shape0[1:]
        shape1 = shape1[1:]
        shape2 = shape2[1:]

    mask_sparse = _create_tensor(
        tensor_type, device, dtype=dtype_mask, shape=shape0, sparsity=sparsity
    )
    mask = mask_sparse.to_dense()

    query = torch.randn(shape1, dtype=dtype, device=device)
    key = torch.randn(shape2, dtype=dtype, device=device)
    value = torch.randn(shape2, dtype=dtype, device=device)

    return query, key, value, mask_sparse, mask


@pytest.mark.parametrize("tensor_type", _tensor_types)
@pytest.mark.parametrize("device", _devices)
def test_core_attention(tensor_type, device):
    query, key, value, mask_sparse, mask = _generate_qkvmask(
        tensor_type, device, torch.float32, torch.bool
    )

    # Check that the sparse and dense computations are equivalent
    r_sparse = scaled_dot_product_attention(query, key, value, mask_sparse)
    r_dense = scaled_dot_product_attention(query, key, value, mask)

    assert torch.allclose(r_sparse, r_dense, atol=1e-6)


@pytest.mark.parametrize("tensor_type", _tensor_types)
@pytest.mark.parametrize("device", _devices)
def test_core_attention_mask_types(tensor_type, device):
    # mask of bools
    query, key, value, mask_sparse, mask = _generate_qkvmask(
        tensor_type, device, torch.float32, torch.bool
    )
    r_sparse_bool = scaled_dot_product_attention(query, key, value, mask_sparse)
    r_dense_bool = scaled_dot_product_attention(query, key, value, mask)

    assert torch.allclose(r_dense_bool, r_sparse_bool, atol=1e-6)

    # Test additive mask. Mask of 0's and -infs.
    query, key, value, mask_sparse, mask = _generate_qkvmask(
        tensor_type, device, torch.float32, torch.float32
    )

    mask[mask == 0] = float("-inf")

    r_sparse_add = scaled_dot_product_attention(query, key, value, mask_sparse)
    r_dense_add = scaled_dot_product_attention(query, key, value, mask)

    # Now properly handled
    assert torch.allclose(r_dense_add, r_sparse_add, atol=1e-6)


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
def test_amp_attention_no_mask(device):
    b, s, d = 8, 64, 32

    a = torch.rand(b, s, d, device=device)
    m = None

    with torch.cuda.amp.autocast():
        r = scaled_dot_product_attention(a, a, a, m)

    expected_device = torch.float16 if device == "cuda" else torch.float32
    assert r.dtype == expected_device


@pytest.mark.parametrize("tensor_type", _tensor_types)
@pytest.mark.parametrize("device", _devices)
def test_amp_attention_sparse(tensor_type, device):
    if tensor_type in [BlockSparseTensor, CausalTensor] and device == "cuda":
        pytest.skip("Triton is currently broken for me on fp16")

    query, key, value, mask_sparse, mask = _generate_qkvmask(
        tensor_type, device, torch.float32, torch.bool
    )

    with torch.cuda.amp.autocast():
        r_sparse = scaled_dot_product_attention(query, key, value, mask_sparse)

    expected_dtype = torch.float32
    if tensor_type in [BlockSparseTensor, CausalTensor] and device == "cuda":
        expected_dtype = torch.float16

    assert r_sparse.dtype == expected_dtype
