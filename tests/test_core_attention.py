# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import functools

import pytest
import torch

from xformers.components.attention._sputnik_sparse import SparseCS
from xformers.components.attention.core import scaled_dot_product_attention


def catch_oor(fn):
    @functools.wraps(fn)
    def fn_and_catch_oor(*args, **kwargs):
        from triton import OutOfResources

        try:
            return fn(*args, **kwargs)
        except OutOfResources as e:
            pytest.skip(str(e))
            return None

    return fn_and_catch_oor


_devices = (
    ["cpu", "cuda"] if torch.cuda.is_available() and torch.version.cuda else ["cpu"]
)


def test_core_attention():
    b, s, d = 2, 400, 8
    prob = 0.95

    a = torch.rand(b, s, d)
    m = torch.rand(b, s, s) > prob
    m = m.to_sparse()

    # Check that the sparse and dense computations are equivalent
    r_sparse = scaled_dot_product_attention(a, a, a, m)
    r_dense = scaled_dot_product_attention(a, a, a, m.to_dense())

    assert torch.allclose(r_sparse, r_dense)


def test_core_attention_mask_types():
    b, s, d = 4, 90, 16
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

    # Test additive mask with mismatched batch dim
    d = b // 2
    mask = torch.rand(d, s, s) > prob
    float_mask_add = torch.zeros_like(mask, dtype=torch.float)
    float_mask_add = float_mask_add.masked_fill(mask, float("-inf"))

    # Make sure masking doesn't return errors
    r_dense_add = scaled_dot_product_attention(a, a, a, float_mask_add)


@pytest.mark.parametrize("device", _devices)
def test_amp_attention_dense_no_mask(device):
    b, s, d = 8, 64, 32

    a = torch.rand(b, s, d, device=device)

    with torch.amp.autocast("cuda"):
        r = scaled_dot_product_attention(a, a, a, att_mask=None)

    expected_device = torch.float16 if device == "cuda" else torch.float32
    assert r.dtype == expected_device


@pytest.mark.parametrize("device", _devices)
def test_amp_attention_dense(device):
    b, s, d = 8, 64, 32
    prob = 0.9

    a = torch.rand(b, s, d, device=device)
    m = torch.rand(s, s, device=device) > prob

    with torch.amp.autocast("cuda"):
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

    with torch.amp.autocast("cuda"):
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

    with torch.amp.autocast("cuda"):
        r = scaled_dot_product_attention(a, a, a, m)

    expected_device = torch.float32
    assert r.dtype == expected_device
