# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch import nn

from xformers import _is_triton_available
from xformers.components.attention._sputnik_sparse import SparseCS
from xformers.components.attention.attention_mask import AttentionMask
from xformers.components.attention.core import scaled_dot_product_attention

if _is_triton_available():
    from xformers.triton.utils import gpu_capabilities_older_than_70

_is_blocksparse_available = (
    _is_triton_available() and not gpu_capabilities_older_than_70()
)

if _is_blocksparse_available:
    import triton.testing

_devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


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


@pytest.mark.skipif(
    not _is_blocksparse_available, reason="Blocksparse is not available"
)
@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("data_type", [torch.float16, torch.float32])
def test_switch_blocksparse(device, data_type):
    b, s, d = 8, 128, 32

    a = torch.rand(b, s, d, device=device, dtype=data_type)

    # Custom causal mask
    m_custom = torch.triu(
        torch.ones(s, s, device=device, dtype=a.dtype) * float("-inf"), diagonal=1
    )
    m_custom_bool = m_custom != float("-inf")
    m_sparse = SparseCS(m_custom_bool, device)
    # Mask with causal flag
    m_att_mask = AttentionMask.make_causal(s, s, device, dtype=a.dtype)

    def kernel():
        return scaled_dot_product_attention(a, a, a, m_att_mask)

    # Check that a switch to blocksparse is only triggered by causal flag
    with torch.cuda.amp.autocast():
        r_custom = scaled_dot_product_attention(a, a, a, m_custom)
        r_sparse = scaled_dot_product_attention(a, a, a, m_sparse)
        r_att_mask = triton.testing.catch_oor(kernel, pytest)

    expected_device = torch.float32
    assert r_sparse.dtype == expected_device

    if r_custom.dtype == r_att_mask.dtype:
        assert torch.allclose(r_custom, r_att_mask, atol=1e-6, rtol=1e-2)
    else:  # r_custom fp16, r_att_mask fp32
        assert torch.allclose(r_custom, r_att_mask.half(), atol=1e-6, rtol=1e-2)


@pytest.mark.skipif(
    not _is_blocksparse_available, reason="Blocksparse is not available"
)
@pytest.mark.parametrize("device", ["cuda"])
def test_switch_blocksparse_dims(device):
    b, s, d, nh = 8, 128, 32, 8
    hs = d // nh

    data_type = torch.float32
    a = torch.rand(b, nh, s, hs, device=device, dtype=data_type)
    # Mask with causal flag
    m = AttentionMask.make_causal(s, s, device, dtype=a.dtype)

    def kernel():
        return scaled_dot_product_attention(a, a, a, m)

    # Check that passing qkv with shape (B, nh, S, hs) is properly handled
    with torch.cuda.amp.autocast():
        r = triton.testing.catch_oor(kernel, pytest)

    expected_device = torch.float32
    assert r.dtype == expected_device


@pytest.mark.skipif(
    not _is_blocksparse_available, reason="Blocksparse is not available"
)
@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("drop_prob", [0.0, 0.3])
def test_switch_blocksparse_dropout(device, training, drop_prob):
    b, s, d = 8, 128, 32

    a = torch.rand(b, s, d, device=device)

    m = AttentionMask.make_causal(s, s, device)
    dropout = nn.Dropout(drop_prob)
    dropout.train(training).cuda()

    def kernel1():
        return scaled_dot_product_attention(a, a, a, m)

    def kernel2():
        return scaled_dot_product_attention(a, a, a, m, dropout)

    with torch.cuda.amp.autocast():
        r = triton.testing.catch_oor(kernel1, pytest)
        r_drop = triton.testing.catch_oor(kernel2, pytest)

    # Check for dropout when applicable
    if dropout.p and dropout.training:
        assert (r_drop != r).any()
    else:
        assert torch.allclose(r, r_drop)
