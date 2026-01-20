# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for AETHER geometric sparse attention operator.
"""

import pytest
import torch

import xformers.ops as xops

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
compute_capability = (0, 0)
if torch.cuda.is_available():
    compute_capability = torch.cuda.get_device_capability("cuda")

requires_sm80 = pytest.mark.skipif(compute_capability < (8, 0), reason="requires sm80+")
parametrize_dtype = pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16], ids=["f16", "bf16"]
)

atol_rtol_kw = {
    torch.float16: {
        "rtol": 5e-3,
        "atol": 5e-3,
    },
    torch.bfloat16: {
        "rtol": 1e-1,
        "atol": 1e-1,
    },
}


def reference_attention(q, k, v, scale=None):
    """Reference dense attention implementation."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    q_scaled = q * scale
    attn = torch.einsum("bmhd,bnhd->bmhn", q_scaled, k)
    attn = torch.softmax(attn, dim=-1)
    out = torch.einsum("bmhn,bnhd->bmhd", attn, v)
    return out


class TestAetherAttention:
    """Test suite for AETHER geometric sparse attention."""

    @cuda_only
    def test_import(self):
        """Test that AETHER attention is importable from xformers.ops."""
        assert hasattr(xops, "aether_attention")
        assert hasattr(xops, "AetherAttention")

    @cuda_only
    @requires_sm80
    @parametrize_dtype
    def test_forward_shape(self, dtype):
        """Test that output shapes are correct."""
        B, M, H, D = 2, 128, 8, 64

        q = torch.randn(B, M, H, D, device="cuda", dtype=dtype)
        k = torch.randn(B, M, H, D, device="cuda", dtype=dtype)
        v = torch.randn(B, M, H, D, device="cuda", dtype=dtype)

        out = xops.aether_attention(q, k, v, threshold=0.15)

        assert out.shape == q.shape
        assert out.dtype == q.dtype
        assert out.device == q.device

    @cuda_only
    @requires_sm80
    @parametrize_dtype
    def test_correctness_vs_reference(self, dtype):
        """Test numerical correctness against reference dense attention."""
        B, M, H, D = 2, 64, 4, 32

        torch.manual_seed(42)
        q = torch.randn(B, M, H, D, device="cuda", dtype=dtype)
        k = torch.randn(B, M, H, D, device="cuda", dtype=dtype)
        v = torch.randn(B, M, H, D, device="cuda", dtype=dtype)

        # With threshold=0 (no pruning), should match reference exactly
        out_aether = xops.aether_attention(q, k, v, threshold=-float("inf"))
        out_ref = reference_attention(q, k, v)

        torch.testing.assert_close(
            out_aether,
            out_ref,
            **atol_rtol_kw[dtype],
            msg="AETHER output differs from reference"
        )

    @cuda_only
    @requires_sm80
    @parametrize_dtype
    def test_gradient_computation(self, dtype):
        """Test that gradients are computed correctly."""
        B, M, H, D = 2, 64, 4, 32

        q = torch.randn(B, M, H, D, device="cuda", dtype=dtype, requires_grad=True)
        k = torch.randn(B, M, H, D, device="cuda", dtype=dtype, requires_grad=True)
        v = torch.randn(B, M, H, D, device="cuda", dtype=dtype, requires_grad=True)

        out = xops.aether_attention(q, k, v, threshold=0.15)
        loss = out.sum()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None
        assert q.grad.shape == q.shape
        assert k.grad.shape == k.shape
        assert v.grad.shape == v.shape

    @cuda_only
    @requires_sm80
    @parametrize_dtype
    def test_module_interface(self, dtype):
        """Test the nn.Module interface."""
        B, M, H, D = 2, 128, 8, 64

        attn = xops.AetherAttention(sparsity_threshold=0.15, block_size=64)

        q = torch.randn(B, M, H, D, device="cuda", dtype=dtype)
        k = torch.randn(B, M, H, D, device="cuda", dtype=dtype)
        v = torch.randn(B, M, H, D, device="cuda", dtype=dtype)

        out = attn(q, k, v)

        assert out.shape == q.shape

    @cuda_only
    @requires_sm80
    @pytest.mark.parametrize("seq_len", [64, 128, 256, 512])
    def test_varying_sequence_lengths(self, seq_len):
        """Test with different sequence lengths."""
        B, H, D = 2, 4, 64
        dtype = torch.float16

        q = torch.randn(B, seq_len, H, D, device="cuda", dtype=dtype)
        k = torch.randn(B, seq_len, H, D, device="cuda", dtype=dtype)
        v = torch.randn(B, seq_len, H, D, device="cuda", dtype=dtype)

        out = xops.aether_attention(q, k, v, threshold=0.15)

        assert out.shape == (B, seq_len, H, D)

    @cuda_only
    @requires_sm80
    def test_different_kv_length(self):
        """Test with different query and key/value lengths."""
        B, M, N, H, D = 2, 64, 128, 4, 64
        dtype = torch.float16

        q = torch.randn(B, M, H, D, device="cuda", dtype=dtype)
        k = torch.randn(B, N, H, D, device="cuda", dtype=dtype)
        v = torch.randn(B, N, H, D, device="cuda", dtype=dtype)

        out = xops.aether_attention(q, k, v, threshold=0.15)

        assert out.shape == (B, M, H, D)

    @cuda_only
    @requires_sm80
    @pytest.mark.parametrize("threshold", [0.0, 0.1, 0.5, 1.0])
    def test_threshold_values(self, threshold):
        """Test different threshold values."""
        B, M, H, D = 2, 64, 4, 32
        dtype = torch.float16

        q = torch.randn(B, M, H, D, device="cuda", dtype=dtype)
        k = torch.randn(B, M, H, D, device="cuda", dtype=dtype)
        v = torch.randn(B, M, H, D, device="cuda", dtype=dtype)

        # Should not raise
        out = xops.aether_attention(q, k, v, threshold=threshold)
        assert out.shape == q.shape
        assert not torch.isnan(out).any()
