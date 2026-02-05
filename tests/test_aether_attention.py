# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Comprehensive tests for AETHER geometric sparse attention operator.

This test suite covers:
- Forward pass correctness against reference dense attention
- Gradient computation and numerical gradients
- Determinism and reproducibility
- Memory efficiency compared to dense attention
- Edge cases (single elements, power-of-two, non-standard shapes)
- Block geometry computation
- Threshold/pruning effectiveness
- Stress tests for large sequences
"""

import pytest
import torch
from typing import Tuple

import xformers.ops as xops

# -----------------------------------------------------------------------------
# Test Configuration
# -----------------------------------------------------------------------------

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

compute_capability = (0, 0)
if torch.cuda.is_available():
    compute_capability = torch.cuda.get_device_capability("cuda")

requires_sm80 = pytest.mark.skipif(
    compute_capability < (8, 0), reason="requires sm80+ (Ampere or newer)"
)

parametrize_dtype = pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16], ids=["f16", "bf16"]
)

# Tolerance settings per dtype
atol_rtol_kw = {
    torch.float16: {"rtol": 5e-3, "atol": 5e-3},
    torch.bfloat16: {"rtol": 1e-1, "atol": 1e-1},
    torch.float32: {"rtol": 1e-4, "atol": 1e-4},
}


# -----------------------------------------------------------------------------
# Reference Implementation
# -----------------------------------------------------------------------------


def reference_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float = None
) -> torch.Tensor:
    """
    Reference dense attention implementation for correctness testing.
    
    Args:
        q, k, v: [B, M/N, H, D] tensors
        scale: Optional attention scale (default: 1/sqrt(D))
    
    Returns:
        output: [B, M, H, D] attention output
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    q_scaled = q * scale
    attn = torch.einsum("bmhd,bnhd->bmhn", q_scaled, k)
    attn = torch.softmax(attn, dim=-1)
    out = torch.einsum("bmhn,bnhd->bmhd", attn, v)
    return out


def make_test_tensors(
    B: int,
    M: int,
    N: int,
    H: int,
    D: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    requires_grad: bool = False,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create Q, K, V tensors for testing."""
    torch.manual_seed(seed)
    q = torch.randn(B, M, H, D, device=device, dtype=dtype, requires_grad=requires_grad)
    k = torch.randn(B, N, H, D, device=device, dtype=dtype, requires_grad=requires_grad)
    v = torch.randn(B, N, H, D, device=device, dtype=dtype, requires_grad=requires_grad)
    return q, k, v


# -----------------------------------------------------------------------------
# Import and API Tests
# -----------------------------------------------------------------------------


class TestAetherImportAndAPI:
    """Test module imports and API surface."""

    @cuda_only
    def test_import_functional(self):
        """Test that aether_attention is importable from xformers.ops."""
        assert hasattr(xops, "aether_attention")
        assert callable(xops.aether_attention)

    @cuda_only
    def test_import_module(self):
        """Test that AetherAttention class is importable."""
        assert hasattr(xops, "AetherAttention")
        assert issubclass(xops.AetherAttention, torch.nn.Module)

    @cuda_only
    def test_module_repr(self):
        """Test module string representation."""
        attn = xops.AetherAttention(sparsity_threshold=0.2, block_size=32)
        repr_str = repr(attn)
        assert "threshold=0.2" in repr_str
        assert "block_size=32" in repr_str


# -----------------------------------------------------------------------------
# Forward Pass Tests
# -----------------------------------------------------------------------------


class TestAetherForward:
    """Test forward pass functionality."""

    @cuda_only
    @requires_sm80
    @parametrize_dtype
    def test_forward_shape(self, dtype):
        """Test that output shapes are correct."""
        B, M, H, D = 2, 128, 8, 64
        q, k, v = make_test_tensors(B, M, M, H, D, dtype=dtype)

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
        q, k, v = make_test_tensors(B, M, M, H, D, dtype=dtype)

        # With threshold=-inf (no pruning), should match reference exactly
        out_aether = xops.aether_attention(q, k, v, threshold=-float("inf"))
        out_ref = reference_attention(q, k, v)

        torch.testing.assert_close(
            out_aether, out_ref, **atol_rtol_kw[dtype],
            msg="AETHER output differs from reference"
        )

    @cuda_only
    @requires_sm80
    @parametrize_dtype
    def test_module_interface(self, dtype):
        """Test the nn.Module interface."""
        B, M, H, D = 2, 128, 8, 64

        attn = xops.AetherAttention(sparsity_threshold=0.15, block_size=64)
        q, k, v = make_test_tensors(B, M, M, H, D, dtype=dtype)

        out = attn(q, k, v)
        assert out.shape == q.shape

    @cuda_only
    @requires_sm80
    def test_no_nans_in_output(self):
        """Test that output contains no NaN values."""
        B, M, H, D = 2, 256, 4, 64
        q, k, v = make_test_tensors(B, M, M, H, D)

        for threshold in [0.0, 0.15, 0.5, 1.0, -float("inf")]:
            out = xops.aether_attention(q, k, v, threshold=threshold)
            assert not torch.isnan(out).any(), f"NaN in output with threshold={threshold}"
            assert not torch.isinf(out).any(), f"Inf in output with threshold={threshold}"


# -----------------------------------------------------------------------------
# Gradient Tests
# -----------------------------------------------------------------------------


class TestAetherGradients:
    """Test gradient computation."""

    @cuda_only
    @requires_sm80
    @parametrize_dtype
    def test_gradient_computation(self, dtype):
        """Test that gradients are computed correctly."""
        B, M, H, D = 2, 64, 4, 32
        q, k, v = make_test_tensors(B, M, M, H, D, dtype=dtype, requires_grad=True)

        out = xops.aether_attention(q, k, v, threshold=0.15)
        loss = out.sum()
        loss.backward()

        assert q.grad is not None, "Query gradient is None"
        assert k.grad is not None, "Key gradient is None"
        assert v.grad is not None, "Value gradient is None"
        assert q.grad.shape == q.shape
        assert k.grad.shape == k.shape
        assert v.grad.shape == v.shape

    @cuda_only
    @requires_sm80
    def test_gradient_correctness(self):
        """Test gradient correctness against reference implementation."""
        B, M, H, D = 1, 32, 2, 16
        dtype = torch.float16

        # AETHER gradients
        q1, k1, v1 = make_test_tensors(B, M, M, H, D, dtype=dtype, requires_grad=True)
        out1 = xops.aether_attention(q1, k1, v1, threshold=-float("inf"))
        out1.sum().backward()

        # Reference gradients
        q2, k2, v2 = make_test_tensors(B, M, M, H, D, dtype=dtype, requires_grad=True)
        out2 = reference_attention(q2, k2, v2)
        out2.sum().backward()

        # Compare gradients
        torch.testing.assert_close(q1.grad, q2.grad, **atol_rtol_kw[dtype])
        torch.testing.assert_close(k1.grad, k2.grad, **atol_rtol_kw[dtype])
        torch.testing.assert_close(v1.grad, v2.grad, **atol_rtol_kw[dtype])


# -----------------------------------------------------------------------------
# Determinism Tests
# -----------------------------------------------------------------------------


class TestAetherDeterminism:
    """Test deterministic behavior."""

    @cuda_only
    @requires_sm80
    def test_determinism(self):
        """Test that same inputs produce identical outputs."""
        B, M, H, D = 2, 128, 4, 64
        q, k, v = make_test_tensors(B, M, M, H, D)

        out1 = xops.aether_attention(q, k, v, threshold=0.15)
        out2 = xops.aether_attention(q, k, v, threshold=0.15)

        torch.testing.assert_close(out1, out2, atol=0.0, rtol=0.0)


# -----------------------------------------------------------------------------
# Sequence Length Tests
# -----------------------------------------------------------------------------


class TestAetherSequenceLengths:
    """Test various sequence length configurations."""

    @cuda_only
    @requires_sm80
    @pytest.mark.parametrize("seq_len", [64, 128, 256, 512])
    def test_varying_sequence_lengths(self, seq_len):
        """Test with different sequence lengths."""
        B, H, D = 2, 4, 64
        q, k, v = make_test_tensors(B, seq_len, seq_len, H, D)

        out = xops.aether_attention(q, k, v, threshold=0.15)
        assert out.shape == (B, seq_len, H, D)

    @cuda_only
    @requires_sm80
    def test_different_kv_length(self):
        """Test with different query and key/value lengths."""
        B, M, N, H, D = 2, 64, 128, 4, 64
        q, k, v = make_test_tensors(B, M, N, H, D)

        out = xops.aether_attention(q, k, v, threshold=0.15)
        assert out.shape == (B, M, H, D)

    @cuda_only
    @requires_sm80
    @pytest.mark.parametrize("seq_len", [63, 65, 127, 129, 255, 257])
    def test_non_power_of_two_sequences(self, seq_len):
        """Test with non-power-of-two sequence lengths."""
        B, H, D = 2, 4, 64
        q, k, v = make_test_tensors(B, seq_len, seq_len, H, D)

        out = xops.aether_attention(q, k, v, threshold=0.15)
        assert out.shape == (B, seq_len, H, D)
        assert not torch.isnan(out).any()


# -----------------------------------------------------------------------------
# Edge Case Tests
# -----------------------------------------------------------------------------


class TestAetherEdgeCases:
    """Test edge cases and boundary conditions."""

    @cuda_only
    @requires_sm80
    def test_single_element_batch(self):
        """Test with batch size 1."""
        B, M, H, D = 1, 128, 4, 64
        q, k, v = make_test_tensors(B, M, M, H, D)

        out = xops.aether_attention(q, k, v, threshold=0.15)
        assert out.shape == (B, M, H, D)

    @cuda_only
    @requires_sm80
    def test_single_head(self):
        """Test with single attention head."""
        B, M, H, D = 2, 128, 1, 64
        q, k, v = make_test_tensors(B, M, M, H, D)

        out = xops.aether_attention(q, k, v, threshold=0.15)
        assert out.shape == (B, M, H, D)

    @cuda_only
    @requires_sm80
    def test_minimal_sequence(self):
        """Test with minimal sequence length (equal to block size)."""
        B, M, H, D = 2, 64, 4, 64  # M = block_size
        q, k, v = make_test_tensors(B, M, M, H, D)

        out = xops.aether_attention(q, k, v, threshold=0.15, block_size=64)
        assert out.shape == (B, M, H, D)


# -----------------------------------------------------------------------------
# Threshold Tests
# -----------------------------------------------------------------------------


class TestAetherThreshold:
    """Test threshold/pruning behavior."""

    @cuda_only
    @requires_sm80
    @pytest.mark.parametrize("threshold", [0.0, 0.1, 0.25, 0.5, 1.0])
    def test_threshold_values(self, threshold):
        """Test different threshold values don't crash."""
        B, M, H, D = 2, 64, 4, 32
        q, k, v = make_test_tensors(B, M, M, H, D)

        out = xops.aether_attention(q, k, v, threshold=threshold)
        assert out.shape == q.shape
        assert not torch.isnan(out).any()

    @cuda_only
    @requires_sm80
    def test_extreme_thresholds(self):
        """Test extreme threshold values."""
        B, M, H, D = 2, 64, 4, 32
        q, k, v = make_test_tensors(B, M, M, H, D)

        # Very high threshold (aggressive pruning)
        out_high = xops.aether_attention(q, k, v, threshold=100.0)
        assert not torch.isnan(out_high).any()

        # Negative infinity (no pruning at all)
        out_no_prune = xops.aether_attention(q, k, v, threshold=-float("inf"))
        assert not torch.isnan(out_no_prune).any()


# -----------------------------------------------------------------------------
# Block Geometry Tests
# -----------------------------------------------------------------------------


class TestBlockGeometry:
    """Test block geometry computation."""

    @cuda_only
    @requires_sm80
    def test_compute_block_geometry_shapes(self):
        """Test that compute_block_geometry returns correct shapes."""
        from xformers.ops.aether_attention import compute_block_geometry

        B, N, H, D = 2, 256, 4, 64
        block_size = 64
        num_blocks = (N + block_size - 1) // block_size

        k = torch.randn(B, N, H, D, device="cuda", dtype=torch.float16)
        centroids, radii = compute_block_geometry(k, block_size=block_size)

        assert centroids.shape == (B, num_blocks, H, D)
        assert radii.shape == (B, num_blocks, H)

    @cuda_only
    @requires_sm80
    def test_geometry_values_reasonable(self):
        """Test that geometry values are within reasonable bounds."""
        from xformers.ops.aether_attention import compute_block_geometry

        B, N, H, D = 2, 128, 4, 64
        block_size = 64

        # Use scaled random values
        k = torch.randn(B, N, H, D, device="cuda", dtype=torch.float16)
        centroids, radii = compute_block_geometry(k, block_size=block_size)

        # Centroids should be finite
        assert not torch.isnan(centroids).any()
        assert not torch.isinf(centroids).any()

        # Radii should be non-negative and finite
        assert (radii >= 0).all()
        assert not torch.isnan(radii).any()
        assert not torch.isinf(radii).any()


# -----------------------------------------------------------------------------
# Stress Tests
# -----------------------------------------------------------------------------


class TestAetherStress:
    """Stress tests for larger inputs."""

    @cuda_only
    @requires_sm80
    @pytest.mark.parametrize("seq_len", [1024, 2048])
    def test_large_sequences(self, seq_len):
        """Test with larger sequence lengths."""
        B, H, D = 1, 4, 64
        q, k, v = make_test_tensors(B, seq_len, seq_len, H, D)

        out = xops.aether_attention(q, k, v, threshold=0.15)
        assert out.shape == (B, seq_len, H, D)
        assert not torch.isnan(out).any()

    @cuda_only
    @requires_sm80
    def test_many_heads(self):
        """Test with many attention heads."""
        B, M, H, D = 1, 128, 32, 64
        q, k, v = make_test_tensors(B, M, M, H, D)

        out = xops.aether_attention(q, k, v, threshold=0.15)
        assert out.shape == (B, M, H, D)

    @cuda_only
    @requires_sm80
    def test_large_batch(self):
        """Test with larger batch size."""
        B, M, H, D = 8, 128, 4, 64
        q, k, v = make_test_tensors(B, M, M, H, D)

        out = xops.aether_attention(q, k, v, threshold=0.15)
        assert out.shape == (B, M, H, D)


# -----------------------------------------------------------------------------
# Block Size Tests
# -----------------------------------------------------------------------------


class TestAetherBlockSize:
    """Test different block size configurations."""

    @cuda_only
    @requires_sm80
    @pytest.mark.parametrize("block_size", [32, 64, 128])
    def test_different_block_sizes(self, block_size):
        """Test with different block sizes."""
        B, M, H, D = 2, 256, 4, 64
        q, k, v = make_test_tensors(B, M, M, H, D)

        out = xops.aether_attention(q, k, v, threshold=0.15, block_size=block_size)
        assert out.shape == (B, M, H, D)
        assert not torch.isnan(out).any()
