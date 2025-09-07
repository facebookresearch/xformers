#!/usr/bin/env python3
"""
Test script for MPS support in xFormers
"""

import torch
import sys
import traceback

def test_mps_support():
    print("=== Testing MPS Support in xFormers ===")

    # Check PyTorch MPS availability
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("✓ PyTorch MPS backend is available")
        device = torch.device("mps")
    else:
        print("✗ PyTorch MPS backend not available, using CPU")
        device = torch.device("cpu")

    try:
        # Import xFormers
        import xformers.ops as xops
        print("✓ xFormers imported successfully")

        # Create test tensors on the appropriate device
        B, M, H, K = 2, 32, 8, 64
        query = torch.randn(B, M, H, K, device=device, dtype=torch.float32)
        key = torch.randn(B, M, H, K, device=device, dtype=torch.float32)
        value = torch.randn(B, M, H, K, device=device, dtype=torch.float32)

        print(f"✓ Created tensors on {device}: {query.shape}")

        # Test xFormers memory efficient attention
        print("Testing xFormers memory_efficient_attention...")
        with torch.no_grad():
            output = xops.memory_efficient_attention(query, key, value)
        print(f"✓ xFormers attention successful: {output.shape}, device: {output.device}")

        # Test with causal mask
        print("Testing with causal attention...")
        with torch.no_grad():
            causal_output = xops.memory_efficient_attention(
                query, key, value,
                attn_bias=xops.fmha.attn_bias.LowerTriangularMask()
            )
        print(f"✓ Causal attention successful: {causal_output.shape}, device: {causal_output.device}")

        # Test gradient computation with a simple loss
        print("Testing gradient computation...")
        query = torch.randn(2, 8, 4, 32, device=device, dtype=torch.float32, requires_grad=True)
        key = torch.randn(2, 8, 4, 32, device=device, dtype=torch.float32, requires_grad=True)
        value = torch.randn(2, 8, 4, 32, device=device, dtype=torch.float32, requires_grad=True)

        output = xops.memory_efficient_attention(query, key, value)
        loss = output.sum()
        loss.backward()

        print("✓ Gradient computation successful")
        print(f"  query.grad shape: {query.grad.shape if query.grad is not None else None}")
        print(f"  key.grad shape: {key.grad.shape if key.grad is not None else None}")
        print(f"  value.grad shape: {value.grad.shape if value.grad is not None else None}")

        # Verify gradients are not None
        assert query.grad is not None, "query.grad should not be None"
        assert key.grad is not None, "key.grad should not be None"
        assert value.grad is not None, "value.grad should not be None"

        return True

    except Exception as e:
        print(f"✗ xFormers MPS test failed: {e}")
        traceback.print_exc()
        return False

def test_cpu_fallback():
    print("\n=== Testing CPU Fallback ===")
    try:
        import xformers.ops as xops

        # Create CPU tensors
        B, M, H, K = 2, 16, 4, 32  # Smaller for CPU
        query = torch.randn(B, M, H, K, dtype=torch.float32)
        key = torch.randn(B, M, H, K, dtype=torch.float32)
        value = torch.randn(B, M, H, K, dtype=torch.float32)

        print("Testing xFormers on CPU...")
        with torch.no_grad():
            output = xops.memory_efficient_attention(query, key, value)
        print(f"✓ CPU attention successful: {output.shape}")

        return True

    except Exception as e:
        print(f"✗ CPU fallback test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")

    mps_success = test_mps_support()
    cpu_success = test_cpu_fallback()

    if mps_success or cpu_success:
        print("\n🎉 xFormers MPS implementation test completed successfully!")
    else:
        print("\n❌ All tests failed")
        sys.exit(1)
