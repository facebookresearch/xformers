#!/usr/bin/env python3
"""
Test script to verify MPS support in xFormers
"""

import torch
import sys
import os

# Add the xformers path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

def test_mps_support():
    print("Testing xFormers MPS support...")

    # Check PyTorch MPS availability
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("✓ PyTorch MPS backend is available")

        # Create MPS tensors
        device = torch.device("mps")
        try:
            query = torch.randn(1, 32, 8, 64, device=device)
            key = torch.randn(1, 32, 8, 64, device=device)
            value = torch.randn(1, 32, 8, 64, device=device)

            print(f"✓ Created tensors on MPS device: {query.device}")

            # Test if xFormers can handle MPS tensors
            from xformers.ops import memory_efficient_attention

            # This should dispatch to CPU implementation for now
            with torch.no_grad():
                output = memory_efficient_attention(query, key, value)
                print(f"✓ Memory efficient attention succeeded on MPS: {output.shape}")
                print(f"  Output device: {output.device}")

        except Exception as e:
            print(f"✗ Error with MPS tensors: {e}")

    else:
        print("✗ PyTorch MPS backend is not available")

    # Test CPU fallback
    print("\nTesting CPU fallback...")
    try:
        query = torch.randn(1, 32, 8, 64)
        key = torch.randn(1, 32, 8, 64)
        value = torch.randn(1, 32, 8, 64)

        from xformers.ops import memory_efficient_attention
        with torch.no_grad():
            output = memory_efficient_attention(query, key, value)
            print(f"✓ CPU fallback works: {output.shape}")

    except Exception as e:
        print(f"✗ CPU fallback failed: {e}")

if __name__ == "__main__":
    test_mps_support()
