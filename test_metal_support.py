#!/usr/bin/env python3
"""
Test script to check MPS support and xFormers behavior
"""

import torch
import sys
import traceback

def check_pytorch_capabilities():
    print("=== PyTorch Capabilities ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if hasattr(torch.backends, 'mps'):
        print(f"MPS available: {torch.backends.mps.is_available()}")
        if torch.backends.mps.is_available():
            print("MPS device count:", torch.cuda.device_count() if torch.cuda.is_available() else "N/A")
    else:
        print("MPS backend not available in this PyTorch version")

    # Test native PyTorch attention on CPU
    print("\n=== Testing PyTorch Native Attention ===")
    try:
        # Create test tensors
        B, M, H, K = 2, 32, 8, 64
        query = torch.randn(B, M, H, K)
        key = torch.randn(B, M, H, K)
        value = torch.randn(B, M, H, K)

        # Test PyTorch's native scaled_dot_product_attention
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            print("Testing PyTorch native attention on CPU...")
            with torch.no_grad():
                output = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, attn_mask=None, dropout_p=0.0
                )
            print(f"✓ PyTorch native attention works: {output.shape}")
        else:
            print("PyTorch native attention not available")

    except Exception as e:
        print(f"✗ PyTorch native attention failed: {e}")
        traceback.print_exc()

def test_xformers_fallback():
    print("\n=== Testing xFormers Fallback ===")
    try:
        # Import xformers
        import xformers.ops as xops
        print("✓ xFormers imported successfully")

        # Create test tensors
        B, M, H, K = 2, 32, 8, 64
        query = torch.randn(B, M, H, K)
        key = torch.randn(B, M, H, K)
        value = torch.randn(B, M, H, K)

        print("Testing xFormers on CPU...")
        with torch.no_grad():
            output = xops.memory_efficient_attention(query, key, value)
        print(f"✓ xFormers works on CPU: {output.shape}")

    except Exception as e:
        print(f"✗ xFormers failed: {e}")
        traceback.print_exc()

def test_mps_if_available():
    print("\n=== Testing MPS Support ===")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:
            device = torch.device("mps")
            print(f"Testing with MPS device: {device}")

            # Create MPS tensors
            B, M, H, K = 2, 32, 8, 64
            query = torch.randn(B, M, H, K, device=device)
            key = torch.randn(B, M, H, K, device=device)
            value = torch.randn(B, M, H, K, device=device)

            print(f"✓ Created tensors on MPS: {query.device}")

            # Test PyTorch native attention on MPS
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                print("Testing PyTorch native attention on MPS...")
                with torch.no_grad():
                    output = torch.nn.functional.scaled_dot_product_attention(
                        query, key, value, attn_mask=None, dropout_p=0.0
                    )
                print(f"✓ PyTorch native attention on MPS: {output.shape}, device: {output.device}")

            # Test xFormers on MPS
            try:
                import xformers.ops as xops
                print("Testing xFormers on MPS...")
                with torch.no_grad():
                    output = xops.memory_efficient_attention(query, key, value)
                print(f"✓ xFormers on MPS: {output.shape}, device: {output.device}")
            except Exception as e:
                print(f"✗ xFormers on MPS failed: {e}")
                print("This is expected - xFormers doesn't have MPS-specific implementations yet")

        except Exception as e:
            print(f"✗ MPS test failed: {e}")
            traceback.print_exc()
    else:
        print("MPS not available, skipping MPS tests")

if __name__ == "__main__":
    check_pytorch_capabilities()
    test_xformers_fallback()
    test_mps_if_available()
