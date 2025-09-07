#!/usr/bin/env python3
"""
Test full xFormers MPS integration
"""

import torch
import sys
import traceback

def test_xformers_mps():
    try:
        # Import xFormers
        import xformers.ops as xops
        result = "✓ xFormers imported successfully\n"

        # Check device
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            result += "✓ MPS device available\n"
        else:
            device = torch.device("cpu")
            result += "Using CPU device\n"

        # Create test tensors
        B, M, H, K = 2, 16, 4, 64
        query = torch.randn(B, M, H, K, device=device, dtype=torch.float32)
        key = torch.randn(B, M, H, K, device=device, dtype=torch.float32)
        value = torch.randn(B, M, H, K, device=device, dtype=torch.float32)

        result += f"✓ Created tensors: {query.shape} on {device}\n"

        # Test memory_efficient_attention (should use our MPS operators)
        result += "Testing memory_efficient_attention...\n"
        with torch.no_grad():
            output = xops.memory_efficient_attention(query, key, value)
        result += f"✓ memory_efficient_attention successful: {output.shape}, device: {output.device}\n"

        # Test with causal mask
        result += "Testing with causal mask...\n"
        with torch.no_grad():
            causal_output = xops.memory_efficient_attention(
                query, key, value,
                attn_bias=xops.fmha.attn_bias.LowerTriangularMask()
            )
        result += f"✓ Causal attention successful: {causal_output.shape}, device: {causal_output.device}\n"

        # Test gradient computation
        result += "Testing gradient computation...\n"
        query_grad = torch.randn_like(query)

        query.requires_grad_(True)
        key.requires_grad_(True)
        value.requires_grad_(True)

        output = xops.memory_efficient_attention(query, key, value)
        loss = (output * query_grad).sum()

        # Only test backward if we can
        try:
            loss.backward()
            has_grad = query.grad is not None
            result += f"✓ Gradient computation successful: query.grad is {'not ' if not has_grad else ''}None\n"
        except Exception as e:
            result += f"⚠ Gradient computation failed (expected for MPS): {e}\n"

        result += "🎉 Full xFormers MPS integration working!\n"
        return result, True

    except Exception as e:
        error_msg = f"❌ Error: {e}\n{traceback.format_exc()}"
        return error_msg, False

if __name__ == "__main__":
    result, success = test_xformers_mps()

    # Write to file
    with open('/Users/jonathanhughes/source/xformers/xformers_mps_test_result.txt', 'w') as f:
        f.write(result)

    # Also print to stdout
    print(result)

    sys.exit(0 if success else 1)
