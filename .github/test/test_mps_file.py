#!/usr/bin/env python3
"""
Test MPS operators by writing results to file
"""

import torch
import sys
import traceback

def test_mps():
    try:
        # Import our MPS operators
        from xformers.ops.fmha.mps import FwOp, BwOp
        result = "✓ MPS operators imported successfully\n"

        # Check device
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            result += "✓ MPS device available\n"
        else:
            device = torch.device("cpu")
            result += "Using CPU device\n"

        # Create test tensors
        B, M, H, K = 1, 8, 2, 32
        query = torch.randn(B, M, H, K, device=device, dtype=torch.float32)
        key = torch.randn(B, M, H, K, device=device, dtype=torch.float32)
        value = torch.randn(B, M, H, K, device=device, dtype=torch.float32)

        result += f"✓ Created tensors: {query.shape} on {device}\n"

        # Test forward operator
        from xformers.ops.fmha.common import Inputs
        inp = Inputs(query=query, key=key, value=value)

        result += "Testing forward operator...\n"
        output, ctx = FwOp.apply(inp, needs_gradient=False)
        result += f"✓ Forward pass successful: {output.shape}\n"

        result += "🎉 MPS operators working!\n"
        return result, True

    except Exception as e:
        error_msg = f"❌ Error: {e}\n{traceback.format_exc()}"
        return error_msg, False

if __name__ == "__main__":
    result, success = test_mps()

    # Write to file
    with open('/Users/jonathanhughes/source/xformers/mps_test_result.txt', 'w') as f:
        f.write(result)

    # Also print to stdout
    print(result)

    sys.exit(0 if success else 1)
