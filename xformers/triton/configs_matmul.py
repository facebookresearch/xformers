# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch
import triton

from xformers.triton.utils import get_current_cuda_device

# CREDITS: Optimized defaults as suggested in the Triton documentation for matrix multiplications


# Handle different SM configurations for the older GPUs
# fmt: off
_configs_P100 = [
        triton.Config({'BLOCK_ROW': 64 , 'BLOCK_COL': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_ROW': 32 , 'BLOCK_COL': 64}, num_stages=5, num_warps=2)
]

_configs_V100 = _configs_P100 + [
        triton.Config({'BLOCK_ROW': 128, 'BLOCK_COL': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_ROW': 256, 'BLOCK_COL': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_ROW': 256, 'BLOCK_COL': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_ROW': 64 , 'BLOCK_COL': 256}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_ROW': 128, 'BLOCK_COL': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_ROW': 128, 'BLOCK_COL': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_ROW': 64 , 'BLOCK_COL': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_ROW': 128, 'BLOCK_COL': 32}, num_stages=4, num_warps=4),
]

# fmt: on

# NOTE: Could be that different configs would be better
_configs_A100 = _configs_V100

kernel_config = (
    {"P100": _configs_P100, "V100": _configs_V100, "A100": _configs_A100}[
        get_current_cuda_device()
    ]
    if torch.cuda.is_available()
    else {}
)
