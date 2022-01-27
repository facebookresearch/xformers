# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import torch
import triton

from xformers.triton.k_sum import k_sum_0


def sum_2d_dim_0(x: torch.Tensor):
    """
    Sum a 2D tensor across the first dimension
    """

    out = torch.empty(x.shape[1], device=x.device, dtype=x.dtype)

    assert (
        x.ndim == 2
    ), "This is a very specific kernel, only for 2-dim tensors and summing along dim 0"
    M, N = x.shape

    # This kernel is not competitive for these sizes
    if M > 2048 or M < 8:
        return x.sum(dim=0)

    assert (
        M >= 4
    ), "This is a very specific kernel, requires the reduction dimension to be bigger than 4"

    assert x.stride(1) == 1, (
        "We're expecting x to be contiguous along dim 1, and non contiguous along dim 0.\n"
        " You would probably be better served with torch.sum()"
    )

    BLOCK_M = min(triton.next_power_of_2(M), 2048)
    BLOCK_N = 32
    if BLOCK_M > 256:
        BLOCK_N = 16
    if BLOCK_M > 1024:
        BLOCK_N = 8

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK_N"]),)

    # fmt: off
    k_sum_0[grid](
        out, x,
        x.stride(0),
        M, N,
        x.dtype == torch.float16,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_stages=4,
    )
    # fmt: on

    return out
