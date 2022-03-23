# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import triton
import triton.language as tl


# fmt: off
@triton.jit
def k_sum_0(
    Y, X,
    stride_xm,
    M, N,
    is_fp16,
    # META-params
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # fmt: om

    """
    Sum a 2d tensor over the first (strided) dimension.
    This extracts some speed through a parallel sum across the second dimension
    """

    # partial row indices. We'll reduce over this dimension
    m = tl.arange(0, BLOCK_M)

    # To get some extra parallelization, we handle several columns in the same thread block
    rn = tl.program_id(axis=0) * BLOCK_N + tl.arange(0, BLOCK_N)

    # the memory address of all the elements that we want to load can be computed as follows
    x_ptrs = X + m[:, None] * stride_xm + rn[None, :]
    x_sum = tl.zeros((BLOCK_N,), dtype=tl.float32)

    tiles = M // BLOCK_M
    if M % BLOCK_M > 0:
        tiles += 1

    col_mask = (rn[None, :] < N)

    for _ in range(tiles):
        # load input data; pad out-of-bounds elements with 0
        # NOTE: make sure to accumulate in fp32 to prevent a trivial overflow
        mask = (m[:, None] < M) & col_mask
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        x_sum += tl.sum(x, 0)

        # move the load pointer
        x_ptrs += BLOCK_M * stride_xm
        m += BLOCK_M  # update the mask check

    tl.store(Y + rn, x_sum, mask=rn < N)
