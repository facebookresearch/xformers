# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


# CREDITS: This comes almost as-is from the Triton dropout tutorial
# https://raw.githubusercontent.com/openai/triton/master/python/tutorials/04-low-memory-dropout.py

import triton
import triton.language as tl


# fmt: off
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE" : 256}, num_warps=1),
        triton.Config({"BLOCK_SIZE" : 512}, num_warps=2),
        triton.Config({"BLOCK_SIZE" : 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE" : 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE" : 4096}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def k_dropout(
    Y, X, SEEDS,
    stride,
    N,
    p,
    **meta,
):
    """
    Apply dropout on an input tensor
    Y : Output (M, N)
    X : Input (M, N)
    S : Seeds (M,)
    p : dropout probability
    """
    # fmt: on

    # compute memory offsets of elements handled by this instance
    BLOCK_SIZE = meta["BLOCK_SIZE"]
    row = tl.program_id(axis=0)
    col = tl.program_id(axis=1)
    offsets = row * stride + col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) < N

    # load data from x
    x_ptrs = X + offsets
    x = tl.load(x_ptrs, mask=mask)

    # randomly prune it
    seed = SEEDS + row
    random = tl.rand(seed.to(tl.int32), offsets)
    x_keep = random > p

    # write-back
    zero = 0.
    zero = zero.to(x.dtype)
    output = tl.where(x_keep, (x / (1 - p)).to(x.dtype), zero)
    y_ptrs = Y + offsets
    tl.store(y_ptrs, output, mask=mask)
