# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


# CREDITS: This comes almost as-is from the Triton layer norm tutorial
# https://github.com/openai/triton/blob/master/python/tutorials/05-layer-norm.py


import triton
import triton.language as tl


# fmt: off
@triton.jit
def rms_norm_fw(X, Y, W, V, stride, N, eps, BLOCK_SIZE_N: tl.constexpr):
    # fmt: on
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N

    # Move to this row
    x_ptrs = X + row * stride + cols
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    x_var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(x_var + eps)

    y = x * rstd
    tl.store(V + row, rstd)

    mask = cols < N
    w = tl.load(W + cols, mask=mask, other=1.0)
    y = y * w

    y_ptrs = Y + row * stride + cols
    tl.store(y_ptrs, y, mask=mask)


# Backward pass (DX + partial DW)
# fmt: off
@triton.jit
def rms_norm_bwd_dx_fused(
    DX, DY, DW,
    X, W, V,
    Lock, stride, N,
    # META-parameters
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # fmt: on

    # position of elements processed by this program
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N

    # offset data pointers to start at the row of interest
    x_ptrs = X + row * stride + cols
    dy_ptrs = DY + row * stride + cols

    # load data to SRAM
    x = tl.load(x_ptrs, mask=mask, other=0)
    dy = tl.load(dy_ptrs, mask=mask, other=0)
    rstd = tl.load(V + row)

    # compute dx
    xhat = x * rstd

    w = tl.load(W + cols, mask=mask, other=0)
    wdy = w * dy


    xhat = tl.where(mask, xhat, 0.)
    wdy = tl.where(mask, wdy, 0.)
    mean1 = tl.sum(xhat * wdy, axis=0) / N
    dx = (wdy - xhat * mean1) * rstd

    # write-back dx
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N  # re-materialize the mask to save registers
    dx_ptrs = DX + row * stride + cols
    tl.store(dx_ptrs, dx, mask=mask)

    # accumulate partial sums for dw
    partial_dw = (dy * xhat).to(w.dtype)

    # offset locks and weight/bias gradient pointer
    # each kernel instance accumulates partial sums for
    # DW into one of GROUP_SIZE_M independent buffers
    # these buffers stay in the L2, which allow this kernel
    # to be fast
    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M

    # - wait for a lock on the accumulated dw/db
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)

    # - we got the lock, accumulate this kernel's results with
    # the stored values.
    dw_ptrs = DW + lock_id * N + cols

    if count == 0:
        # first store doesn't accumulate
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(dw_ptrs, mask=mask, other=0.)

    tl.store(dw_ptrs, partial_dw, mask=mask)

    # release lock
    tl.atomic_xchg(Lock, 0)


# Backward pass (total DW)
# fmt: off
@triton.jit
def rms_norm_bwd_dw(
    DW, FINAL_DW,
    M, N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    # fmt: on
    pid = tl.program_id(0)

    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_cols = cols < N

    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        offs = rows[:, None] * N + cols[None, :]
        mask_rm = rows < M

        dw += tl.load(DW + offs, mask=mask_rm[:, None] & mask_cols[None, :], other=0.0)

    sum_dw = tl.sum(dw, axis=0)

    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_cols = cols < N

    tl.store(FINAL_DW + cols, sum_dw, mask=mask_cols)

