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
def layer_norm_fw(X, Y, W, B, M, V, stride, N, eps, affine: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # fmt: on
    """
    Fused layernorm kernel over a 3d tensor.
    The layer norm is applied over the last dimension.

    Compute
        y = (x - E(x))/(sqrt(var(x) + epsilon)) * gamma + beta
    """

    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N

    # Move to this row
    x_ptrs = X + row * stride + cols
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Compute mean and variance
    mean = tl.sum(x, axis=0) / N
    x_zm = tl.where(mask, x - mean, 0.0)
    tl.store(M + row, mean)

    x_var = tl.sum(x_zm * x_zm, axis=0) / N
    rstd = 1.0 / tl.sqrt(x_var + eps)

    # Normalize, optionally affine
    y = x_zm * rstd
    tl.store(V + row, rstd)

    mask = cols < N
    if affine:
        w = tl.load(W + cols, mask=mask, other=1.0)
        b = tl.load(B + cols, mask=mask, other=0.0)
        y = y * w + b

    y_ptrs = Y + row * stride + cols
    tl.store(y_ptrs, y, mask=mask)


# Backward pass (DX + partial DW + partial DB)
# fmt: off
@triton.jit
def layer_norm_bwd_dx_fused(
    DX, DY, DW, DB,
    X, W, M, V,
    Lock, stride, N,
    # META-parameters
    affine: tl.constexpr,
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
    mean = tl.load(M + row)
    rstd = tl.load(V + row)

    # compute dx
    xhat = (x - mean) * rstd

    if affine:
        w = tl.load(W + cols, mask=mask, other=0)
        wdy = w * dy
    else:
        wdy = dy

    xhat = tl.where(mask, xhat, 0.)
    wdy = tl.where(mask, wdy, 0.)
    mean1 = tl.sum(xhat * wdy, axis=0) / N
    mean2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * mean1 + mean2)) * rstd

    # write-back dx
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N  # re-materialize the mask to save registers
    dx_ptrs = DX + row * stride + cols
    tl.store(dx_ptrs, dx, mask=mask)

    if affine:
        # accumulate partial sums for dw/db
        partial_dw = (dy * xhat).to(w.dtype)
        partial_db = dy.to(w.dtype)

        # offset locks and weight/bias gradient pointer
        # each kernel instance accumulates partial sums for
        # DW and DB into one of GROUP_SIZE_M independent buffers
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
        db_ptrs = DB + lock_id * N + cols

        if count == 0:
            # first store doesn't accumulate
            tl.atomic_xchg(Count, 1)
        else:
            partial_dw += tl.load(dw_ptrs, mask=mask, other=0.)
            partial_db += tl.load(db_ptrs, mask=mask, other=0.)

        tl.store(dw_ptrs, partial_dw, mask=mask)
        tl.store(db_ptrs, partial_db, mask=mask)

        # release lock
        tl.atomic_xchg(Lock, 0)


# Backward pass (total DW + total DB)
# fmt: off
@triton.jit
def layer_norm_bwd_dwdb(
    DW, DB, FINAL_DW, FINAL_DB,
    M, N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    # fmt: on

    pid = tl.program_id(0)

    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_cols = cols < N

    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        offs = rows[:, None] * N + cols[None, :]
        mask_rm = rows < M

        dw += tl.load(DW + offs, mask=mask_rm[:, None] & mask_cols[None, :], other=0.0)
        db += tl.load(DB + offs, mask=mask_rm[:, None] & mask_cols[None, :], other=0.0)

    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)

    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_cols = cols < N

    tl.store(FINAL_DW + cols, sum_dw, mask=mask_cols)
    tl.store(FINAL_DB + cols, sum_db, mask=mask_cols)
