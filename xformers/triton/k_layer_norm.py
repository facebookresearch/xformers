# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


# CREDITS: This comes almost as-is from the Triton layer norm tutorial
# https://github.com/openai/triton/blob/master/python/tutorials/05-layer-norm.py


import triton
import triton.language as tl


@triton.jit
def _affine(W, B, N, x, META):
    cols = tl.arange(0, META["BLOCK_SIZE_N"])

    w = tl.load(W + cols, mask=cols < N, other=1.0)
    zero = 0.0
    zero = zero.to(w.dtype)  # Triton bug workarounds
    w = tl.where(cols < N, w, zero)

    b = tl.load(B + cols, mask=cols < N, other=0.0)
    b = tl.where(cols < N, b, zero)
    y = x * w + b
    return y


@triton.jit
def _store(y, Y, stride, N, META):
    row = tl.program_id(0)
    cols = tl.arange(0, META["BLOCK_SIZE_N"])

    y_ptrs = Y + row * stride + cols
    tl.store(y_ptrs, y, mask=cols < N)


@triton.jit
def layer_norm_non_affine(X, M, V, stride, N, eps, META):
    # fmt: on
    """
    Fused layernorm kernel over a 3d tensor.
    The layer norm is applied over the last dimension.

    Compute
        y = (x - E(x))/(sqrt(var(x) + epsilon)) * gamma + beta
    """

    row = tl.program_id(0)
    cols = tl.arange(0, META["BLOCK_SIZE_N"])

    # Move to this row
    x_ptrs = X + row * stride + cols
    x = tl.load(x_ptrs, mask=cols < N, other=0.0).to(tl.float32)
    x = tl.where(cols < N, x, 0.0)  # Triton bug workarounds

    # Compute variance
    x_mean = tl.sum(x, axis=0) / N
    x_zm = x - x_mean
    x_zm = tl.where(cols < N, x_zm, 0.0)  # Triton bug workaround
    x_var = tl.sum(x_zm * x_zm, axis=0) / N
    x_inv_sigma = 1.0 / tl.sqrt(x_var + eps)

    # write-back per sample mean/rstd, used in the backward pass
    tl.store(M + row, x_mean)
    tl.store(V + row, x_inv_sigma)

    return x_zm * x_inv_sigma


# fmt: off
@triton.jit
def layer_norm_non_affine_fw(X, Y, M, V, stride, N, eps, **META):
    _store(layer_norm_non_affine(X, M, V, stride, N, eps, META), Y, stride, N, META)


# fmt: off
@triton.jit
def layer_norm_fw(X, Y, W, B, M, V, stride, N, eps, **META):
    # fmt: on
    """
    Fused layernorm kernel over a 3d tensor.
    The layer norm is applied over the last dimension.

    Compute
        y = (x - E(x))/(sqrt(var(x) + epsilon)) * gamma + beta
    """
    y = layer_norm_non_affine(X, M, V, stride, N, eps, META)
    y = _affine(W, B, N, y, META)

    _store(y, Y, stride, N, META)


# Backward pass (DX + partial DW + partial DB)
# fmt: off
@triton.jit
def layer_norm_bwd_dx_fused(
    DX, DY, DW, DB,
    Y, W, B, V,
    Lock, stride, N,
    **META
):
    # fmt: on

    GROUP_SIZE_M = META["GROUP_SIZE_M"]
    BLOCK_SIZE_N = META["BLOCK_SIZE_N"]

    # position of elements processed by this program
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)

    # offset data pointers to start at the row of interest
    y_ptrs = Y + row * stride + cols
    dy_ptrs = DY + row * stride + cols
    w_ptrs = W + cols
    b_ptrs = B + cols

    # offset locks and weight/bias gradient pointer
    # each kernel instance accumulates partial sums for
    # DW and DB into one of GROUP_SIZE_M independent buffers
    # these buffers stay in the L2, which allow this kernel
    # to be fast
    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M

    # load data to SRAM
    y = tl.load(y_ptrs, mask=cols < N, other=0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=cols < N, other=0).to(tl.float32)
    w = tl.load(w_ptrs, mask=cols < N, other=0).to(tl.float32)
    b = tl.load(b_ptrs, mask=cols < N, other=0).to(tl.float32)

    rstd = tl.load(V + row)

    # compute dx
    xhat = (y - b) / w
    wdy = w * dy
    xhat = tl.where(cols < N, xhat, 0.0)
    wdy = tl.where(cols < N, wdy, 0.0)
    mean1 = tl.sum(xhat * wdy, axis=0) / N
    mean2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * mean1 + mean2)) * rstd

    # write-back dx
    _store(dx, DX, stride, N, META)

    # accumulate partial sums for dw/db
    partial_dw = (dy * xhat).to(w.dtype)
    partial_db = dy.to(w.dtype)

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
        partial_dw += tl.load(dw_ptrs, mask=cols < N, other=0.)
        partial_db += tl.load(db_ptrs, mask=cols < N, other=0.)

    tl.store(dw_ptrs, partial_dw, mask=cols < N)
    tl.store(db_ptrs, partial_db, mask=cols < N)

    # release lock
    tl.atomic_xchg(Lock, 0)


@triton.jit
def layer_norm_no_affine_bwd(
    DX, DY,
    Y, V,
    stride, N,
    **META
):
    # fmt: on

    # position of elements processed by this program
    row = tl.program_id(0)
    cols = tl.arange(0, META["BLOCK_SIZE_N"])

    # offset data pointers to start at the row of interest
    y_ptrs = Y + row * stride + cols
    dy_ptrs = DY + row * stride + cols

    # load data to SRAM
    y = tl.load(y_ptrs, mask=cols < N, other=0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=cols < N, other=0).to(tl.float32)

    rstd = tl.load(V + row)

    # compute dx
    xhat = tl.where(cols < N, y, 0.0)
    wdy = tl.where(cols < N, dy, 0.0)
    mean1 = tl.sum(xhat * wdy, axis=0) / N
    mean2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * mean1 + mean2)) * rstd

    # write-back dx
    _store(dx, DX, stride, N, META)


# Backward pass (total DW + total DB)
# fmt: off
@triton.jit
def layer_norm_bwd_dwdb(DW, DB, FINAL_DW, FINAL_DB, M, N, **meta):
    # fmt: on
    pid = tl.program_id(0)
    BLOCK_SIZE_M = meta["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = meta["BLOCK_SIZE_N"]

    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, meta["BLOCK_SIZE_M"])
        offs = rows[:, None] * N + cols[None, :]

        dw += tl.load(DW + offs, mask=(rows[:, None] < M) & (cols[None, :] < N), other=0.0)
        db += tl.load(DB + offs, mask=(rows[:, None] < M) & (cols[None, :] < N), other=0.0)

    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)

    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)
