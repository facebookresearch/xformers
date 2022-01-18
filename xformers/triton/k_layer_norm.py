# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


# CREDITS: This comes almost as-is from the Triton layer norm tutorial
# https://github.com/openai/triton/blob/master/python/tutorials/05-layer-norm.py

import logging

import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_bwd, custom_fwd

_triton_layernorm_fp16_enabled = False  # NOTE: PyTorch keeps layernorm as fp32
_triton_registered_warnings = False


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

    row = tl.program_id(0)
    cols = tl.arange(0, META["BLOCK_SIZE_N"])

    # Move to this row
    x_ptrs = X + row * stride + cols
    x = tl.load(x_ptrs, mask=cols < N, other=0.0).to(tl.float32)
    x = tl.where(cols < N, x, 0.)  # Triton bug workarounds

    # Compute variance
    x_mean = tl.sum(x, axis=0) / N
    x_zm = x - x_mean
    x_zm = tl.where(cols < N, x_zm, 0.0)  # Triton bug workaround
    x_var = tl.sum(x_zm * x_zm, axis=0) / N
    x_inv_sigma = 1.0 / tl.sqrt(x_var + eps)

    # write-back per sample mean/rstd, used in the backward pass
    tl.store(M + row, x_mean)
    tl.store(V + row, x_inv_sigma)

    # Normalize the inputs
    w = tl.load(W + cols, mask=cols < N, other=1.0)
    zero = 0.
    zero = zero.to(w.dtype)  # Triton bug workarounds
    w = tl.where(cols < N, w, zero)

    b = tl.load(B + cols, mask=cols < N, other=0.0)
    b = tl.where(cols < N, b, zero)
    y = x_zm * x_inv_sigma * w + b

    # write back to Y.
    y_ptrs = Y + row * stride + cols
    tl.store(y_ptrs, y, mask=cols < N)


# Backward pass (DX + partial DW + partial DB)
# fmt: off
@triton.jit
def _layer_norm_bwd_dx_fused(
        DX, DY, DW, DB,
        X, W, M, V,
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
    x_ptrs = X + row * stride + cols
    dy_ptrs = DY + row * stride + cols
    w_ptrs = W + cols

    # offset locks and weight/bias gradient pointer
    # each kernel instance accumulates partial sums for
    # DW and DB into one of GROUP_SIZE_M independent buffers
    # these buffers stay in the L2, which allow this kernel
    # to be fast
    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M

    # load data to SRAM
    x = tl.load(x_ptrs, mask=cols < N, other=0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=cols < N, other=0).to(tl.float32)
    w = tl.load(w_ptrs, mask=cols < N, other=0).to(tl.float32)

    mean = tl.load(M + row)
    rstd = tl.load(V + row)

    # compute dx
    xhat = (x - mean) * rstd
    wdy = w * dy
    xhat = tl.where(cols < N, xhat, 0.0)
    wdy = tl.where(cols < N, wdy, 0.0)
    mean1 = tl.sum(xhat * wdy, axis=0) / N
    mean2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * mean1 + mean2)) * rstd

    # write-back dx
    cols = tl.arange(0, BLOCK_SIZE_N)
    dx_ptrs = DX + row * stride + cols
    tl.store(dx_ptrs, dx, mask=cols < N)

    # accumulate partial sums for dw/db
    partial_dw = (dy * xhat).to(w.dtype)
    partial_db = (dy).to(w.dtype)

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


# Backward pass (total DW + total DB)
# fmt: off
@triton.jit
def _layer_norm_bwd_dwdb(DW, DB, FINAL_DW, FINAL_DB, M, N, **meta):
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


# FIXME: @lefaudeux tensor shape changes are not well handled, see shape3
class _LayerNorm(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16 if _triton_layernorm_fp16_enabled else None)
    def forward(ctx, x, weight, bias, eps):

        # allocate output
        y = torch.empty_like(x)

        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape

        # allocate mean and std, they'll be used in the backward pass
        mean = torch.empty((M,), dtype=torch.float32, device="cuda")
        rstd = torch.empty((M,), dtype=torch.float32, device="cuda")

        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE_N:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

        if not x_arg.is_contiguous() or not y.is_contiguous():
            global _triton_registered_warnings
            if not _triton_registered_warnings:
                logging.warning("Non-contiguous input tensor found. Making it contiguous,"
                                + " but could have perf or trainer implications")

                _triton_registered_warnings = True

            x_arg = x_arg.contiguous()
            y = y.contiguous()

        # heuristics for number of warps.
        num_warps = min(max(BLOCK_SIZE_N // 256, 1), 8)

        # enqueue kernel
        # fmt: off
        layer_norm_fw[(M,)](
            x_arg, y, weight, bias, mean, rstd,
            x_arg.stride(0),
            N,
            eps,
            num_warps=num_warps,
            BLOCK_SIZE_N=BLOCK_SIZE_N
        )
        # fmt: on

        ctx.save_for_backward(x, weight, mean, rstd)
        ctx.BLOCK_SIZE_N = BLOCK_SIZE_N
        ctx.num_warps = num_warps
        ctx.eps = eps

        return y.reshape_as(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, dy):
        x, weight, mean, var = ctx.saved_tensors

        # heuristics for amount of parallel reduction stream for DG/DB
        N = weight.shape[0]
        GROUP_SIZE_M = 64
        if N <= 8192:
            GROUP_SIZE_M = 96
        if N <= 4096:
            GROUP_SIZE_M = 128
        if N <= 1024:
            GROUP_SIZE_M = 256

        # flatten the batch dimension, if any.
        # We're interested in 'samples' x norm_dimension
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape

        # allocate output
        locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device="cuda")
        t_args = {"dtype" : x.dtype, "device" : x.device}
        _dw = torch.empty((GROUP_SIZE_M, weight.shape[0]), **t_args)
        _db = torch.empty((GROUP_SIZE_M, weight.shape[0]), **t_args)
        dw = torch.empty((weight.shape[0],), **t_args)
        db = torch.empty((weight.shape[0],), **t_args)
        dx = torch.empty_like(dy)

        # Check the tensor shapes and layouts
        # we suppose in the kernel that they have the same size and are contiguous
        assert dx.numel() == x.numel(), \
            "Something is wrong in the backward graph, possibly because of an inplace operation after the layernorm"

        assert dx.is_contiguous() and dy.is_contiguous()

        # enqueue kernel using forward pass heuristics
        # also compute partial sums for DW and DB

        # fmt: off
        _layer_norm_bwd_dx_fused[(M,)](
            dx, dy, _dw, _db,
            x_arg, weight, mean, var,
            locks,
            x_arg.stride(0),
            N,
            BLOCK_SIZE_N=ctx.BLOCK_SIZE_N,
            GROUP_SIZE_M=GROUP_SIZE_M,
            num_warps=ctx.num_warps
        )
        # fmt: on

        def grid(meta):
            return [triton.cdiv(N, meta["BLOCK_SIZE_N"])]

        # accumulate partial sums in separate kernel
        # fmt: off
        _layer_norm_bwd_dwdb[grid](
            _dw, _db, dw, db,
            GROUP_SIZE_M,
            N,
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_N=128
        )
        # fmt: on

        dx = dx.reshape_as(x)
        return dx, dw, db, None
