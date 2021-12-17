# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


# CREDITS: This was initially inspired by the Triton dropout tutorial
# https://raw.githubusercontent.com/openai/triton/master/python/tutorials/04-low-memory-dropout.py

import triton
import triton.language as tl


# fmt: off
@triton.heuristics({"SIZE_RAND_BLOCK": lambda *_, **meta: meta["BLOCK_N"] * meta["BLOCK_M"]})
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 32}, num_warps=1),
        triton.Config({"BLOCK_N": 64}, num_warps=2),
        triton.Config({"BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_N": 256}, num_warps=8),
    ],
    key=["M", "N"],
)
@triton.jit
def k_dropout_fw(
    Y, X, BIAS, SEEDS,
    stride,
    M, N,
    p,
    **meta,
):
    """
    Apply dropout on an input tensor
    Y : Output  (M, N)
    X : Input   (M, N)
    BIAS        (N,)
    SEEDS       (M,)
    p : dropout probability

    This kernel goes through the tensor columns (N dimension), per block (to keep memory parallelism).
    This allows the backward pass to follow the same path, with the same seeds,
    and start reducing on the gradient bias.
    """
    # fmt: on

    BLOCK_M = meta["BLOCK_M"]
    BLOCK_N = meta["BLOCK_N"]
    SIZE_RAND_BLOCK = meta["SIZE_RAND_BLOCK"]

    row_id = tl.program_id(axis=0)
    rows = row_id * BLOCK_M * 4 + tl.arange(0, BLOCK_M)

    col_id = tl.program_id(axis=1)
    cols = col_id * BLOCK_N + tl.arange(0, BLOCK_N)
    seed = SEEDS + col_id  # FIXME index the seed properly

    # pointers starting point
    x_ptrs = X + rows[:, None] * stride + cols[None, :]
    y_ptrs = Y + rows[:, None] * stride + cols[None, :]

    # go over all the tiles, one by one
    rand_offsets = tl.arange(0, SIZE_RAND_BLOCK) + row_id * BLOCK_M * 4
    rand1, rand2, rand3, rand4 = tl.randint4x(seed.to(tl.int32), rand_offsets)
    threshold = ((p - 0.5) * 2147483648.).to(tl.int32)

    # binarize masks, save registers
    rand_mask1 = rand1 > threshold
    rand_mask2 = rand2 > threshold
    rand_mask3 = rand3 > threshold
    rand_mask4 = rand4 > threshold

    col_mask = cols[None, :] < N
    p_scale = 1 / (1 - p) if p < 1. else 1.
    zero = 0.0

    if meta["USE_BIAS"]:
        b_ptrs = BIAS + cols[None, :]
        bias = tl.load(b_ptrs, mask=cols[None, :] < N, other=0.)

    for i in range(4):
        # cycle through the binary masks (workaround / no indexing)
        if i == 0:
            rand_mask = rand_mask1
        elif i == 1:
            rand_mask = rand_mask2
        elif i == 2:
            rand_mask = rand_mask3
        else:
            rand_mask = rand_mask4

        block_mask = (rows[:, None] < M) & col_mask
        x = tl.load(x_ptrs, mask=block_mask, other=0.)

        # optionally apply a fused bias
        if meta["USE_BIAS"]:
            x += bias

        # optional: fused activation (while the data is in shared memory)
        if meta["ACTIVATION"]:
            x = meta["ACTIVATION"](x)

        # randomly prune and scale
        if p > 0.:
            # generate all the random numbers for the block at once, then reshape
            keep = tl.reshape(rand_mask, x.shape)

            # prune and normalize in one go
            output = tl.where(keep, (x * p_scale).to(x.dtype), zero.to(x.dtype))
        else:
            output = x

        tl.store(y_ptrs, output, mask=block_mask)

        # Update the pointers
        rows += BLOCK_M  # needs to be updated for the mask to be correct
        x_ptrs += BLOCK_M * stride
        y_ptrs += BLOCK_M * stride


# fmt: off
@triton.heuristics({"SIZE_RAND_BLOCK": lambda *_, **meta: meta["BLOCK_N"] * meta["BLOCK_M"]})
@triton.jit
def k_dropout_bw(
    GRAD_IN, GRAD_BIAS, GRAD_OUT,
    INPUTS, BIAS, SEEDS,
    stride_grad, stride_inputs,
    M, N,
    p,
    **meta,
):
    """
    Apply dropout on an input tensor
    GRAD_OUT    (M, N)
    GRAD_BIAS   (N,)
    GRAD_IN     (M, N)
    BIAS        (N,)
    SEEDS       (N,)
    p : dropout probability
    """
    # fmt: on

    BLOCK_M = meta["BLOCK_M"]
    BLOCK_N = meta["BLOCK_N"]
    SIZE_RAND_BLOCK = meta["SIZE_RAND_BLOCK"]
    TRAINABLE_BIAS = meta["TRAINABLE_BIAS"]

    rows = tl.arange(0, BLOCK_M)
    row_id = tl.program_id(axis=0)
    rows = row_id * BLOCK_M * 4 + tl.arange(0, BLOCK_M)

    col_id = tl.program_id(axis=1)
    cols = col_id * BLOCK_N + tl.arange(0, BLOCK_N)
    seed = SEEDS + col_id  # FIXME index the seed properly

    # pointers starting point
    grad_out_ptrs = GRAD_OUT + rows[:, None] * stride_grad + cols[None, :]
    grad_in_ptrs = GRAD_IN + rows[:, None] * stride_grad + cols[None, :]
    input_ptrs = INPUTS + rows[:, None] * stride_inputs + cols[None, :]

    # random binary masks, save registers
    rand_offsets = tl.arange(0, SIZE_RAND_BLOCK) + row_id * BLOCK_M * 4
    rand1, rand2, rand3, rand4 = tl.randint4x(seed.to(tl.int32), rand_offsets)
    threshold = ((p - 0.5) * 2147483648.).to(tl.int32)

    rand_mask1 = rand1 > threshold
    rand_mask2 = rand2 > threshold
    rand_mask3 = rand3 > threshold
    rand_mask4 = rand4 > threshold

    # now go over the tiles
    grad_bias = tl.zeros((BLOCK_N,), dtype=tl.float32)
    col_mask = cols[None, :] < N
    zero = 0.0
    p_scale = 1 / (1 - p) if p < 1. else 1.

    if meta["USE_BIAS"]:
        b_ptrs = BIAS + cols[None, :]
        bias = tl.load(b_ptrs, mask=col_mask, other=0.)

    for i in range(4):
        # cycle through the binary masks (workaround / no indexing)
        if i == 0:
            rand_mask = rand_mask1
        elif i == 1:
            rand_mask = rand_mask2
        elif i == 2:
            rand_mask = rand_mask3
        else:
            rand_mask = rand_mask4

        block_mask = (rows[:, None] < M) & col_mask
        grad_out = tl.load(grad_out_ptrs, mask=block_mask, other=0.)

        # optional: fused activation (while the data is in shared memory)
        if meta["ACTIVATION_GRAD"]:
            inputs = tl.load(input_ptrs, mask=block_mask, other=0.)

            # optionally apply a fused bias
            if meta["USE_BIAS"]:
                inputs += bias

            act_grad = meta["ACTIVATION_GRAD"](inputs).to(grad_out.dtype)
            grad_out *= act_grad

        # randomly prune and scale
        if p > 0.:
            # generate all the random numbers for the block at once, then reshape
            keep = tl.reshape(rand_mask, grad_out.shape)

            # prune and normalize in one go
            output = tl.where(
                keep,
                (grad_out * p_scale).to(grad_out.dtype),
                zero.to(grad_out.dtype)
            )
        else:
            output = grad_out

        # write-back
        tl.store(grad_in_ptrs, output, mask=block_mask)

        # optionally accumulate the bias gradient
        if TRAINABLE_BIAS:
            grad_bias += tl.sum(output, axis=0)

        # Update the pointers
        rows += BLOCK_M  # needs to be updated for the mask to be correct
        grad_out_ptrs += BLOCK_M * stride_grad
        input_ptrs += BLOCK_M * stride_inputs
        grad_in_ptrs += BLOCK_M * stride_grad

    if TRAINABLE_BIAS:
        grad_bias_ptr = GRAD_BIAS + row_id * N + cols
        tl.store(grad_bias_ptr, grad_bias, mask=cols < N)
