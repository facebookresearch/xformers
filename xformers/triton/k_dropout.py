# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


# CREDITS: This is heavily inspired by the Triton dropout tutorial
# https://raw.githubusercontent.com/openai/triton/master/python/tutorials/04-low-memory-dropout.py

import triton
import triton.language as tl

from xformers.triton.k_activations import (
    gelu,
    gelu_grad,
    leaky_relu,
    leaky_relu_grad,
    relu,
    relu_grad,
    smelu,
    smelu_grad,
    squared_relu,
    squared_relu_grad,
)

_configs = [
    triton.Config({}, num_warps=1),
    triton.Config({}, num_warps=2),
    triton.Config({}, num_warps=4),
    triton.Config({}, num_warps=8),
    triton.Config({}, num_warps=16),
]


# fmt: off
@triton.heuristics({"SIZE_RAND_BLOCK": lambda args: args["BLOCK_N"] * args["BLOCK_M"]})
@triton.autotune(
    configs=_configs,
    key=["M", "N", "is_fp16"],
)
@triton.jit
def k_dropout_fw(
    Y, X, BIAS, SEEDS,
    stride,
    M, N,
    p: tl.constexpr,
    is_fp16: tl.constexpr,  # autotune
    ACTIVATION: tl.constexpr,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SIZE_RAND_BLOCK: tl.constexpr,
    USE_BIAS: tl.constexpr,
):
    """
    Apply dropout on an input tensor
    Y : Output  (M, N)
    X : Input   (M, N)
    BIAS        (N,)
    SEEDS       (M,)
    p : dropout probability
    """
    # fmt: on

    row_id = tl.program_id(axis=0)
    rows = row_id * BLOCK_M + tl.arange(0, BLOCK_M)

    col_id = tl.program_id(axis=1)
    cols = col_id * BLOCK_N + tl.arange(0, BLOCK_N)

    # pointers starting point
    x_ptrs = X + rows[:, None] * stride + cols[None, :]
    y_ptrs = Y + rows[:, None] * stride + cols[None, :]

    # good to go, start the layer computations
    col_mask = cols[None, :] < N
    p_scale = 1. / (1. - p)
    if USE_BIAS:
        b_ptrs = BIAS + cols[None, :]
        bias = tl.load(b_ptrs, mask=cols[None, :] < N, other=0.)
    else:
        bias = x_ptrs  # will not be used

    block_mask = (rows[:, None] < M) & col_mask
    x = tl.load(x_ptrs, mask=block_mask, other=0.0)

    # optionally apply a fused bias
    if USE_BIAS:
        x += bias

    # optional: fused activation (while the data is in shared memory)
    if ACTIVATION == 1:
        x = relu(x)
    elif ACTIVATION == 2:
        x = leaky_relu(x)
    elif ACTIVATION == 3:
        x = gelu(x)
    elif ACTIVATION == 4:
        x = squared_relu(x)
    elif ACTIVATION == 5:
        x = smelu(x)

    # get the random keep mask
    rand_offsets = tl.arange(0, SIZE_RAND_BLOCK)
    seed_int = tl.load(SEEDS + col_id)
    r = tl.rand(seed_int, rand_offsets)
    keep_mask = r > p

    # prune and normalize in one go
    keep = tl.reshape(keep_mask, x.shape)
    output = tl.where(keep, (x * p_scale).to(x.dtype), 0.)

    tl.store(y_ptrs, output, mask=block_mask)  # output


# fmt: off
@triton.heuristics({"SIZE_RAND_BLOCK": lambda args: args["BLOCK_N"] * args["BLOCK_M"]})
@triton.autotune(
    configs=_configs,
    key=["M", "N", "is_fp16"],
)
@triton.jit
def k_dropout_bw(
    GRAD_IN, GRAD_BIAS, GRAD_OUT,
    INPUTS, BIAS, SEEDS,
    stride_grad, stride_inputs,
    M, N,
    p: tl.constexpr,
    is_fp16: tl.constexpr,  # autotune
    ACTIVATION: tl.constexpr,
    # Meta-parameters
    BLOCK_M: tl.constexpr,  # heuristics
    BLOCK_N: tl.constexpr,
    SIZE_RAND_BLOCK: tl.constexpr,
    TRAINABLE_BIAS: tl.constexpr,
    USE_BIAS: tl.constexpr,
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

    row_id = tl.program_id(axis=0)
    rows = row_id * BLOCK_M + tl.arange(0, BLOCK_M)

    col_id = tl.program_id(axis=1)
    cols = col_id * BLOCK_N + tl.arange(0, BLOCK_N)

    # pointers starting point
    grad_out_ptrs = GRAD_OUT + rows[:, None] * stride_grad + cols[None, :]
    grad_in_ptrs = GRAD_IN + rows[:, None] * stride_grad + cols[None, :]
    input_ptrs = INPUTS + rows[:, None] * stride_inputs + cols[None, :]

    # now go over the tiles
    grad_bias = tl.zeros((BLOCK_N,), dtype=tl.float32)
    col_mask = cols[None, :] < N
    p_scale = 1. / (1. - p)

    if USE_BIAS:
        b_ptrs = BIAS + cols[None, :]
        bias = tl.load(b_ptrs, mask=col_mask, other=0.)

    block_mask = (rows[:, None] < M) & col_mask
    grad_out = tl.load(grad_out_ptrs, mask=block_mask, other=0.)

    # optional: fused activation (while the data is in shared memory)
    if ACTIVATION:
        inputs = tl.load(input_ptrs, mask=block_mask, other=0.)

        # optionally apply a fused bias
        if USE_BIAS:
            inputs += bias

        if ACTIVATION == 1:
            act_grad = relu_grad(inputs)
        elif ACTIVATION == 2:
            act_grad = leaky_relu_grad(inputs)
        elif ACTIVATION == 3:
            act_grad = gelu_grad(inputs)
        elif ACTIVATION == 4:
            act_grad = squared_relu_grad(inputs)
        elif ACTIVATION == 5:
            act_grad = smelu_grad(inputs)

        grad_out *= act_grad

    # randomly prune (and scale) the resulting buffer, possibly a no-op
    # note that even if we did not save the mask from the FW pass, it is generated
    # from the same seeds, so the same drop mask is applied here
    rand_offsets = tl.arange(0, SIZE_RAND_BLOCK)
    seed_int = tl.load(SEEDS + col_id)
    r = tl.rand(seed_int, rand_offsets)
    r = tl.reshape(r, grad_out.shape)
    output = tl.where(r > p, (grad_out * p_scale).to(grad_out.dtype), 0.)

    # write-back
    tl.store(grad_in_ptrs, output, mask=block_mask)

    # optionally accumulate the bias gradient
    if TRAINABLE_BIAS:
        grad_bias += tl.sum(output, axis=0)

    if TRAINABLE_BIAS:
        grad_bias_ptr = GRAD_BIAS + row_id * N + cols
        tl.store(grad_bias_ptr, grad_bias, mask=cols < N)
