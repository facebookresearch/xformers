# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


# CREDITS: This comes almost as-is from the Triton dropout tutorial
# https://raw.githubusercontent.com/openai/triton/master/python/tutorials/04-low-memory-dropout.py

import triton
import triton.language as tl

_configs = [
    triton.Config({}, num_warps=1),
    triton.Config({}, num_warps=2),
    triton.Config({}, num_warps=4),
    triton.Config({}, num_warps=8),
    triton.Config({}, num_warps=16),
]


@triton.jit
def _get_4_bin_masks(seed_ptr, rand_offsets, p):
    seed = tl.load(seed_ptr)
    rand1, rand2, rand3, rand4 = tl.randint4x(seed, rand_offsets)

    # binarize masks, save registers
    # NOTE: We keep the random numbers as is there (integers over uint32),
    # and convert the threshold instead, for speed

    # The initial distribution is over 2**32 -1
    # and our float threshold  is in between [0, 1]
    # The full computation is: `start_point + full range * p`
    threshold = (4294967296.0 * p).to(tl.int32)
    rand_mask1 = rand1 > threshold
    rand_mask2 = rand2 > threshold
    rand_mask3 = rand3 > threshold
    rand_mask4 = rand4 > threshold

    return rand_mask1, rand_mask2, rand_mask3, rand_mask4


@triton.jit
def _random_prune_and_scale(x, rand_mask, p, p_scale):
    zero = 0.0

    # generate all the random numbers for the block at once, then reshape
    keep = tl.reshape(rand_mask, x.shape)

    # prune and normalize in one go
    x = tl.where(keep, (x * p_scale).to(x.dtype), zero.to(x.dtype))
    return x


@triton.jit
def tile_random_drop(
    x_ptrs,
    y_ptrs,
    block_mask,
    use_bias,
    bias,
    rand_mask,
    p,
    p_scale,
    ACTIVATION,
):
    x = tl.load(x_ptrs, mask=block_mask, other=0.0)

    # optionally apply a fused bias
    if use_bias:
        x += bias

    # optional: fused activation (while the data is in shared memory)
    if ACTIVATION:
        x = ACTIVATION(x)

    # randomly prune (and scale) the resulting buffer, possibly a no-op
    output = _random_prune_and_scale(x, rand_mask, p, p_scale)

    tl.store(y_ptrs, output, mask=block_mask)  # output


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
    p,
    is_fp16,  # autotune
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SIZE_RAND_BLOCK: tl.constexpr,
    USE_BIAS: tl.constexpr,
    ACTIVATION: tl.constexpr,
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
    rows = row_id * BLOCK_M * 4 + tl.arange(0, BLOCK_M)

    col_id = tl.program_id(axis=1)
    cols = col_id * BLOCK_N + tl.arange(0, BLOCK_N)
    seed = SEEDS + col_id

    # pointers starting point
    x_ptrs = X + rows[:, None] * stride + cols[None, :]
    y_ptrs = Y + rows[:, None] * stride + cols[None, :]

    # go over all the tiles, one by one
    rand_offsets = tl.arange(0, SIZE_RAND_BLOCK) + row_id * BLOCK_M * 4
    rand_mask1, rand_mask2, rand_mask3, rand_mask4 = _get_4_bin_masks(seed, rand_offsets, p)

    col_mask = cols[None, :] < N
    p_scale = 1 / (1 - p)

    if USE_BIAS:
        b_ptrs = BIAS + cols[None, :]
        bias = tl.load(b_ptrs, mask=cols[None, :] < N, other=0.)
    else:
        bias = x_ptrs  # will not be used

    # cycle through the binary masks (workaround / no indexing)
    for i in range(4):
        if i == 0:
            rand_mask = rand_mask1
        elif i == 1:
            rand_mask = rand_mask2
        elif i == 2:
            rand_mask = rand_mask3
        else:
            rand_mask = rand_mask4

        block_mask = (rows[:, None] < M) & col_mask
        tile_random_drop(x_ptrs, y_ptrs, block_mask, USE_BIAS, bias, rand_mask, p, p_scale, ACTIVATION)

        rows += BLOCK_M  # needs to be updated for the mask to be correct
        x_ptrs += BLOCK_M * stride
        y_ptrs += BLOCK_M * stride


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
    p,
    is_fp16,  # autotune
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SIZE_RAND_BLOCK: tl.constexpr,
    TRAINABLE_BIAS: tl.constexpr,
    USE_BIAS: tl.constexpr,
    ACTIVATION_GRAD: tl.constexpr,
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
    rand_mask1, rand_mask2, rand_mask3, rand_mask4 = _get_4_bin_masks(seed, rand_offsets, p)

    # now go over the tiles
    grad_bias = tl.zeros((BLOCK_N,), dtype=tl.float32)
    col_mask = cols[None, :] < N
    p_scale = 1 / (1 - p)

    if USE_BIAS:
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
        if ACTIVATION_GRAD:
            inputs = tl.load(input_ptrs, mask=block_mask, other=0.)

            # optionally apply a fused bias
            if USE_BIAS:
                inputs += bias

            act_grad = ACTIVATION_GRAD(inputs).to(grad_out.dtype)
            grad_out *= act_grad

        # randomly prune (and scale) the resulting buffer, possibly a no-op
        # note that even if we did not save the mask from the FW pass, it is generated
        # from the same seeds, so the same drop mask is applied here
        output = _random_prune_and_scale(grad_out, rand_mask, p, p_scale)

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
