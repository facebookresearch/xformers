# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


# CREDITS: This comes almost as-is from the Triton dropout tutorial
# https://raw.githubusercontent.com/openai/triton/master/python/tutorials/04-low-memory-dropout.py

import triton
import triton.language as tl

_k_configs = [
    triton.Config({"BLOCK_SIZE": 128}, num_warps=1),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=2),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    triton.Config({"BLOCK_SIZE": 4096}, num_warps=16),
]


@triton.jit
def _drop_and_scale(SEEDS, row, p, offsets, x):
    # randomly prune the weights
    seed = SEEDS + row
    random = tl.rand(seed.to(tl.int32), offsets)
    x_keep = random > p

    zero = 0.0
    zero = zero.to(x.dtype)

    # prune and normalize in one go
    return tl.where(x_keep, (x / (1 - p)).to(x.dtype), zero)


# fmt: off
@triton.autotune(
    configs=_k_configs,
    key=["N"],
)
@triton.jit
def k_dropout_fw(
    Y, X, BIAS, SEEDS,
    stride,
    N,
    p,
    **META,
):
    """
    Apply dropout on an input tensor
    Y : Output (M, N)
    X : Input (M, N)
    S : Seeds (M,)
    p : dropout probability
    """
    # fmt: on

    BLOCK_SIZE = META["BLOCK_SIZE"]
    row = tl.program_id(axis=0)
    col = tl.program_id(axis=1)

    # compute memory offsets of elements handled by this instance
    offsets = row * stride + col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) < N

    # load data from x
    x_ptrs = X + offsets
    x = tl.load(x_ptrs, mask=mask)

    # optionally apply a fused bias
    if META["USE_BIAS"]:
        b_ptrs = BIAS + col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        b = tl.load(b_ptrs, mask=mask)
        x += b

    # optional: fused activation (while the data is in shared memory)
    if META["ACTIVATION"]:
        x = META["ACTIVATION"](x)

    # randomly prune it
    if p > 0.:
        output = _drop_and_scale(SEEDS, row, p, offsets, x)
    else:
        output = x

    y_ptrs = Y + offsets
    tl.store(y_ptrs, output, mask=mask)


# fmt: off
@triton.autotune(
    configs=_k_configs,
    key=["N"],
)
@triton.jit
def k_dropout_bw(
    GRAD_IN, GRAD_OUT, INPUTS, BIAS, SEEDS,
    stride_grad, stride_inputs,
    N,
    p,
    **META,
):
    """
    Apply dropout on an input tensor
    GRAD_OUT    (M, N)
    GRAD_IN     (M, N)
    BIAS        (N,)
    SEEDS       (M,)
    p : dropout probability
    """
    # fmt: on

    BLOCK_SIZE = META["BLOCK_SIZE"]
    row = tl.program_id(axis=0)
    col = tl.program_id(axis=1)

    # compute memory offsets of elements handled by this instance
    grad_offsets = row * stride_grad + col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) < N

    # load data from x
    grad_out_ptrs = GRAD_OUT + grad_offsets
    grad_out = tl.load(grad_out_ptrs, mask=mask)

    # optional: fused activation (while the data is in shared memory)
    if META["ACTIVATION_GRAD"]:
        input_ptrs = INPUTS + row * stride_inputs + col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        inputs = tl.load(input_ptrs, mask=mask)

        # optionally apply a fused bias
        if META["USE_BIAS"]:
            b_ptrs = BIAS + col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            b = tl.load(b_ptrs, mask=mask)
            inputs += b

        act_grad = META["ACTIVATION_GRAD"](inputs)
        grad_out *= act_grad

    # randomly prune it
    if p > 0.:
        output = _drop_and_scale(SEEDS, row, p, grad_offsets, grad_out)
    else:
        output = grad_out

    # write-back
    y_ptrs = GRAD_IN + grad_offsets
    tl.store(y_ptrs, output, mask=mask)
