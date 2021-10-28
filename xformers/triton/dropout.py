# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


# CREDITS: This comes almost as-is from the Triton dropout tutorial
# https://raw.githubusercontent.com/openai/triton/master/python/tutorials/04-low-memory-dropout.py

import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_bwd, custom_fwd


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
    Y, X, S,
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
    seed = S + row
    random = tl.rand(seed.to(tl.int32), offsets)
    x_keep = random > p

    # write-back
    zero = 0.
    zero = zero.to(x.dtype)
    output = tl.where(x_keep, (x / (1 - p)).to(x.dtype), zero)
    y_ptrs = Y + offsets
    tl.store(y_ptrs, output, mask=mask)


# Helper to handle the SPMD launch grid and error cases
class _dropout(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, x, p):
        # Soft-flatten an hypothetical 3rd dimension
        x_ = x.reshape(-1, x.shape[-1])
        y = torch.empty_like(x_)
        _, N = x_.shape

        assert y.stride(-1) == 1 and x_.stride(-1) == 1

        # Generate one seed per sample
        # seed max is int32 max for positive numbers: 2**16
        seeds = torch.randint(65536, (x_.shape[0],), device=x.device).to(torch.int32)

        # SPMD launch grid
        def grid(meta):
            return (
                x_.shape[0],
                triton.cdiv(x_.shape[1], meta["BLOCK_SIZE"]),
            )

        k_dropout[grid](y, x_, seeds, y.stride(0), N, p)

        ctx.save_for_backward(seeds)
        ctx.p = p

        return y.reshape_as(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        (seeds,) = ctx.saved_tensors

        # Soft-flatten an hypothetical 3rd dimension
        grad_out_ = grad_out.reshape(-1, grad_out.shape[-1])
        grad_in = torch.empty_like(grad_out_)
        _, N = grad_out_.shape

        assert grad_in.stride(-1) == 1 and grad_out_.stride(-1) == 1

        # SPMD launch grid
        def grid(meta):
            return (
                grad_out_.shape[0],
                triton.cdiv(grad_out_.shape[1], meta["BLOCK_SIZE"]),
            )

        k_dropout[grid](grad_in, grad_out_, seeds, grad_out_.stride(0), N, ctx.p)

        return grad_in.reshape_as(grad_out), None


def dropout(x: torch.Tensor, p: float):
    return _dropout.apply(x, p)
