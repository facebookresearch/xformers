# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


# CREDITS: This comes almost as-is from the Triton dropout tutorial
# https://raw.githubusercontent.com/openai/triton/master/python/tutorials/04-low-memory-dropout.py

import torch
import triton
from torch.cuda.amp import custom_bwd, custom_fwd

from xformers.triton.k_dropout import k_dropout


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
    if p > 0.0:
        return _dropout.apply(x, p)

    return x
