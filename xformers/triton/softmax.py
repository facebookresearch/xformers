# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import logging

import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_bwd, custom_fwd

from xformers.triton.utils import next_power_of_2

# Credits: This is adapted from the vanilla Triton example. See https://openai.com/blog/triton/
# and https://triton-lang.org/getting-started/tutorials/02-fused-softmax.html


_triton_register_overflow = False
_triton_softmax_fp16_enabled = False  # NOTE: PyTorch keeps softmax as fp32

kernel_configs = [
    triton.Config({}, num_warps=1),
    triton.Config({}, num_warps=2),
    triton.Config({}, num_warps=4),
    triton.Config({}, num_warps=8),
    triton.Config({}, num_warps=16),
]


def _get_depth(*args, **kwargs):
    return next_power_of_2(args[-1])


def _get_fp16(*args, **kwargs):
    return args[0].dtype == torch.float16


def _get_num_warps(*args, **kwargs):
    num_warps = 4
    if kwargs["depth"] >= 2048:
        num_warps = 8
    if kwargs["depth"] >= 4096:
        num_warps = 16

    return num_warps


# autotune: Triton will test out these configurations, and automatically pick the fastest one.
# heuristic: add arguments to the kernel call automatically given some heuristics. These arguments are passed in "meta"
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
    ],
    key=["K"],
)
@triton.heuristics(values={"depth": _get_depth, "is_fp16": _get_fp16})
@triton.jit
def _softmax(
    Y,
    stride_ym,
    stride_yn,
    stride_yk,
    X,
    stride_xm,
    stride_xn,
    stride_xk,
    K,
    **meta,  # extra parameters which can be automatically filled in given some heuristics
):
    """
    Fused softmax kernel over a 3d tensor.
    The softmax is applied over the last dimension, meaning that this is equivalent to torch.softmax(tensor, dim=-1)

    Note, if the last dimension is large, say 128K elements, the kernel compile time can shot up to many minutes when
    the kernel is run for the first time.
    """
    m = tl.program_id(0)
    n = tl.program_id(1)

    # col indices
    k = tl.arange(0, meta["depth"])

    # the memory address of all the elements that we want to load can be computed as follows
    X = X + m * stride_xm + n * stride_xn + k * stride_xk

    # load input data; pad out-of-bounds elements with 0
    x = tl.load(X, mask=k < K, other=float("-inf"))

    # compute numerically-stable softmax
    z = x - tl.max(x, axis=0)

    if meta["is_fp16"]:
        # tl.exp() crashes on fp16 values
        # See https://github.com/openai/triton/issues/241
        z = z.to(tl.float32)

    num = tl.exp(z)
    denom = tl.sum(num, axis=0)

    if meta["log"]:
        y = z - tl.log(denom)
    else:
        y = num / denom

    # write back to Y.
    # we only write once, hence the "fused" softmax naming
    Y = Y + m * stride_ym + n * stride_yn + k * stride_yk
    tl.store(Y, y, mask=k < K)


@triton.autotune(
    configs=kernel_configs,
    key=["K"],
)
@triton.heuristics(values={"is_fp16": _get_fp16})
@triton.jit
def _softmax_backward(
    B,
    stride_bm,
    stride_bn,
    stride_bk,
    G,
    stride_gm,
    stride_gn,
    stride_gk,
    Out,
    stride_om,
    stride_on,
    stride_ok,
    K,
    **meta,
):
    """
    Compute the softmax gradients.
    ..Note: Not autotuning for now because this would lead to broken accumulated gradients
    """

    m = tl.program_id(0)
    n = tl.program_id(1)

    # col indices
    k = tl.arange(0, meta["depth"])

    # the memory address of all the elements that we want to load can be computed as follows
    G = G + m * stride_gm + n * stride_gn + k * stride_gk
    Out = Out + m * stride_om + n * stride_on + k * stride_ok

    # load input data; pad out-of-bounds elements with 0
    g = tl.load(G, mask=k < K, other=float(0))
    o = tl.load(Out, mask=k < K, other=float(0))

    if meta["log"]:
        s = tl.sum(g, 0)
        if meta["is_fp16"]:
            o = o.to(tl.float32)
        b = g - tl.exp(o) * s
    else:
        # Step 1: Compute the intermediate sum used for the gradient
        s = tl.sum(g * o, 0)

        # Step 2: Compute the gradients
        b = o * (g - s)

    # write back to B.
    # we only write once, hence the "fused" softmax naming
    B = B + m * stride_bm + n * stride_bn + k * stride_bk
    tl.store(B, b, mask=k < K)


# Helper to handle the SPMD launch grid and error cases
class _softmax_triton(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16 if _triton_softmax_fp16_enabled else None)
    def forward(ctx, x, log_outputs):
        """
        Fused softmax implementation, using the Triton programming model.
        This only supports a reduction over the last dimension for now
        """

        assert x.ndim == 3, "This implementation only supports 3-dim tensors"

        y = torch.empty_like(x)

        # SPMD launch grid
        grid_2d = (
            x.shape[0],
            x.shape[1],
        )

        # enqueue GPU kernel
        _softmax[grid_2d](
            y,
            y.stride(0),
            y.stride(1),
            y.stride(2),
            x,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.shape[2],
            log=log_outputs,
        )

        ctx.save_for_backward(y)
        ctx.log_outputs = log_outputs
        return y

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        (out,) = ctx.saved_tensors

        assert out.ndim == 3, "This implementation only supports 3-dim tensors"

        # SPMD launch grid
        grid_2d = (
            grad.shape[0],
            grad.shape[1],
        )

        depth = next_power_of_2(out.shape[2])

        # enqueue GPU kernel
        ga = torch.empty_like(out)
        _softmax_backward[grid_2d](
            ga,
            ga.stride(0),
            ga.stride(1),
            ga.stride(2),
            grad,
            grad.stride(0),
            grad.stride(1),
            grad.stride(2),
            out,
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.shape[2],
            depth=depth,
            log=ctx.log_outputs,
        )
        return ga, None


def softmax(x: torch.Tensor) -> torch.Tensor:
    return _softmax_dispatch(x, log=False)


def log_softmax(x: torch.Tensor) -> torch.Tensor:
    return _softmax_dispatch(x, log=True)


def _softmax_dispatch(x: torch.Tensor, log: bool) -> torch.Tensor:
    # Triton is used if
    # - CUDA
    # - there's enough data to make it faster than pytorch. This could change over time, Triton is improving
    # - there was no previous failure

    global _triton_register_overflow

    try:
        if (
            torch.cuda.is_available()
            and x.is_cuda
            and x.numel()
            and not _triton_register_overflow
        ):
            return _softmax_triton.apply(x, log)
    except triton.code_gen.OutOfResources:
        # Catch cases where the current GPU does not have enough registers to hold a full tensor line
        # fallback to PyTorch's implementation, which streams the tensor in and out
        _triton_register_overflow = True
        logging.warning(
            "Triton softmax kernel register spillover caught."
            "Deactivating this kernel, please file an issue int the xFormers repository"
        )

    return torch.softmax(x, dim=-1)
