# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import logging
from enum import Enum
from typing import Optional

import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_bwd, custom_fwd

# CREDITS: This is adapted from the vanilla Triton example. See https://openai.com/blog/triton/
# and https://triton-lang.org/getting-started/tutorials/02-fused-softmax.html


_triton_registered_overflow = False
_triton_registered_warnings = False
_triton_softmax_fp16_enabled = False  # NOTE: PyTorch keeps softmax as fp32


class MaskType(str, Enum):
    ADD = "add"
    MUL = "mul"


def get_depth(*args, **_):
    return triton.next_power_of_2(args[-1])


# autotune: Triton will test out these configurations, and automatically pick the fastest one.
# heuristic: add arguments to the kernel call automatically given some heuristics. These arguments are passed in "meta"
# fmt: off
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["K"],
)
@triton.heuristics(values={"depth": get_depth , "is_fp16": lambda *args, **_: args[0].dtype == torch.float16})
@triton.jit
def _softmax(
    Y, X, M,
    stride_ym, stride_yn,
    stride_xm, stride_xn,
    stride_mn,
    K,
    **meta,  # extra parameters which can be automatically filled in given some heuristics
):
    # fmt: om

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
    x_ptrs = X + m * stride_xm + n * stride_xn + k

    # load input data; pad out-of-bounds elements with 0
    io_mask = k < K

    # Causal - 1: skip on the loads directly
    if meta["causal"]:
        io_mask = io_mask & (k <= n)

    x = tl.load(x_ptrs, mask=io_mask, other=float("-inf"))

    # Causal - 2: enforce correctness over a couple of misloaded values
    if meta["causal"]:
        off = float("-inf")
        off = off.to(x.dtype)
        x = tl.where(k > n, off, x)

    if meta["use_mask"]:
        mask_ptrs = M + n * stride_mn + k
        add_mask = tl.load(mask_ptrs, io_mask, other=float("-inf"))
        x += add_mask

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
    y_ptrs = Y + m * stride_ym + n * stride_yn + k

    # technically we could write only the lower triangular matrix in the causal case
    # but this is deemed to error prone
    tl.store(y_ptrs, y, mask=k < K)


# fmt: off
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    key=["K"],
)
@triton.heuristics(values={"is_fp16": lambda *args, **_: args[0].dtype == torch.float16})
@triton.jit
def _softmax_backward(
    GradIn, GradOut, Out,
    stride_bm, stride_bn,
    stride_gm, stride_gn,
    stride_om, stride_on,
    K,
    **meta,
):
    # fmt: on

    """
    Compute the softmax gradients.
    ..Note: Not autotuning for now because this would lead to broken accumulated gradients
    """

    m = tl.program_id(0)
    n = tl.program_id(1)

    # col indices
    k = tl.arange(0, meta["depth"])

    # the memory address of all the elements that we want to load can be computed as follows
    grad_out_ptrs = GradOut + m * stride_gm + n * stride_gn + k
    out_ptrs = Out + m * stride_om + n * stride_on + k

    # load input data; pad out-of-bounds elements with 0
    io_mask = k < K

    # Causal - 1: skip on the loads directly
    if meta["causal"]:
        io_mask = io_mask & (k <= n)

    g = tl.load(grad_out_ptrs, mask=io_mask, other=float(0))
    o = tl.load(out_ptrs, mask=io_mask, other=float(0))

    # Causal - 2: enforce correctness over a couple of misloaded values
    if meta["causal"]:
        zero = float(0)
        zero = zero.to(g.dtype)
        g = tl.where(k > n, zero, g)
        o = tl.where(k > n, zero, o)

    if meta["log"]:
        s = tl.sum(g, 0)
        if meta["is_fp16"]:
            o = o.to(tl.float32)
        grad_in = g - tl.exp(o) * s
    else:
        # Step 1: Compute the intermediate sum used for the gradient
        s = tl.sum(g * o, 0)

        # Step 2: Compute the gradients
        grad_in = o * (g - s)

    # write back to the input gradients
    # technically we could write only the lower triangular matrix in the causal case
    # but this is deemed to error prone
    grad_in_ptrs = GradIn + m * stride_bm + n * stride_bn + k
    tl.store(grad_in_ptrs, grad_in, mask=k < K)


# Helper to handle the SPMD launch grid and error cases
class _softmax_triton(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16 if _triton_softmax_fp16_enabled else None)
    def forward(ctx, x, mask, log_outputs, causal):
        """
        Fused softmax implementation, using the Triton programming model.
        This only supports a reduction over the last dimension for now
        """

        y = torch.empty_like(x)

        assert x.ndim == 3, "This implementation only supports 3-dim tensors"
        assert y.stride(2) == 1 and x.stride(2) == 1

        # SPMD launch grid
        grid_2d = (
            x.shape[0],
            x.shape[1],
        )

        # enqueue GPU kernel
        use_mask = True
        if mask is None:
            #  placeholder, will not be used
            mask = x
            use_mask = False
        else:
            # Make sure that the mask is binary
            assert mask.dtype == x.dtype, "An additive mask is requested"

        _softmax[grid_2d](
            y, x, mask,
            y.stride(0), y.stride(1),
            x.stride(0), x.stride(1),
            mask.stride(0),
            x.shape[2],
            log=log_outputs,
            use_mask=use_mask,
            causal=causal
        )

        ctx.save_for_backward(y)
        ctx.log_outputs = log_outputs
        ctx.causal = causal
        return y

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        (out,) = ctx.saved_tensors

        assert out.ndim == 3, "This implementation only supports 3-dim tensors"

        # SPMD launch grid
        grid_2d = (
            grad_out.shape[0],
            grad_out.shape[1],
        )

        depth = triton.next_power_of_2(out.shape[2])
        grad_in = torch.empty_like(out)  # torch.zeros is measurably slower, we'll zero out in the kernel

        assert grad_in.stride(2) == 1 and grad_out.stride(2) == 1 and out.stride(2) == 1

        # fmt: off
        _softmax_backward[grid_2d](
            grad_in, grad_out, out,
            grad_in.stride(0), grad_in.stride(1),
            grad_out.stride(0), grad_out.stride(1),
            out.stride(0), out.stride(1),
            out.shape[2],
            depth=depth,
            log=ctx.log_outputs,
            causal=ctx.causal
        )
        # fmt: on
        return grad_in, None, None, None


def softmax(x: torch.Tensor, mask: Optional[torch.Tensor] = None, causal: bool = False) -> torch.Tensor:
    r"""Applies the Softmax function to an 3-dimensional input Tensor
    rescaling them so that the elements of the n-dimensional output Tensor
    lie in the range [0,1] and sum to 1.

    Softmax is defined as:

    .. math::
        \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

    .. warning: softmax is computed on the last dimension of the input tensor.


    Args:
        x: input tensor.
        mask: optional mask, its application will be fused to the softmax computation if triton is used
        causal: optional performance optimization, if triton is used and the attention is causal

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [0, 1] and sum to 1
    """
    return _softmax_dispatch(x, log=False, mask=mask, causal=causal)


def log_softmax(x: torch.Tensor, mask: Optional[torch.Tensor] = None, causal: bool = False) -> torch.Tensor:
    r"""Applies the :math:`\log(\text{Softmax}(x))` function to an 3-dimensional
    input Tensor. The LogSoftmax formulation can be simplified as:

    .. math::
        \text{LogSoftmax}(x_{i}) = \log\left(\frac{\exp(x_i) }{ \sum_j \exp(x_j)} \right)

    Args:
        x: input tensor.

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [-inf, 0)
    """
    return _softmax_dispatch(x, log=True, mask=mask, causal=causal)


def _softmax_dispatch(x: torch.Tensor, log: bool, mask: Optional[torch.Tensor], causal: bool = False) -> torch.Tensor:
    # Triton is used if
    # - CUDA
    # - there's enough data to make it faster than pytorch. This could change over time, Triton is improving
    # - there was no previous failure

    global _triton_registered_overflow
    global _triton_registered_warnings

    try:
        if (
            torch.cuda.is_available()
            and x.is_cuda
            and not _triton_registered_overflow
        ):
            return _softmax_triton.apply(x, mask, log, causal)
    except triton.code_gen.OutOfResources:
        # Catch cases where the current GPU does not have enough registers to hold a full tensor line
        # fallback to PyTorch's implementation, which streams the tensor in and out
        _triton_registered_overflow = True
        logging.warning(
            "Triton softmax kernel register spillover caught."
            "Deactivating this kernel, please file an issue int the xFormers repository"
        )

    if causal and not _triton_registered_warnings:
        logging.warning(
            "Triton softmax could not be used. \
                The causal flags is being passed but it does not provide any benefit with PyTorch softmax."
        )
        _triton_registered_warnings = True

    if mask is not None:
        x += mask

    return torch.softmax(x, dim=-1)
