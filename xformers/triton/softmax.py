# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import logging
from typing import Optional

import torch
import triton
from torch.cuda.amp import custom_bwd, custom_fwd

from xformers.triton.k_softmax import _softmax, _softmax_backward

# CREDITS: This is adapted from the vanilla Triton example. See https://openai.com/blog/triton/
# and https://triton-lang.org/getting-started/tutorials/02-fused-softmax.html


logger = logging.getLogger("xformers")


_triton_softmax_fp16_enabled = False  # NOTE: PyTorch keeps softmax as fp32
_triton_registered_warnings = False


# Helper to handle the SPMD launch grid and error cases
class _softmax_triton(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16 if _triton_softmax_fp16_enabled else None)
    def forward(ctx, x, mask, log_outputs, causal):
        """
        Fused softmax implementation, using the Triton programming model.
        This only supports a reduction over the last dimension for now
        """

        # Handle 2D/3D tensors
        x_ = x.unsqueeze(0) if x.ndim == 2 else x
        x_ = x_.flatten(0, -3)

        if not x_.is_contiguous():
            x_ = x_.contiguous()

        y = torch.empty_like(x_)
        assert (
            y.stride(2) == 1 and x_.stride(2) == 1
        ), f"{x.shape} - {x_.shape} - {x_.stride()}"

        # SPMD launch grid
        grid_2d = (
            x_.shape[0],
            x_.shape[1],
        )

        # enqueue GPU kernel
        use_mask = True
        if mask is None:
            #  placeholder, will not be used
            mask = x_
            use_mask = False
        else:
            # Make sure that the mask is binary
            assert mask.dtype == x.dtype, "An additive mask is requested"

        _softmax[grid_2d](
            y,
            x_,
            mask,
            y.stride(0),
            y.stride(1),
            x_.stride(0),
            x_.stride(1),
            mask.stride(0),
            x_.shape[2],
            log=log_outputs,
            use_mask=use_mask,
            causal=causal,
        )

        ctx.save_for_backward(y)
        ctx.log_outputs = log_outputs
        ctx.causal = causal
        return y.reshape_as(x)

    @staticmethod
    @custom_bwd
    def backward(
        ctx, grad_out
    ):  # pragma: no cover  # this is covered, but called directly from C++
        (out,) = ctx.saved_tensors

        # Handle 2D/3D tensors
        grad_out_ = grad_out.unsqueeze(0) if grad_out.ndim == 2 else grad_out
        grad_out_ = grad_out_.flatten(0, -3)

        # SPMD launch grid
        grid_2d = (
            grad_out_.shape[0],
            grad_out_.shape[1],
        )

        depth = triton.next_power_of_2(grad_out_.shape[2])
        grad_in = torch.empty_like(
            out
        )  # torch.zeros is measurably slower, we'll zero out in the kernel

        # Make sure that the tensor are contiguous
        grad_in, grad_out, out = map(lambda x: x.contiguous(), [grad_in, grad_out, out])

        # fmt: off
        _softmax_backward[grid_2d](
            grad_in, grad_out_, out,
            grad_in.stride(0), grad_in.stride(1),
            grad_out_.stride(0), grad_out_.stride(1),
            out.stride(0), out.stride(1),
            out.shape[2],
            depth=depth,
            log=ctx.log_outputs,
            causal=ctx.causal
        )
        # fmt: on
        return grad_in.reshape_as(grad_out), None, None, None


def softmax(
    x: torch.Tensor, mask: Optional[torch.Tensor] = None, causal: bool = False
) -> torch.Tensor:
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


def log_softmax(
    x: torch.Tensor, mask: Optional[torch.Tensor] = None, causal: bool = False
) -> torch.Tensor:
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


def _softmax_dispatch(
    x: torch.Tensor, log: bool, mask: Optional[torch.Tensor], causal: bool = False
) -> torch.Tensor:
    # Triton is used if
    # - CUDA
    # - there's enough data to make it faster than pytorch. This could change over time, Triton is improving
    # - there was no previous failure

    global _triton_registered_warnings

    try:
        if torch.cuda.is_available() and x.is_cuda and not _triton_registered_warnings:
            return _softmax_triton.apply(x, mask, log, causal)
    except RuntimeError as e:
        # Catch cases where the current GPU does not have enough registers to hold a full tensor line
        # fallback to PyTorch's implementation, which streams the tensor in and out
        _triton_registered_warnings = True
        logger.warning(
            "Triton softmax kernel register spillover or invalid image caught."
            "Deactivating this kernel, please file an issue int the xFormers repository"
        )
        logger.warning(e)

    if mask is not None:
        x = x + mask

    if causal:
        x = x + torch.triu(torch.full_like(x, float("-inf")), diagonal=1)

    if log:
        return torch.log_softmax(x, dim=-1)
    else:
        return torch.softmax(x, dim=-1)
