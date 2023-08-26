# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# CREDITS: the underlying kernel comes straight from the Triton tutorials
# see https://github.com/openai/triton/blob/master/python/tutorials/05-layer-norm.py

import logging
from typing import Optional

import torch
import torch.nn as nn
import triton
from torch.cuda.amp import custom_bwd, custom_fwd

from xformers.triton.k_rms_norm import (
    rms_norm_bwd_dw,
    rms_norm_bwd_dx_fused,
    rms_norm_fw,
)

logger = logging.getLogger("xformers")


_triton_rmsnorm_fp16_enabled = False  # NOTE: PyTorch keeps layernorm as fp32
_triton_registered_warnings = False


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16 if _triton_rmsnorm_fp16_enabled else None)
    def forward(ctx, x, weight, eps):
        # catch eps being too small if the tensors are fp16
        if x.dtype == torch.float16:
            eps = max(eps, 1.6e-5)

        # allocate output
        y = torch.empty_like(x)

        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape

        # allocate std, used in the backward pass
        rstd = torch.empty((M,), dtype=torch.float32, device="cuda")

        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE_N:
            raise RuntimeError("This RMS norm doesn't support feature dim >= 64KB.")

        if not x_arg.is_contiguous() or not y.is_contiguous():
            global _triton_registered_warnings
            if not _triton_registered_warnings:
                logger.warning(
                    "Non-contiguous input tensor found. Making it contiguous,"
                    + " but could have perf or trainer implications"
                )

                _triton_registered_warnings = True

            x_arg = x_arg.contiguous()
            y = y.contiguous()

        # heuristics for number of warps.
        num_warps = min(max(BLOCK_SIZE_N // 256, 1), 16)

        # enqueue kernel
        # fmt: off
        rms_norm_fw[(M,)](
            x_arg, y, weight, rstd,
            x_arg.stride(0),
            N,
            eps,
            num_warps=num_warps,
            BLOCK_SIZE_N=BLOCK_SIZE_N
        )
        # fmt: on

        ctx.save_for_backward(x, rstd, weight)
        ctx.BLOCK_SIZE_N = BLOCK_SIZE_N
        ctx.num_warps = num_warps

        return y.reshape_as(x)

    @staticmethod
    @custom_bwd
    def backward(
        ctx, dy
    ):  # pragma: no cover  # this is covered, but called directly from C++
        x, rstd, weight = ctx.saved_tensors

        # flatten the batch dimension, if any.
        # We're interested in 'samples' x norm_dimension
        x = x.reshape(-1, x.size(-1))
        M, N = x.size()

        # heuristics for amount of parallel reduction stream for DG/DB
        GROUP_SIZE_M = 32
        if N <= 8192:
            GROUP_SIZE_M = 64
        if N <= 4096:
            GROUP_SIZE_M = 96
        if N <= 2048:
            GROUP_SIZE_M = 128
        if N <= 1024:
            GROUP_SIZE_M = 256

        if dy.dtype == torch.float32:
            GROUP_SIZE_M = GROUP_SIZE_M // 2

        # allocate output
        locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device="cuda")
        t_args = {"dtype": x.dtype, "device": x.device}
        _dw = torch.empty((GROUP_SIZE_M, x.size(-1)), **t_args)
        dw = torch.empty((x.size(-1),), **t_args)
        dy = dy.contiguous()
        dx = torch.empty_like(dy)

        # Check the tensor shapes and layouts
        # we suppose in the kernel that they have the same size and are contiguous
        assert (
            dy.numel() == x.numel()
        ), "Something is wrong in the backward graph, possibly because of an inplace operation after the rmsnorm"

        # enqueue kernel using forward pass heuristics
        # also compute partial sums for DW
        num_warps = min(max(ctx.BLOCK_SIZE_N // 256, 1), 16)

        # fmt: off
        rms_norm_bwd_dx_fused[(M,)](
            dx, dy, _dw, x,
            weight,
            rstd,
            locks,
            x.stride(0),
            N,
            GROUP_SIZE_M=GROUP_SIZE_M,
            BLOCK_SIZE_N=ctx.BLOCK_SIZE_N,
            num_warps=num_warps
        )
        # fmt: on

        def grid(meta):
            return [triton.cdiv(N, meta["BLOCK_SIZE_N"])]

        # accumulate partial sums in separate kernel
        # fmt: off
        rms_norm_bwd_dw[grid](
            _dw, dw,
            GROUP_SIZE_M,
            N,
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_N=64
        )
        # fmt: on

        dx = dx.reshape_as(dy)
        return dx, dw, None


class FusedRMSNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-06):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.epsilon = eps

    def forward(self, x):
        return _RMSNorm.apply(x, self.weight, self.epsilon)

    def init_weights(self, *args, **kwargs):
        with torch.no_grad():
            if self.weight is not None:
                self.weight.fill_(1.0)

