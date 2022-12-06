# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Optional, Set, Union

import torch

from .common import (
    AttentionMask,
    AttentionOpBase,
    AttentionOpDispatch,
    LowerTriangularMask,
)

try:
    from flash_attn.flash_attn_triton import (
        _flash_attn_backward as triton_flash_backward,
    )
    from flash_attn.flash_attn_triton import _flash_attn_forward as triton_flash_forward

    has_triton_flashattention = True
except ImportError:
    has_triton_flashattention = False


class Op(AttentionOpBase):
    FORWARD_OPERATOR = None
    SUPPORTED_DEVICES = {"cuda"}
    SUPPORTED_DTYPES = {torch.half, torch.bfloat16}
    SUPPORTED_MAX_K = 128
    SUPPORTED_ATTN_BIAS_TYPES: Set[Any] = {
        type(None),
        LowerTriangularMask,
        # TODO: backwards accuracy is failing for a few cases, perhaps we want to disable this for now.
        # torch.Tensor,
    }
    SUPPORTS_DROPOUT = False
    SUPPORTS_CUSTOM_SCALE = True
    NAME = "tritonflashatt"

    @classmethod
    def info(cls):
        if not has_triton_flashattention:
            return "not built"
        return "available"

    @classmethod
    def supports(cls, d: "AttentionOpDispatch") -> bool:
        if not has_triton_flashattention:
            return False
        device_capability = torch.cuda.get_device_capability(d.device)
        if not device_capability >= (7, 5):
            return False
        return super(Op, cls).supports(d)

    @classmethod
    def forward_no_grad(
        cls,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: Optional[Union[torch.Tensor, AttentionMask]],
        p: float,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        return cls.forward(
            ctx=None,
            query=query,
            key=key,
            value=value,
            attn_bias=attn_bias,
            p=p,
            scale=scale,
        )

    @classmethod
    def forward(cls, ctx, query, key, value, attn_bias, p, scale):
        softmax_scale = query.shape[-1] ** (-0.5) if scale is None else scale
        causal = isinstance(attn_bias, LowerTriangularMask)
        if not causal and attn_bias is not None and attn_bias.ndim == 3:
            B = query.shape[0]
            h = attn_bias.shape[0] // B
            attn_bias = attn_bias.reshape(B, h, attn_bias.shape[1], attn_bias.shape[2])
        bias = None if causal else attn_bias

        # Make sure that the last dimension is contiguous
        query, key, value = [
            x if x.stride(-1) == 1 else x.contiguous() for x in [query, key, value]
        ]

        o, lse, softmax_scale = triton_flash_forward(
            q=query,
            k=key,
            v=value,
            bias=bias,
            softmax_scale=softmax_scale,
            causal=causal,
        )

        if ctx is not None:
            ctx.save_for_backward(query, key, value, o, lse, bias)
            ctx.causal = causal
            ctx.softmax_scale = softmax_scale
        return o

    @staticmethod
    def backward(ctx, grad):
        q, k, v, o, lse, bias = ctx.saved_tensors
        assert not ctx.needs_input_grad[
            3
        ], "FlashAttention does not support bias gradient yet"
        # Triton's autotune causes the Tensor._version to change, and so Pytorch autograd
        # does a memcpy. To avoid this we run in inference_mode, which doesn't track the version.
        with torch.inference_mode():
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            triton_flash_backward(
                grad,
                q,
                k,
                v,
                o,
                lse,
                dq,
                dk,
                dv,
                bias=bias,
                causal=ctx.causal,
                softmax_scale=ctx.softmax_scale,
            )
        return dq, dk, dv, None, None, None
