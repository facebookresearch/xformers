# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import Any, List, Optional, Set, Union

import torch

from ..common import get_xformers_operator
from .common import (
    AttentionMask,
    AttentionOpBase,
    AttentionOpDispatch,
    LowerTriangularMask,
)


class Op(AttentionOpBase):
    """xFormers' MHA kernel based on CUTLASS.
    Supports a large number of settings (including without TensorCores, f32 ...)
    and GPUs as old as P100 (Sm60)
    """

    FORWARD_OPERATOR = get_xformers_operator("efficient_attention_forward_cutlass")
    SUPPORTED_DEVICES: Set[str] = {"cuda"}
    SUPPORTED_DTYPES: Set[torch.dtype] = {torch.float, torch.half, torch.bfloat16}
    SUPPORTED_MAX_K = math.inf
    SUPPORTED_ATTN_BIAS_TYPES: Set[Any] = {type(None), LowerTriangularMask}
    SUPPORTS_DROPOUT = False
    SUPPORTS_CUSTOM_SCALE = True
    SUPPORTS_DIFFERENT_VALUE_EMBED = True
    NAME = "cutlass"

    _TEST_K: List[int] = [
        32,  # 64x64 kernel
        128,  # 64x128 kernel
        256,  # 64x128 with accumulation in gmem
    ]

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
        if attn_bias is not None and not isinstance(attn_bias, LowerTriangularMask):
            raise NotImplementedError("Unsupported attn_bias type")
        return cls.FORWARD_OPERATOR(
            query=query,
            key=key,
            value=value,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=-1,
            compute_logsumexp=False,
            causal=isinstance(attn_bias, LowerTriangularMask),
            scale=scale,
        )[0]

    @classmethod
    def forward(cls, ctx, query, key, value, attn_bias, p, scale):
        if attn_bias is not None and not isinstance(attn_bias, LowerTriangularMask):
            raise NotImplementedError("Unsupported attn_bias type")
        causal = isinstance(attn_bias, LowerTriangularMask)
        out, lse = cls.FORWARD_OPERATOR(
            query=query,
            key=key,
            value=value,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=-1,
            compute_logsumexp=True,
            causal=causal,
            scale=scale,
        )
        ctx.save_for_backward(query, key, value, lse, out)
        ctx.p = p
        ctx.causal = causal
        ctx.scale = scale
        return out

    @classmethod
    def uses_tensorcores(cls, d: "AttentionOpDispatch", is_half: bool) -> bool:
        sm_major = torch.cuda.get_device_capability(d.device)[0]
        if sm_major >= 8:
            return True
        if sm_major >= 7:
            return is_half
        return False

    @classmethod
    def supports(cls, d: "AttentionOpDispatch") -> bool:
        if not super(Op, cls).supports(d):
            return False
        cap = torch.cuda.get_device_capability(d.device)
        sm = cap[0] * 10 + cap[1]
        bits_per_scalar = {torch.float: 32, torch.half: 16, torch.bfloat16: 16}[d.dtype]
        uses_tensorcores = cls.uses_tensorcores(d, bits_per_scalar == 16)
        matmul_alignment_mn = 1
        if sm >= 80:
            matmul_alignment_mn = 4
        if uses_tensorcores:
            matmul_alignment_mn = max(matmul_alignment_mn, 128 // bits_per_scalar)
        if (d.k % matmul_alignment_mn != 0) or (d.kv % matmul_alignment_mn != 0):
            return False
        # Sm86 does not have enough shared-memory
        # See https://github.com/facebookresearch/xformers/issues/517
        if (
            d.requires_grad
            and sm >= 80
            and sm != 80
            and d.dtype is torch.float
            and max(d.kv, d.k) > 64
        ):
            return False
        return True

    @classmethod
    def backward(cls, ctx, grad):
        query, key, value, lse, out = ctx.saved_tensors

        dtype = query.dtype
        (
            grad_q,
            grad_k,
            grad_v,
        ) = torch.ops.xformers.efficient_attention_backward_cutlass(
            grad.to(dtype),
            query,
            key,
            value,
            lse,
            out.to(dtype),
            causal=ctx.causal,
            scale=ctx.scale,
        )
        return grad_q, grad_k, grad_v, None, None, None
