# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, List, Optional, Set, Tuple

import torch

from ..common import get_xformers_operator, register_operator
from .common import (
    AttentionBwOpBase,
    AttentionFwOpBase,
    Context,
    Gradients,
    Inputs,
    LowerTriangularMask,
)
from .tensor_with_seqlen import TensorWithSeqLen


def _uses_tensorcores(sm: int, is_half: bool) -> bool:
    if sm >= 80:
        return True
    if sm >= 70:
        return is_half
    return False


def _minimum_gemm_alignment(inp: Inputs) -> int:
    cap = torch.cuda.get_device_capability(inp.query.device)
    sm = cap[0] * 10 + cap[1]
    bits_per_scalar = {torch.float: 32, torch.half: 16, torch.bfloat16: 16}[
        inp.query.dtype
    ]
    uses_tensorcores = _uses_tensorcores(sm, bits_per_scalar == 16)
    matmul_alignment_mn = 1
    if sm >= 80:
        matmul_alignment_mn = 4
    if uses_tensorcores:
        matmul_alignment_mn = max(matmul_alignment_mn, 128 // bits_per_scalar)
    return matmul_alignment_mn


def _get_seqlen_info(
    inp: Inputs,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
    MISMATCH_ERR = (
        "All of query/key/value should have seqlen information, or none of them"
    )

    if isinstance(inp.key, TensorWithSeqLen):
        assert isinstance(inp.query, TensorWithSeqLen), MISMATCH_ERR
        assert isinstance(inp.value, TensorWithSeqLen), MISMATCH_ERR
        cu_seqlen_k = inp.key.cu_seqlen
        cu_seqlen_q = inp.query.cu_seqlen
        max_seqlen_q = inp.query.max_seqlen
    else:
        assert not isinstance(inp.query, TensorWithSeqLen), MISMATCH_ERR
        assert not isinstance(inp.value, TensorWithSeqLen), MISMATCH_ERR
        cu_seqlen_k = None
        cu_seqlen_q = None
        max_seqlen_q = -1

    return cu_seqlen_k, cu_seqlen_q, max_seqlen_q


@register_operator
class FwOp(AttentionFwOpBase):
    """xFormers' MHA kernel based on CUTLASS.
    Supports a large number of settings (including without TensorCores, f32 ...)
    and GPUs as old as P100 (Sm60)
    """

    OPERATOR = get_xformers_operator("efficient_attention_forward_cutlass")
    SUPPORTED_DEVICES: Set[str] = {"cuda"}
    SUPPORTED_DTYPES: Set[torch.dtype] = {torch.float, torch.half, torch.bfloat16}
    SUPPORTED_MAX_K = 65536
    SUPPORTED_ATTN_BIAS_TYPES: Set[Any] = {type(None), LowerTriangularMask}
    SUPPORTS_DROPOUT = False
    SUPPORTS_CUSTOM_SCALE = True
    SUPPORTS_DIFFERENT_VALUE_EMBED = True
    SUPPORTS_TENSOR_WITH_SEQLEN = True
    NAME = "cutlassF"

    _TEST_K: List[int] = [
        32,  # 64x64 kernel
        128,  # 64x128 kernel
        256,  # 64x128 with accumulation in gmem
    ]

    @classmethod
    def apply(
        cls, inp: Inputs, needs_gradient: bool
    ) -> Tuple[torch.Tensor, Optional[Context]]:
        if inp.attn_bias is not None and not isinstance(
            inp.attn_bias, LowerTriangularMask
        ):
            raise NotImplementedError("Unsupported attn_bias type")
        causal = isinstance(inp.attn_bias, LowerTriangularMask)
        cu_seqlen_k, cu_seqlen_q, max_seqlen_q = _get_seqlen_info(inp)
        out, lse = cls.OPERATOR(
            query=inp.query,
            key=inp.key,
            value=inp.value,
            cu_seqlens_q=cu_seqlen_q,
            cu_seqlens_k=cu_seqlen_k,
            max_seqlen_q=max_seqlen_q,
            compute_logsumexp=needs_gradient,
            causal=causal,
            scale=inp.scale,
        )
        ctx: Optional[Context] = None
        if needs_gradient:
            ctx = Context(lse=lse, out=out)
        return out, ctx

    @classmethod
    def supports(cls, d: Inputs) -> bool:
        if not super(FwOp, cls).supports(d):
            return False
        matmul_alignment_mn = _minimum_gemm_alignment(d)
        if (d.query.shape[-1] % matmul_alignment_mn != 0) or (
            d.value.shape[-1] % matmul_alignment_mn != 0
        ):
            return False
        return True


@register_operator
class BwOp(AttentionBwOpBase):
    OPERATOR = get_xformers_operator("efficient_attention_backward_cutlass")
    SUPPORTED_DEVICES = FwOp.SUPPORTED_DEVICES
    SUPPORTED_DTYPES = FwOp.SUPPORTED_DTYPES
    SUPPORTED_MAX_K = FwOp.SUPPORTED_MAX_K
    SUPPORTED_ATTN_BIAS_TYPES = FwOp.SUPPORTED_ATTN_BIAS_TYPES
    SUPPORTS_DROPOUT = FwOp.SUPPORTS_DROPOUT
    SUPPORTS_CUSTOM_SCALE = FwOp.SUPPORTS_CUSTOM_SCALE
    SUPPORTS_DIFFERENT_VALUE_EMBED = FwOp.SUPPORTS_DIFFERENT_VALUE_EMBED
    SUPPORTS_TENSOR_WITH_SEQLEN = False
    NAME = "cutlassB"

    _TEST_K: List[int] = [
        32,  # 64x64 kernel
        128,  # 64x128/128x128 kernel
        256,  # 64x128 with accumulation in gmem
    ]

    @classmethod
    def supports(cls, d: Inputs) -> bool:
        if not FwOp.supports(d):
            return False
        cap = torch.cuda.get_device_capability(d.query.device)
        sm = cap[0] * 10 + cap[1]
        # Sm86 does not have enough shared-memory
        # See https://github.com/facebookresearch/xformers/issues/517
        if (
            sm >= 80
            and sm != 80
            and d.query.dtype is torch.float
            and max(d.query.shape[-1], d.key.shape[-1]) > 64
        ):
            return False
        matmul_alignment_mn = _minimum_gemm_alignment(d)
        if (
            (d.query.shape[-1] % matmul_alignment_mn != 0)
            or (d.value.shape[-1] % matmul_alignment_mn != 0)
            or (d.key.shape[-1] % matmul_alignment_mn != 0)
        ):
            return False
        return True

    @classmethod
    def apply(cls, ctx: Context, inp: Inputs, grad: torch.Tensor) -> Gradients:
        if inp.attn_bias is not None and not isinstance(
            inp.attn_bias, LowerTriangularMask
        ):
            raise NotImplementedError("Unsupported attn_bias type")
        causal = isinstance(inp.attn_bias, LowerTriangularMask)
        dtype = inp.query.dtype

        force_pad_inf = torch.cuda.get_device_capability(inp.query.device) == (7, 5)
        (grad_q, grad_k, grad_v,) = cls.OPERATOR(
            grad.to(dtype),
            inp.query,
            inp.key,
            inp.value,
            ctx.get_padded_lse(32, force_pad_inf=force_pad_inf),
            ctx.out.to(dtype),
            causal=causal,
            scale=inp.scale,
        )
        return Gradients(dq=grad_q, dk=grad_k, dv=grad_v)
