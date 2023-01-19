# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, List, Mapping, Optional, Set, Tuple, Union

import torch

from ..common import get_xformers_operator, register_operator
from .attn_bias import (
    AttentionBias,
    BlockDiagonalCausalMask,
    BlockDiagonalMask,
    LowerTriangularMask,
    LowerTriangularMaskWithTensorBias,
)
from .common import (
    AttentionBwOpBase,
    AttentionFwOpBase,
    Context,
    Gradients,
    Inputs,
    check_lastdim_alignment_stride1,
)


def _uses_tensorcores(sm: int, is_half: bool) -> bool:
    if sm >= 80:
        return True
    if sm >= 70:
        return is_half
    return False


def _minimum_gemm_alignment(inp: Inputs) -> int:
    if inp.device.type != "cuda":
        return 1
    cap = torch.cuda.get_device_capability(inp.device)
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
    attn_bias = inp.attn_bias
    if isinstance(attn_bias, BlockDiagonalMask):
        attn_bias.k_seqinfo.cu_seqlen = attn_bias.k_seqinfo.cu_seqlen.to(
            inp.query.device, non_blocking=True
        )
        attn_bias.q_seqinfo.cu_seqlen = attn_bias.q_seqinfo.cu_seqlen.to(
            inp.query.device, non_blocking=True
        )
        cu_seqlen_k = attn_bias.k_seqinfo.cu_seqlen
        cu_seqlen_q = attn_bias.q_seqinfo.cu_seqlen
        max_seqlen_q = attn_bias.q_seqinfo.max_seqlen
    else:
        cu_seqlen_k = None
        cu_seqlen_q = None
        max_seqlen_q = -1

    return cu_seqlen_k, cu_seqlen_q, max_seqlen_q


def _get_tensor_bias(
    attn_bias: Optional[Union[torch.Tensor, AttentionBias]]
) -> Optional[torch.Tensor]:
    if isinstance(attn_bias, torch.Tensor):
        return attn_bias
    elif isinstance(attn_bias, LowerTriangularMaskWithTensorBias):
        return attn_bias._bias
    return None


def _check_bias_alignment(
    reasons: List[str], attn_bias: Optional[Union[torch.Tensor, AttentionBias]]
) -> None:
    attn_bias_tensor = _get_tensor_bias(attn_bias)
    if attn_bias_tensor is not None:
        alignment = 128 // torch.finfo(attn_bias_tensor.dtype).bits
        check_lastdim_alignment_stride1(
            reasons, "attn_bias", attn_bias_tensor, alignment
        )


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
    SUPPORTED_ATTN_BIAS_TYPES: Set[Any] = {
        type(None),
        torch.Tensor,
        LowerTriangularMask,
        LowerTriangularMaskWithTensorBias,
        BlockDiagonalMask,
        BlockDiagonalCausalMask,
    }
    SUPPORTS_DROPOUT = True
    SUPPORTS_CUSTOM_SCALE = True
    SUPPORTS_DIFFERENT_VALUE_EMBED = True
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
        if type(inp.attn_bias) not in FwOp.SUPPORTED_ATTN_BIAS_TYPES:
            raise NotImplementedError("Unsupported attn_bias type")
        causal = isinstance(
            inp.attn_bias, (LowerTriangularMask, BlockDiagonalCausalMask)
        )
        cu_seqlen_k, cu_seqlen_q, max_seqlen_q = _get_seqlen_info(inp)
        out, lse, rng_seed, rng_offset = cls.OPERATOR(
            query=inp.query,
            key=inp.key,
            value=inp.value,
            attn_bias=_get_tensor_bias(inp.attn_bias),
            cu_seqlens_q=cu_seqlen_q,
            cu_seqlens_k=cu_seqlen_k,
            max_seqlen_q=max_seqlen_q,
            dropout_p=inp.p,
            compute_logsumexp=needs_gradient,
            causal=causal,
            scale=inp.scale,
        )
        ctx: Optional[Context] = None
        if needs_gradient:
            ctx = Context(
                out=out,
                lse=lse,
                # cutlass forward is only compatible with cutlass backward if
                # dropout is used (because of the way RNG states are passed and the
                # way random numbers are generated during backward)
                op_bw=BwOp if inp.p != 0 else None,
            )
            if inp.p != 0:
                ctx.rng_state = torch.tensor(
                    [rng_seed, rng_offset], dtype=torch.int64, device="cpu"
                )
        return out, ctx

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:
        reasons = super(FwOp, cls).not_supported_reasons(d)
        matmul_alignment_mn = _minimum_gemm_alignment(d)
        check_lastdim_alignment_stride1(reasons, "query", d.query, matmul_alignment_mn)
        check_lastdim_alignment_stride1(reasons, "value", d.value, matmul_alignment_mn)
        _check_bias_alignment(reasons, d.attn_bias)
        return reasons


@register_operator
class BwOp(AttentionBwOpBase):
    OPERATOR = get_xformers_operator("efficient_attention_backward_cutlass")
    SUPPORTED_DEVICES = FwOp.SUPPORTED_DEVICES
    SUPPORTED_DTYPES = FwOp.SUPPORTED_DTYPES
    SUPPORTED_MAX_K = FwOp.SUPPORTED_MAX_K
    SUPPORTED_ATTN_BIAS_TYPES: Set[Any] = {
        type(None),
        torch.Tensor,
        LowerTriangularMask,
        # TODO: Fix handling of gradient through the fMHA autograd function
        # LowerTriangularMaskWithTensorBias,
    }
    SUPPORTS_ATTN_BIAS_GRAD = True
    SUPPORTS_DROPOUT = FwOp.SUPPORTS_DROPOUT
    SUPPORTS_CUSTOM_SCALE = FwOp.SUPPORTS_CUSTOM_SCALE
    SUPPORTS_DIFFERENT_VALUE_EMBED = FwOp.SUPPORTS_DIFFERENT_VALUE_EMBED
    NAME = "cutlassB"

    ERROR_ATOL: Mapping[torch.dtype, float] = {
        torch.float: 5e-4,
        # increased from 9e-2, more opportunities for numerical errors when bias is
        # used, noticed in gK on SM80
        torch.half: 9.5e-2,
        torch.bfloat16: 7e-1,
    }
    ERROR_RTOL: Mapping[torch.dtype, float] = {
        torch.float: 1e-4,
        torch.half: 2e-2,
        torch.bfloat16: 1e-1,
    }

    _TEST_K: List[int] = [
        32,  # 64x64 kernel
        128,  # 64x128/128x128 kernel
        256,  # 64x128 with accumulation in gmem
    ]

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:
        reasons = super(BwOp, cls).not_supported_reasons(d)
        matmul_alignment_mn = _minimum_gemm_alignment(d)

        check_lastdim_alignment_stride1(reasons, "query", d.query, matmul_alignment_mn)
        check_lastdim_alignment_stride1(reasons, "key", d.key, matmul_alignment_mn)
        check_lastdim_alignment_stride1(reasons, "value", d.value, matmul_alignment_mn)
        _check_bias_alignment(reasons, d.attn_bias)
        attn_bias_tensor = _get_tensor_bias(d.attn_bias)

        # Backprop of gradient through broadcasted bias is not supported
        if attn_bias_tensor is not None and attn_bias_tensor.requires_grad:
            # Don't forget that inputs are either in BMK or BMHK!
            if d.query.ndim == 3 and attn_bias_tensor.ndim == 3:
                expected_bias_shape = (*d.query.shape[:2], d.key.shape[1])
            else:
                # bias is B H Mq Mk
                expected_bias_shape = (
                    d.query.shape[0],
                    d.query.shape[2] if d.query.ndim == 4 else 1,
                    d.query.shape[1],
                    d.key.shape[1],
                )
            if tuple(attn_bias_tensor.shape) != expected_bias_shape:
                reasons.append(
                    "Broadcasting the `attn_bias` tensor is not supported "
                    f"(shape: {tuple(attn_bias_tensor.shape)}"
                    f"/ expected: {expected_bias_shape})"
                )
        if d.device.type == "cuda":
            cap = torch.cuda.get_device_capability(d.device)
            sm = cap[0] * 10 + cap[1]
            # Sm86 does not have enough shared-memory
            # See https://github.com/facebookresearch/xformers/issues/517
            if (
                sm >= 80
                and sm != 80
                and d.query.dtype is torch.float
                and max(d.query.shape[-1], d.key.shape[-1]) > 64
            ):
                reasons.append(
                    f"Sm{sm} does not have enough shared-memory to run this kernel"
                    " - see https://github.com/facebookresearch/xformers/issues/517"
                )
        return reasons

    @classmethod
    def apply(cls, ctx: Context, inp: Inputs, grad: torch.Tensor) -> Gradients:
        if type(inp.attn_bias) not in BwOp.SUPPORTED_ATTN_BIAS_TYPES:
            raise NotImplementedError("Unsupported attn_bias type")

        causal = isinstance(
            inp.attn_bias, (LowerTriangularMask, BlockDiagonalCausalMask)
        )
        dtype = inp.query.dtype

        rng_seed = rng_offset = 0
        if inp.p != 0.0:
            if (
                ctx.rng_state is None
                or ctx.rng_state.dtype != torch.int64
                or ctx.rng_state.device.type != "cpu"
                or ctx.rng_state.shape != (2,)
            ):
                raise NotImplementedError(f"Invalid rng_state: {ctx.rng_state}")
            rng_seed, rng_offset = ctx.rng_state.tolist()

        force_pad_inf = torch.cuda.get_device_capability(inp.query.device) == (7, 5)
        (grad_q, grad_k, grad_v, grad_bias) = cls.OPERATOR(
            grad.to(dtype),
            inp.query,
            inp.key,
            inp.value,
            _get_tensor_bias(inp.attn_bias),
            ctx.get_padded_lse(32, force_pad_inf=force_pad_inf),
            ctx.out.to(dtype),
            dropout_p=inp.p,
            # if not using dropout, seed and offset are irrelevant but still expected
            # in function signature so just pass 0
            # seed and offset could be None if a different FW op other than cutlass
            # was used.
            rng_seed=rng_seed,
            rng_offset=rng_offset,
            causal=causal,
            scale=inp.scale,
        )

        # c++/CUDA implementation returns an uninitialized tensor if bias doesn't
        # require grad
        if not (
            isinstance(inp.attn_bias, torch.Tensor) and inp.attn_bias.requires_grad
        ):
            grad_bias = None

        return Gradients(dq=grad_q, dk=grad_k, dv=grad_v, db=grad_bias)
