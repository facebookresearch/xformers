# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import replace
from enum import Enum
from typing import Any, Iterable, List, Mapping, Optional, Set, Tuple, Union

import torch

from ..common import get_operator, register_operator
from . import attn_bias
from .attn_bias import (
    AttentionBias,
    BlockDiagonalCausalFromBottomRightMask,
    BlockDiagonalCausalLocalAttentionFromBottomRightMask,
    BlockDiagonalCausalLocalAttentionMask,
    BlockDiagonalCausalLocalAttentionPaddedKeysMask,
    BlockDiagonalCausalMask,
    BlockDiagonalCausalWithOffsetGappyKeysMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    BlockDiagonalGappyKeysMask,
    BlockDiagonalMask,
    BlockDiagonalPaddedKeysMask,
    LowerTriangularFromBottomRightLocalAttentionMask,
    LowerTriangularFromBottomRightMask,
    LowerTriangularMask,
    LowerTriangularMaskWithTensorBias,
    PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
    PagedBlockDiagonalGappyKeysMask,
    PagedBlockDiagonalPaddedKeysMask,
)
from .common import (
    AttentionBwOpBase,
    AttentionFwOpBase,
    check_lastdim_alignment_stride1,
    Context,
    Gradients,
    Inputs,
)


def _minimum_gemm_alignment(inp: Inputs) -> int:
    return 1


def _get_seqlen_info(
    inp: Inputs,
) -> Tuple[
    Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], int, int
]:
    attn_bias = inp.attn_bias
    if isinstance(
        attn_bias,
        (
            BlockDiagonalMask,
            BlockDiagonalPaddedKeysMask,
            BlockDiagonalGappyKeysMask,
            PagedBlockDiagonalPaddedKeysMask,
            PagedBlockDiagonalGappyKeysMask,
        ),
    ):
        attn_bias.k_seqinfo.to(inp.query.device)
        attn_bias.q_seqinfo.to(inp.query.device)
        seqstart_k = attn_bias.k_seqinfo.seqstart
        seqstart_q = attn_bias.q_seqinfo.seqstart
        max_seqlen_q = attn_bias.q_seqinfo.max_seqlen
        max_seqlen_k = attn_bias.k_seqinfo.max_seqlen
        seqlen = (
            None
            if isinstance(attn_bias, BlockDiagonalMask)
            else attn_bias.k_seqinfo.seqlen
        )
    else:
        seqstart_k = None
        seqstart_q = None
        max_seqlen_q = -1
        max_seqlen_k = -1
        seqlen = None

    if isinstance(attn_bias, PagedBlockDiagonalGappyKeysMask):
        seqstart_k = attn_bias.k_seqinfo.seqstart[:-1]
        seqlen = seqlen - seqstart_k  # type: ignore

    return seqstart_k, seqstart_q, seqlen, max_seqlen_q, max_seqlen_k


def _get_tensor_bias(
    attn_bias: Optional[Union[torch.Tensor, AttentionBias]],
) -> Optional[torch.Tensor]:
    if isinstance(attn_bias, LowerTriangularMaskWithTensorBias):
        return attn_bias._bias
    if isinstance(attn_bias, torch.Tensor):
        return attn_bias
    return None


def _check_bias_alignment(
    reasons: List[str], attn_bias: Optional[Union[torch.Tensor, AttentionBias]]
) -> None:
    attn_bias_tensor = _get_tensor_bias(attn_bias)
    if attn_bias_tensor is not None:
        alignment = 128 // torch.finfo(attn_bias_tensor.dtype).bits
        show_padding_hint = False
        for d in range(attn_bias_tensor.ndim - 1):
            if attn_bias_tensor.stride(d) % alignment != 0:
                reasons.append(
                    f"attn_bias.stride(-2) % {alignment} != 0 (attn_bias.stride() = {attn_bias_tensor.stride()})"
                )
                show_padding_hint = True
        if show_padding_hint:
            reasons.append(
                """\
HINT: To use an `attn_bias` with a sequence length that is not a multiple of 8, \
you need to ensure memory is aligned by slicing a bigger tensor. \
Example: use `attn_bias = torch.zeros([1, 1, 5, 8])[:,:,:,:5]` instead of `torch.zeros([1, 1, 5, 5])`"""
            )
        # We can have stride=0 sometimes if dimension=1
        if attn_bias_tensor.stride(-1) > 1:
            reasons.append(
                f"attn_bias.stride(-1) > 1 (attn_bias.stride() = {attn_bias_tensor.stride()}) - "
                "you should call `.contiguous()` on the bias"
            )


class _CustomMaskType(int, Enum):
    """
    (Matches CustomMaskType in C++.)
    """

    NoCustomMask = 0
    CausalFromTopLeft = 1
    CausalFromBottomRight = 2


def _custom_mask_type(bias: Optional[Union[torch.Tensor, AttentionBias]]) -> int:
    if isinstance(
        bias,
        (
            LowerTriangularMask,
            BlockDiagonalCausalMask,
            BlockDiagonalCausalLocalAttentionMask,
        ),
    ):
        return int(_CustomMaskType.CausalFromTopLeft)
    if isinstance(
        bias,
        (
            LowerTriangularFromBottomRightMask,
            LowerTriangularFromBottomRightLocalAttentionMask,
            attn_bias.BlockDiagonalCausalFromBottomRightMask,
            BlockDiagonalCausalWithOffsetPaddedKeysMask,
            BlockDiagonalCausalLocalAttentionPaddedKeysMask,
            BlockDiagonalCausalLocalAttentionFromBottomRightMask,
            PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
        ),
    ):
        return int(_CustomMaskType.CausalFromBottomRight)
    return int(_CustomMaskType.NoCustomMask)


@register_operator
class FwOp(AttentionFwOpBase):
    """xFormers' MHA kernel based on Composable Kernel."""

    OPERATOR = get_operator("xformers", "efficient_attention_forward_ck")
    SUPPORTED_DEVICES: Set[str] = {"cuda"}
    SUPPORTED_DTYPES: Set[torch.dtype] = {torch.half, torch.bfloat16}
    SUPPORTED_MAX_K = 512

    SUPPORTED_ATTN_BIAS_TYPES: Iterable[Any] = (
        type(None),
        torch.Tensor,
        LowerTriangularMask,
        LowerTriangularFromBottomRightMask,
        LowerTriangularFromBottomRightLocalAttentionMask,
        LowerTriangularMaskWithTensorBias,
        BlockDiagonalMask,
        BlockDiagonalCausalMask,
        BlockDiagonalCausalWithOffsetGappyKeysMask,
        BlockDiagonalCausalWithOffsetPaddedKeysMask,
        BlockDiagonalCausalLocalAttentionPaddedKeysMask,
        BlockDiagonalGappyKeysMask,
        BlockDiagonalPaddedKeysMask,
        BlockDiagonalCausalFromBottomRightMask,
        BlockDiagonalCausalLocalAttentionMask,
        BlockDiagonalCausalLocalAttentionFromBottomRightMask,
        PagedBlockDiagonalPaddedKeysMask,
        PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
        PagedBlockDiagonalGappyKeysMask,
    )

    SUPPORTS_DROPOUT = True
    SUPPORTS_CUSTOM_SCALE = True
    SUPPORTS_DIFFERENT_VALUE_EMBED = True
    SUPPORTS_PARTIAL = True
    SUPPORTS_BMGHK = True
    NAME = "ckF"

    ERROR_ATOL: Mapping[torch.dtype, float] = {
        torch.float: 3e-4,
        torch.half: 6e-3,
        torch.bfloat16: 2.8e-2,
    }
    ERROR_RTOL: Mapping[torch.dtype, float] = {
        torch.float: 2e-5,
        torch.half: 3e-3,
        torch.bfloat16: 2e-2,
    }

    _TEST_K: List[int] = [
        32,  # 64x64 kernel
        96,
        128,  # 64x128 kernel
        256,  # 64x128 with accumulation in gmem
        512,
    ]

    @classmethod
    def apply(
        cls, inp: Inputs, needs_gradient: bool
    ) -> Tuple[torch.Tensor, Optional[Context]]:
        if type(inp.attn_bias) not in FwOp.SUPPORTED_ATTN_BIAS_TYPES:
            raise NotImplementedError("Unsupported attn_bias type")
        if inp.query.ndim in [1, 2, 3]:
            raise NotImplementedError("Unsupported number of dimensions")
        if inp.query.ndim in [4]:
            return cls.apply_bmhk(inp, needs_gradient=needs_gradient)
        assert inp.query.ndim == 5, f"query has shape {inp.query.shape}"
        ctx: Optional[Context] = None

        # when the input is expanded 5-D, the group dimension has zero stride
        if inp.key.stride()[3] == 0:
            assert (
                inp.value.stride()[3] == 0
            ), "key and value should be expanded in the same way"
            k_shape = inp.key.size()
            k_stride = inp.key.stride()
            key = inp.key.as_strided(
                (k_shape[0], k_shape[1], k_shape[2], k_shape[4]),
                (k_stride[0], k_stride[1], k_stride[2], k_stride[4]),
            )
            v_shape = inp.value.size()
            v_stride = inp.value.stride()
            value = inp.value.as_strided(
                (v_shape[0], v_shape[1], v_shape[2], v_shape[4]),
                (v_stride[0], v_stride[1], v_stride[2], v_stride[4]),
            )
        else:
            key = inp.key.flatten(2, 3)
            value = inp.value.flatten(2, 3)

        [_, _, G, Hq, _] = inp.query.shape
        attn_bias_replace = inp.attn_bias
        if isinstance(inp.attn_bias, LowerTriangularMaskWithTensorBias):
            bias_tensor = _get_tensor_bias(inp.attn_bias)
            if bias_tensor is not None and bias_tensor.ndim == 5:
                attn_bias_replace = LowerTriangularMaskWithTensorBias(
                    bias_tensor.flatten(1, 2)
                )
        elif isinstance(inp.attn_bias, torch.Tensor) and inp.attn_bias.ndim == 5:
            attn_bias_replace = inp.attn_bias.flatten(1, 2)
        inp = replace(
            inp,
            query=inp.query.flatten(2, 3),
            key=key,
            value=value,
            attn_bias=attn_bias_replace,
        )
        out, ctx = cls.apply_bmhk(inp, needs_gradient=needs_gradient)
        out = out.unflatten(2, (G, Hq))
        if ctx is not None:
            lse = ctx.lse.unflatten(1, (G, Hq))
            ctx = replace(ctx, lse=lse, out=out)
        return out, ctx

    @classmethod
    def apply_bmhk(
        cls, inp: Inputs, needs_gradient: bool
    ) -> Tuple[torch.Tensor, Optional[Context]]:
        if type(inp.attn_bias) not in FwOp.SUPPORTED_ATTN_BIAS_TYPES:
            raise NotImplementedError("Unsupported attn_bias type")
        seqstart_k, seqstart_q, seqlen_k, max_seqlen_q, _ = _get_seqlen_info(inp)
        out, lse, rng_seed, rng_offset = cls.OPERATOR(
            query=inp.query,
            key=inp.key,
            value=inp.value,
            attn_bias=_get_tensor_bias(inp.attn_bias),
            seqstart_q=seqstart_q,
            seqstart_k=seqstart_k,
            max_seqlen_q=max_seqlen_q,
            dropout_p=inp.p,
            compute_logsumexp=needs_gradient,
            custom_mask_type=_custom_mask_type(inp.attn_bias),
            scale=inp.scale,
            seqlen_k=seqlen_k,
            window_size=(
                inp.attn_bias._window_size
                if isinstance(
                    inp.attn_bias,
                    (
                        BlockDiagonalCausalLocalAttentionMask,
                        BlockDiagonalCausalLocalAttentionFromBottomRightMask,
                        LowerTriangularFromBottomRightLocalAttentionMask,
                        BlockDiagonalCausalLocalAttentionPaddedKeysMask,
                    ),
                )
                else None
            ),
            block_tables=(
                inp.attn_bias.block_tables
                if isinstance(
                    inp.attn_bias,
                    (
                        PagedBlockDiagonalPaddedKeysMask,
                        PagedBlockDiagonalGappyKeysMask,
                    ),
                )
                else None
            ),
            page_size=(
                inp.attn_bias.page_size
                if isinstance(
                    inp.attn_bias,
                    (
                        PagedBlockDiagonalPaddedKeysMask,
                        PagedBlockDiagonalGappyKeysMask,
                    ),
                )
                else None
            ),
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
    __doc__ = FwOp.__doc__

    OPERATOR = get_operator("xformers", "efficient_attention_backward_ck")
    SUPPORTED_DEVICES = FwOp.SUPPORTED_DEVICES
    SUPPORTED_DTYPES = FwOp.SUPPORTED_DTYPES
    SUPPORTED_MAX_K = 256
    SUPPORTED_ATTN_BIAS_TYPES: Iterable[Any] = (
        type(None),
        torch.Tensor,
        LowerTriangularMask,
        LowerTriangularFromBottomRightMask,
        LowerTriangularFromBottomRightLocalAttentionMask,
        # TODO: Fix handling of gradient through the fMHA autograd function
        # LowerTriangularMaskWithTensorBias,
        BlockDiagonalMask,
        BlockDiagonalCausalMask,
        attn_bias.BlockDiagonalCausalFromBottomRightMask,
        attn_bias.BlockDiagonalCausalLocalAttentionMask,
    )
    SUPPORTS_ATTN_BIAS_GRAD = True
    SUPPORTS_DROPOUT = FwOp.SUPPORTS_DROPOUT
    SUPPORTS_CUSTOM_SCALE = FwOp.SUPPORTS_CUSTOM_SCALE
    SUPPORTS_DIFFERENT_VALUE_EMBED = FwOp.SUPPORTS_DIFFERENT_VALUE_EMBED
    SUPPORTS_UNPADDED_LSE = True
    NAME = "ckB"

    _TEST_K: List[int] = [
        32,  # 64x64 kernel
        64,
        96,
        128,  # 64x128/128x128 kernel
        256,
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

        return reasons

    @classmethod
    def apply(cls, ctx: Context, inp: Inputs, grad: torch.Tensor) -> Gradients:
        if type(inp.attn_bias) not in BwOp.SUPPORTED_ATTN_BIAS_TYPES:
            raise NotImplementedError("Unsupported attn_bias type")

        seqstart_k, seqstart_q, seqlen_k, max_seqlen_q, max_seqlen_k = _get_seqlen_info(
            inp
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

        (grad_q, grad_k, grad_v, grad_bias) = cls.OPERATOR(
            grad.to(dtype),
            inp.query,
            inp.key,
            inp.value,
            attn_bias=_get_tensor_bias(inp.attn_bias),
            seqstart_q=seqstart_q,
            seqstart_k=seqstart_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            seqlen_k=seqlen_k,
            logsumexp=ctx.lse,
            output=ctx.out.to(dtype),
            dropout_p=inp.p,
            # if not using dropout, seed and offset are irrelevant but still expected
            # in function signature so just pass 0
            # seed and offset could be None if a different FW op other than cutlass
            # was used.
            rng_seed=rng_seed,
            rng_offset=rng_offset,
            custom_mask_type=_custom_mask_type(inp.attn_bias),
            scale=inp.scale,
            window_size=(
                inp.attn_bias._window_size
                if isinstance(
                    inp.attn_bias,
                    (
                        BlockDiagonalCausalLocalAttentionMask,
                        BlockDiagonalCausalLocalAttentionFromBottomRightMask,
                        LowerTriangularFromBottomRightLocalAttentionMask,
                    ),
                )
                else None
            ),
        )

        # c++/CUDA implementation returns an uninitialized tensor if bias doesn't
        # require grad
        if not (
            isinstance(inp.attn_bias, torch.Tensor) and inp.attn_bias.requires_grad
        ):
            grad_bias = None

        return Gradients(dq=grad_q, dk=grad_k, dv=grad_v, db=grad_bias)
