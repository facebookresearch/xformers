# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Iterable, List, Optional, Sequence, Set, Tuple

import torch

from ..common import get_operator, register_operator
from .attn_bias import (
    VARLEN_BIASES,
    BlockDiagonalCausalFromBottomRightMask,
    BlockDiagonalCausalMask,
    BlockDiagonalCausalWithOffsetGappyKeysMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    BlockDiagonalGappyKeysMask,
    BlockDiagonalMask,
    BlockDiagonalPaddedKeysMask,
    LowerTriangularFromBottomRightMask,
    LowerTriangularMask,
)
from .common import (
    AttentionBwOpBase,
    AttentionFwOpBase,
    Context,
    Gradients,
    Inputs,
    check_lastdim_alignment_stride1,
)
from .flash import (
    _check_needs_no_topleft,
    _convert_input_format,
    _is_causal,
    _post_process_lse,
)

FLASH_VERSION = "0.0.0"
try:
    from ... import _C_flashattention3  # type: ignore[attr-defined]
    from ..._cpp_lib import _build_metadata

    if _build_metadata is not None:
        FLASH_VERSION = _build_metadata.flash_version
except ImportError:
    try:
        from flash_attn_interface import flashattn_hopper_cuda as _C_flashattention3
    except ImportError:
        # We end up here is arch is not 90a
        _C_flashattention3 = None

if _C_flashattention3 is not None:
    # returns: out, q_padded, k_padded, v_padded, out_padded, softmax_lse, p
    @torch.library.custom_op(
        "xformers_flash3::flash_fwd", mutates_args=(), device_types=["cuda"]
    )
    def mha_fwd(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        seqused_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        p: float,
        softmax_scale: float,
        is_causal: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor,]:
        if cu_seqlens_q is None:
            assert cu_seqlens_k is None
            assert seqused_k is None
            (
                out,
                q_padded,
                k_padded,
                v_padded,
                out_padded,
                softmax_lse,
                p,
            ) = _C_flashattention3.fwd(
                query, key, value, None, softmax_scale, None, None, None, is_causal
            )
        else:
            out, q, k, v, out_padded, softmax_lse = _C_flashattention3.varlen_fwd(
                query,
                key,
                value,
                None,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_k,
                max_seqlen_q,
                max_seqlen_k,
                softmax_scale,
                is_causal,
            )
        return out, softmax_lse

    @torch.library.register_fake("xformers_flash3::flash_fwd")
    def mha_fwd_fake(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        seqused_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        p: float,
        softmax_scale: float,
        is_causal: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor,]:
        query_shape = query.shape
        out = query.new_empty(query_shape)
        # Query is (B, M, H, K) or (total_M, H, K)
        # LSE is (B, H, M) or (H, total_M)
        lse_shape = (
            (query_shape[0], query_shape[2], query_shape[1])
            if cu_seqlens_q is None
            else (query_shape[1], query_shape[0])
        )
        lse = query.new_empty(lse_shape, dtype=torch.float32)
        return out, lse

    def _create_dq_dk_dv(
        grads_share_storage: bool, query, key, value
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Create dq,dk,dv
        # If Q/K/V come from a single QKV tensor, let's put the gradient in the
        # right strides, so we can avoid a `cat`
        if grads_share_storage:
            chunk = torch.empty(
                (*query.shape[0:-2], 3, query.shape[-2], query.shape[-1]),
                dtype=query.dtype,
                device=query.device,
            )
            return chunk.select(-3, 0), chunk.select(-3, 1), chunk.select(-3, 2)
        return torch.empty_like(query), torch.empty_like(key), torch.empty_like(value)

    @torch.library.custom_op(
        "xformers_flash3::flash_bwd", mutates_args=(), device_types=["cuda"]
    )
    def mha_bwd(
        grads_share_storage: bool,
        dout: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        out: torch.Tensor,
        softmax_lse: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float,
        is_causal: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dq, dk, dv = _create_dq_dk_dv(grads_share_storage, query, key, value)
        is_deterministic = False
        if cu_seqlens_q is None:
            assert cu_seqlens_k is None
            dq, dk, dv, softmax_d, *rest = _C_flashattention3.bwd(
                dout,
                query,
                key,
                value,
                out,
                softmax_lse,
                dq,
                dk,
                dv,
                softmax_scale,
                is_causal,
                is_deterministic,
            )
        else:
            dq, dk, dv, softmax_d, *rest = _C_flashattention3.varlen_bwd(
                dout,
                query,
                key,
                value,
                out,
                softmax_lse,
                dq,
                dk,
                dv,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                softmax_scale,
                is_causal,
                is_deterministic,
            )
        return dq, dk, dv

    @torch.library.register_fake("xformers_flash3::flash_bwd")
    def mha_bwd_fake(
        grads_share_storage: bool,
        dout: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        out: torch.Tensor,
        softmax_lse: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float,
        is_causal: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dq = torch.empty_like(query)
        dk = torch.empty_like(key)
        dv = torch.empty_like(value)
        return dq, dk, dv


@register_operator
class FwOp(AttentionFwOpBase):
    """Operator that computes memory-efficient attention using \
        `Flash-Attention <https://github.com/HazyResearch/flash-attention>`_ \
        implementation.
    """

    OPERATOR = get_operator("xformers_flash3", "flash_fwd")
    SUPPORTED_DEVICES: Set[str] = {"cuda"}
    CUDA_MINIMUM_COMPUTE_CAPABILITY = (9, 0)
    SUPPORTED_DTYPES: Set[torch.dtype] = {torch.half, torch.bfloat16}
    SUPPORTED_MAX_K = 256
    SUPPORTED_MIN_K = 64
    SUPPORTED_ATTN_BIAS_TYPES: Iterable[Any] = (
        type(None),
        LowerTriangularMask,
        LowerTriangularFromBottomRightMask,
        BlockDiagonalMask,
        BlockDiagonalCausalMask,
        BlockDiagonalCausalFromBottomRightMask,
        BlockDiagonalCausalWithOffsetGappyKeysMask,
        BlockDiagonalCausalWithOffsetPaddedKeysMask,
        BlockDiagonalGappyKeysMask,
        BlockDiagonalPaddedKeysMask,
    )

    SUPPORTS_DROPOUT = False
    SUPPORTS_CUSTOM_SCALE = True
    SUPPORTS_DIFFERENT_VALUE_EMBED = False
    SUPPORTS_BMGHK = True
    SUPPORTS_PARTIAL = True
    UNPADDED_LSE = True
    NAME = f"fa3F@{FLASH_VERSION}"
    VERSION = FLASH_VERSION

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:
        reasons = super(FwOp, cls).not_supported_reasons(d)
        check_lastdim_alignment_stride1(reasons, "query", d.query, 8)
        if d.query.shape[-1] not in [64, 128, 256]:
            reasons.append("only head-dim 64,128,256 is supported")

        _check_needs_no_topleft(d, reasons)

        return reasons

    @classmethod
    def apply(
        cls, inp: Inputs, needs_gradient: bool
    ) -> Tuple[torch.Tensor, Optional[Context]]:

        original_query_shape = inp.query.shape
        out_shape = [
            *inp.query.shape[:-1],
            inp.value.shape[-1],
        ]
        (
            inp,
            cu_seqlens_q,
            max_seqlen_q,
            cu_seqlens_k,
            max_seqlen_k,
            seqused_k,
        ) = _convert_input_format(inp, supports_mqa=True)

        if inp.query.numel() > 0 and inp.key.numel() > 0:
            (out, softmax_lse,) = cls.OPERATOR(
                inp.query,
                inp.key,
                inp.value,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_k,
                max_seqlen_q,
                max_seqlen_k,
                inp.p,
                inp.scale_float,
                _is_causal(inp.attn_bias),
            )
            out = out.reshape(out_shape)
        else:
            out = torch.zeros(
                inp.query.shape, device=inp.query.device, dtype=inp.query.dtype
            )
            softmax_lse = torch.empty(
                [inp.query.shape[0], inp.query.shape[2], inp.query.shape[1]],
                device=inp.query.device,
                dtype=torch.float32,
            )
        ctx = Context(
            out=out,
            lse=softmax_lse,
        )

        if not needs_gradient:
            return out, None
        ctx = Context(
            out=out,
            lse=_post_process_lse(
                softmax_lse, inp, tuple(original_query_shape), varlen_lse_packed=True
            ),
        )
        return (out, ctx)


@register_operator
class BwOp(AttentionBwOpBase):
    __doc__ = FwOp.__doc__

    OPERATOR = get_operator("xformers_flash3", "flash_bwd")
    SUPPORTED_DEVICES = FwOp.SUPPORTED_DEVICES
    CUDA_MINIMUM_COMPUTE_CAPABILITY = FwOp.CUDA_MINIMUM_COMPUTE_CAPABILITY
    SUPPORTED_DTYPES = FwOp.SUPPORTED_DTYPES
    SUPPORTED_MAX_K = FwOp.SUPPORTED_MAX_K
    SUPPORTED_MIN_K = FwOp.SUPPORTED_MIN_K
    SUPPORTED_ATTN_BIAS_TYPES = (
        # Exclude padded or gappy masks, since seqused_k is not supported by the kernel.
        type(None),
        LowerTriangularMask,
        LowerTriangularFromBottomRightMask,
        BlockDiagonalMask,
        BlockDiagonalCausalMask,
        BlockDiagonalCausalFromBottomRightMask,
    )

    SUPPORTS_DROPOUT = FwOp.SUPPORTS_DROPOUT
    SUPPORTS_CUSTOM_SCALE = FwOp.SUPPORTS_CUSTOM_SCALE
    SUPPORTS_DIFFERENT_VALUE_EMBED = FwOp.SUPPORTS_DIFFERENT_VALUE_EMBED
    IS_DETERMINISTIC = False
    SUPPORTS_BMGHK = False
    SUPPORTS_LSE_FORMATS: Sequence[str] = ["", "varlen_flat"]
    NAME = f"fa3B@{FLASH_VERSION}"
    VERSION = FLASH_VERSION

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:
        reasons = super(BwOp, cls).not_supported_reasons(d)
        check_lastdim_alignment_stride1(reasons, "query", d.query, 8)
        _check_needs_no_topleft(d, reasons)
        if d.query.shape[-1] not in [64, 128]:
            reasons.append("only head-dim 64 or 128 is supported")

        _check_needs_no_topleft(d, reasons)
        return reasons

    @classmethod
    def apply(cls, ctx: Context, inp: Inputs, grad: torch.Tensor) -> Gradients:

        dq_shape, dk_shape, dv_shape = inp.query.shape, inp.key.shape, inp.value.shape
        (
            inp,
            cu_seqlens_q,
            max_seqlen_q,
            cu_seqlens_k,
            max_seqlen_k,
            _,  # seqused_k,
        ) = _convert_input_format(inp, supports_mqa=False)
        ctx_lse = ctx.lse

        if isinstance(inp.attn_bias, VARLEN_BIASES):
            assert ctx_lse.shape[0] == 1
            ctx_lse = ctx_lse[0]
        else:
            # NOTE: cutlass pads the last dimension, we need to slice it
            assert ctx_lse.shape[2] >= max_seqlen_q
            ctx_lse = ctx_lse[:, :, :max_seqlen_q].contiguous()

        kernel_out_shape = [
            *inp.query.shape[:-1],
            inp.value.shape[-1],
        ]
        assert grad.dtype in cls.SUPPORTED_DTYPES

        if inp.query.numel() and inp.key.numel():
            dq, dk, dv = cls.OPERATOR(
                ctx.qkv_share_storage,
                grad.reshape(kernel_out_shape).contiguous(),
                inp.query,
                inp.key,
                inp.value,
                ctx.out.reshape(kernel_out_shape),
                ctx.lse,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                softmax_scale=inp.scale_float,
                is_causal=_is_causal(inp.attn_bias),
            )
            grads = Gradients(dq, dk, dv)
        else:
            grads = Gradients(
                dq=torch.zeros_like(inp.query),
                dk=torch.zeros_like(inp.key),
                dv=torch.zeros_like(inp.value),
            )

        grads.dq = grads.dq.reshape(dq_shape)
        grads.dk = grads.dk.reshape(dk_shape)
        grads.dv = grads.dv.reshape(dv_shape)
        return grads
