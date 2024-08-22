# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Iterable, List, Optional, Set, Tuple

import torch

from ..common import get_operator, register_operator
from .attn_bias import (
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
from .common import AttentionFwOpBase, Context, Inputs, check_lastdim_alignment_stride1
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
                query, key, value, None, softmax_scale, is_causal
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
            lse=_post_process_lse(softmax_lse, inp, tuple(original_query_shape)),
        )
        return (out, ctx)
