# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import importlib.util
import logging
import os
from typing import Any, Iterable, List, Optional, Sequence, Set, Tuple, TypeVar

import torch
from torch._C import parse_schema
from torch.utils.flop_counter import (
    _unpack_flash_attention_nested_shapes,
    register_flop_formula,
)

from ..common import get_operator, register_operator
from .attn_bias import (
    BlockDiagonalCausalFromBottomRightMask,
    BlockDiagonalCausalLocalAttentionFromBottomRightMask,
    BlockDiagonalCausalLocalAttentionMask,
    BlockDiagonalCausalLocalAttentionPaddedKeysMask,
    BlockDiagonalCausalMask,
    BlockDiagonalCausalWithOffsetGappyKeysMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    BlockDiagonalGappyKeysMask,
    BlockDiagonalLocalAttentionPaddedKeysMask,
    BlockDiagonalMask,
    BlockDiagonalPaddedKeysMask,
    LocalAttentionFromBottomRightMask,
    LowerTriangularFromBottomRightLocalAttentionMask,
    LowerTriangularFromBottomRightMask,
    LowerTriangularMask,
    PagedBlockDiagonalCausalWithOffsetGappyKeysMask,
    PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
    PagedBlockDiagonalGappyKeysMask,
    PagedBlockDiagonalPaddedKeysMask,
    VARLEN_BIASES,
)
from .common import (
    AttentionBwOpBase,
    AttentionFwOpBase,
    check_lastdim_alignment_stride1,
    Context,
    Gradients,
    Inputs,
    ScaledTensor,
)
from .flash import (
    _check_needs_no_topleft,
    _convert_input_format,
    _is_causal,
    _post_process_lse,
    _window_size,
)

FLASH_VERSION = "0.0.0"
logger = logging.getLogger(__name__)

T = TypeVar("T")


def maybe_contiguous(x: T) -> T:
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x  # type: ignore[attr-defined]


def _flash_attention3_incompatible_reason() -> Optional[str]:
    if not hasattr(torch.ops.flash_attn_3, "fwd") or not hasattr(
        torch.ops.flash_attn_3, "bwd"
    ):
        return "PyTorch has no `flash_attn_3` - is your Flash-Attention version recent enough?"
    if not torch.ops.flash_attn_3.fwd.default._schema.is_backward_compatible_with(  # type: ignore
        parse_schema(
            "flash_attn_3::fwd(Tensor q, Tensor k, Tensor v, Tensor(k_new!)? k_new=None, "
            "Tensor(v_new!)? v_new=None, Tensor? q_v=None, Tensor(out!)? out=None, "
            "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, "
            "Tensor? cu_seqlens_k_new=None, Tensor? seqused_q=None, Tensor? seqused_k=None, "
            "int? max_seqlen_q=None, int? max_seqlen_k=None, Tensor? page_table=None, "
            "Tensor? kv_batch_idx=None, Tensor? leftpad_k=None, Tensor? rotary_cos=None, Tensor? rotary_sin=None, "
            "Tensor? seqlens_rotary=None, Tensor? q_descale=None, Tensor? k_descale=None, Tensor? v_descale=None, "
            "float? softmax_scale=None, bool is_causal=False, int window_size_left=-1, int window_size_right=-1, "
            "int attention_chunk=0, float softcap=0., bool is_rotary_interleaved=False, "
            "Tensor? scheduler_metadata=None, int num_splits=0, bool? pack_gqa=None, int sm_margin=0) "
            "-> (Tensor(out!), Tensor, Tensor, Tensor)"
        )
    ):
        return "flash_attn_3::fwd operator is not compatible"
    if not torch.ops.flash_attn_3.bwd.default._schema.is_backward_compatible_with(  # type: ignore
        parse_schema(
            "flash_attn_3::bwd(Tensor dout, Tensor q, Tensor k, Tensor v, Tensor out, Tensor softmax_lse, "
            "Tensor(dq!)? dq=None, Tensor(dk!)? dk=None, Tensor(dv!)? dv=None, Tensor? cu_seqlens_q=None, "
            "Tensor? cu_seqlens_k=None, Tensor? seqused_q=None, Tensor? seqused_k=None, int? max_seqlen_q=None, "
            "int? max_seqlen_k=None, float? softmax_scale=None, bool is_causal=False, int window_size_left=-1, "
            "int window_size_right=-1, float softcap=0., bool deterministic=False, int sm_margin=0) "
            "-> (Tensor(dq!), Tensor(dk!), Tensor(dv!), Tensor, Tensor, Tensor, Tensor, Tensor)"
        )
    ):
        return "flash_attn_3::bwd operator is not compatible"
    return None


FLASH3_HAS_PAGED_ATTENTION = True
FLASH3_HAS_FLOAT8 = False
FLASH3_HAS_DETERMINISTIC_MODE = False
_C_flashattention3 = None
if importlib.util.find_spec("...flash_attn_3._C", package=__package__):
    from ..._cpp_lib import _build_metadata
    from ...flash_attn_3 import _C  # type: ignore[attr-defined]  # noqa: F401

    if _build_metadata is not None:
        FLASH_VERSION = _build_metadata.flash_version.lstrip("v")
    FLASH3_HAS_DETERMINISTIC_MODE = True
    _C_flashattention3 = torch.ops.flash_attn_3

elif importlib.util.find_spec("flash_attn_3") and importlib.util.find_spec(
    "flash_attn_3._C"
):
    import flash_attn_3._C  # type: ignore[attr-defined]  # noqa: F401

    incompat_reason = _flash_attention3_incompatible_reason()
    if incompat_reason is None:
        _C_flashattention3 = torch.ops.flash_attn_3
        FLASH_VERSION = "pip_pkg"
        FLASH3_HAS_PAGED_ATTENTION = True
        FLASH3_HAS_FLOAT8 = True
    else:
        logger.warning(f"Flash-Attention 3 package can't be used: {incompat_reason}")


def _heuristic_kvsplit(
    inp: Inputs,
    enable_kvsplit_attn: bool,
) -> bool:
    atten_bias = inp.attn_bias

    # make sure Q doesn't have varlen
    # pyre-ignore Undefined attribute [16]
    if atten_bias.q_seqinfo.min_seqlen != atten_bias.q_seqinfo.max_seqlen:  # type: ignore[union-attr]
        return False

    # filter out prefill case
    # pyre-ignore Undefined attribute [16]
    if atten_bias.q_seqinfo.max_seqlen == atten_bias.k_seqinfo.max_seqlen:  # type: ignore[union-attr]
        return False

    return enable_kvsplit_attn


def mask_non_zeros(s_q: int, s_k: int, window_left: int, window_right: int) -> int:
    # Exact formula for easy cases
    if window_left < 0 and window_right < 0:  # full
        return s_q * s_k
    if window_left < 0 and window_right == 0:  # causal
        # (from bottom right)
        return (s_q * (s_q + 1)) // 2 + s_q * max(0, s_k - s_q)

    # NOTE: Flops calculations here assume `s_q == s_k`
    # otherwise the local attention computations are too involved
    # See also https://docs.google.com/spreadsheets/d/1u1ItCZcHLArcqXLj7mwR4H1pI3lMKU1zlxCYi8JCYgk/edit?usp=sharing
    if window_left < 0:
        window_left = s_k
    if window_right < 0:
        window_right = s_k

    # below the diagonal
    # ┌───────┐
    # │ ╲     │
    # │  ╲    │ <- Upper triangle ("ut")
    # │┄┄┄╲   │ <--- `lastq_ut`
    # │╲   ╲  │
    # │ ╲   ╲ │ <- Lower part
    # │  ╲   ╲│
    # └───────┘
    mask_nz = min(s_q, s_k)  # diagonal
    # Below diagonal (with `window_left`)
    lastq_ut = min(window_left, s_q)
    mask_nz += ((lastq_ut - 1) * lastq_ut) // 2  # upper triangle
    mask_nz += (s_q - lastq_ut) * window_left  # lower part
    # Above diagonal (with `window_right`)
    # (counting rows from the bottom for symmetry)
    firstq_bt = min(window_right + 1, s_q)
    mask_nz += ((firstq_bt - 1) * firstq_bt) // 2  # bottom triangle
    mask_nz += (s_q - firstq_bt) * window_right

    return mask_nz


# Copied from PyTorch, modified to support MQA/GQA and local attention
# No need to take care of this for the bwd because we don't "unexpand" the keys
# and values (in the fwd we expand to help with the seqlen/headdim swap trick).
def sdpa_flop_count(
    query_shape, key_shape, value_shape, window_left: int, window_right: int
):
    """
    Count flops for self-attention.

    NB: We can assume that value_shape == key_shape
    """
    b, h_q, s_q, d_q = query_shape
    _b2, h_kv, s_k, _d2 = key_shape
    _b3, _h2, _s3, d_v = value_shape
    assert b == _b2 == _b3
    assert h_kv == _h2
    assert d_q == _d2
    assert s_k == _s3
    assert d_q == _d2
    assert h_q % h_kv == 0
    # How many values are computed in the attention?
    mask_nz = mask_non_zeros(s_q, s_k, window_left, window_right)

    # q@k.T
    total_flops = 2 * b * h_q * d_q * mask_nz
    # attn@v
    total_flops += 2 * b * h_q * d_v * mask_nz
    return total_flops


if _C_flashattention3 is not None:
    # returns: out, q_padded, k_padded, v_padded, out_padded, softmax_lse, p
    @torch.library.custom_op(
        "xformers_flash3::flash_fwd", mutates_args=(), device_types=["cuda"]
    )
    def mha_fwd(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor],
        cu_seqlens_k: Optional[torch.Tensor],
        seqused_k: Optional[torch.Tensor],
        leftpad_k: Optional[torch.Tensor],
        max_seqlen_q: int,
        max_seqlen_k: int,
        p: float,
        softmax_scale: float,
        is_causal: bool,
        descale_q: Optional[torch.Tensor] = None,
        descale_k: Optional[torch.Tensor] = None,
        descale_v: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        use_kvsplit: bool = False,
        window_left: int = -1,
        window_right: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query, key = [maybe_contiguous(x) for x in (query, key)]
        value = (
            value.contiguous()
            if value.stride(-1) != 1 and value.stride(-3) != 1
            else value
        )
        cu_seqlens_q, cu_seqlens_k, seqused_k = [
            maybe_contiguous(x) for x in (cu_seqlens_q, cu_seqlens_k, seqused_k)
        ]
        block_table = maybe_contiguous(block_table)

        def _get_batch():
            if cu_seqlens_q is not None:
                return cu_seqlens_q.shape[0] - 1
            return query.shape[0]

        is_paged = block_table is not None
        bs = _get_batch()
        orig_query_shape = query.shape

        pack_gqa = None
        if use_kvsplit:
            # For KV split, we need to make sure query in shape [batch, seqlen, num_heads, head_dim_q]
            # to be compatible with `pack_gqa` feature
            query = query.view(bs, -1, query.shape[-2], query.shape[-1])
            cu_seqlens_q = None

            # Auto-detect if we should use GQA parallel mode
            if query.shape[1] <= 64 and query.shape[2] != key.shape[2]:
                pack_gqa = True

        assert _C_flashattention3 is not None
        out, softmax_lse, *rest = _C_flashattention3.fwd(
            query,
            key,
            value,
            None,
            None,  # k_new, v_new
            None,  # qv
            None,  # out
            cu_seqlens_q,
            cu_seqlens_k if not is_paged else None,
            None,  # cu_seqlens_k_new
            None,  # seqused_q
            seqused_k,
            max_seqlen_q,
            max_seqlen_k,
            block_table,
            None,  # kv_batch_idx
            leftpad_k,
            None,  # rotary_cos
            None,  # rotary_sin
            None,  # seqlens_rotary
            descale_q,
            descale_k,
            descale_v,
            softmax_scale,
            is_causal,
            window_left,
            window_right,
            0,  # attention_chunk
            0.0,  # softcap
            not use_kvsplit,  # rotary_interleaved
            None,  # scheduler_metadata
            1 if not use_kvsplit else 0,  # num_splits
            pack_gqa,  # pack_gqa
            0,  # sm_margin
        )

        if query.shape != orig_query_shape:
            # Reshape softmax_lse to match expected output format
            num_heads_q = query.shape[-2]
            orig_lse_shape = softmax_lse.shape
            softmax_lse = softmax_lse.view(
                orig_lse_shape[0], num_heads_q, -1, orig_lse_shape[2]
            )
            softmax_lse = softmax_lse.permute(1, 0, 2, 3).reshape(num_heads_q, -1)

        return out, softmax_lse

    @torch.library.register_fake("xformers_flash3::flash_fwd")
    def mha_fwd_fake(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor],
        cu_seqlens_k: Optional[torch.Tensor],
        seqused_k: Optional[torch.Tensor],
        leftpad_k: Optional[torch.Tensor],
        max_seqlen_q: int,
        max_seqlen_k: int,
        p: float,
        softmax_scale: float,
        is_causal: bool,
        descale_q: Optional[torch.Tensor] = None,
        descale_k: Optional[torch.Tensor] = None,
        descale_v: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        use_kvsplit: bool = False,
        window_left: int = -1,
        window_right: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query_shape = query.shape
        out_shape = (*query_shape[:-1], value.shape[-1])
        if query.dtype == torch.float8_e4m3fn or query.dtype == torch.float8_e5m2:
            out = query.new_empty(out_shape, dtype=torch.bfloat16)
        else:
            out = query.new_empty(out_shape)
        # Query is (B, M, H, K) or (total_M, H, K)
        # LSE is (B, H, M) or (H, total_M)
        lse_shape = (
            (query_shape[0], query_shape[2], query_shape[1])
            if cu_seqlens_q is None
            else (query_shape[1], query_shape[0])
        )
        lse = query.new_empty(lse_shape, dtype=torch.float32)
        return out, lse

    @register_flop_formula(torch.ops.xformers_flash3.flash_fwd, get_raw=True)
    def mha_fwd_flops(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor],
        cu_seqlens_k: Optional[torch.Tensor],
        seqused_k: Optional[torch.Tensor],
        leftpad_k: Optional[torch.Tensor],
        max_seqlen_q: int,
        max_seqlen_k: int,
        p: float,
        softmax_scale: float,
        is_causal: bool,
        descale_q: Optional[torch.Tensor] = None,
        descale_k: Optional[torch.Tensor] = None,
        descale_v: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        use_kvsplit: bool = False,
        window_left: int = -1,
        window_right: int = -1,
        # The FLOPs counter might pass more args (out_val, out_shape, ...)
        *args,
        **kwargs,
    ):
        assert 3 <= query.ndim <= 4
        assert 3 <= key.ndim <= 4
        assert 3 <= value.ndim <= 4
        # This FLOP formula is used by torch.compile's partitioner "automatic
        # activation checkpointing" (AutoAC) to decide which ops to preserve
        # for backward or to recompute. However, this formula is data-dependent!
        # This makes all invocations reuse the choices made based on the first
        # inputs, which may be sub-optimal but also lead to inconsistent
        # behavior across runs. In the presence of tensor parallelism it might
        # also lead to deadlocks if AutoAC recomputes different collectives
        # on different ranks. For distributed jobs it seems more robust to have
        # all ranks always use the "worst case" FLOP estimate. Ranks are in
        # lockstep anyways and will be going as fast as the slowest one.
        if os.environ.get("XFORMERS_FLOP_FORMULA_WORST_CASE", "0") == "1":
            cu_seqlens_q = cu_seqlens_k = max_seqlen_q = max_seqlen_k = None  # type: ignore[assignment]
            query = query.unsqueeze(0) if query.ndim == 3 else query
            key = key.unsqueeze(0) if key.ndim == 3 else key
            value = value.unsqueeze(0) if value.ndim == 3 else value
        sizes = _unpack_flash_attention_nested_shapes(
            query=query.transpose(-2, -3) if query.ndim == 4 else query,
            key=key.transpose(-2, -3) if key.ndim == 4 else key,
            value=value.transpose(-2, -3) if value.ndim == 4 else value,
            cum_seq_q=cu_seqlens_q,
            cum_seq_k=cu_seqlens_k,
            max_q=max_seqlen_q,
            max_k=max_seqlen_k,
        )
        if is_causal:
            window_right = 0
        res = sum(
            sdpa_flop_count(
                query_shape,
                key_shape,
                value_shape,
                window_left=window_left,
                window_right=window_right,
            )
            for query_shape, key_shape, value_shape, _ in sizes
        )
        return res

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
        window_left: int,
        window_right: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dq, dk, dv = _create_dq_dk_dv(grads_share_storage, query, key, value)
        is_deterministic = (
            torch.are_deterministic_algorithms_enabled()
            and FLASH3_HAS_DETERMINISTIC_MODE
        )
        if cu_seqlens_q is None:
            assert cu_seqlens_k is None

        assert _C_flashattention3 is not None
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
            cu_seqlens_q,
            cu_seqlens_k,
            None,  # seqused_q
            None,  # seqused_k
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale,
            is_causal,
            window_left,
            window_right,
            0.0,  # not used, softcap
            is_deterministic,
            0,  # not used, sm_margin
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
        window_left: int,
        window_right: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _create_dq_dk_dv(grads_share_storage, query, key, value)

    @register_flop_formula(torch.ops.xformers_flash3.flash_bwd, get_raw=True)
    def mha_bwd_flops(
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
        window_left: int,
        window_right: int,
        # The FLOPs counter might pass more args (out_val, out_shape, ...)
        *args,
        **kwargs,
    ):
        return (
            5
            * mha_fwd_flops(
                query,
                key,
                value,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                seqused_k=None,
                leftpad_k=None,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                p=0.0,
                softmax_scale=1.0,
                is_causal=is_causal,
                descale_q=None,
                descale_k=None,
                descale_v=None,
                block_table=None,
                use_kvsplit=False,
                window_left=window_left,
                window_right=window_right,
            )
            // 2
        )


def _check_different_value_headdim_ampere(d: Inputs, reasons: List[str]) -> None:
    if (
        d.query.device.type == "cuda"
        and (torch.version.hip is None)
        and d.query.shape[-1] != d.value.shape[-1]
    ):
        device_capability = torch.cuda.get_device_capability(d.device)
        if device_capability < (9, 0):
            reasons.append(
                f"Q/K head-dim ({d.query.shape[-1]}) must be equal "
                f"to V head-dim ({d.value.shape[-1]}) for Ampere GPUs"
            )


def _get_blocktables(inp_attn_bias) -> Optional[torch.Tensor]:
    return (
        inp_attn_bias.block_tables
        if isinstance(
            inp_attn_bias,
            (PagedBlockDiagonalGappyKeysMask, PagedBlockDiagonalPaddedKeysMask),
        )
        else None
    )


@register_operator
class FwOp(AttentionFwOpBase):
    """Operator that computes memory-efficient attention using \
        `Flash-Attention <https://github.com/HazyResearch/flash-attention>`_ \
        implementation.
    """

    OPERATOR = get_operator("xformers_flash3", "flash_fwd")
    SUPPORTED_DEVICES: Set[str] = {"cuda"}
    CUDA_MINIMUM_COMPUTE_CAPABILITY = (8, 0)
    CUDA_MAXIMUM_COMPUTE_CAPABILITY = (9, 0)
    SUPPORTED_DTYPES: Set[torch.dtype] = {
        torch.half,
        torch.bfloat16,
    } | ({torch.float8_e4m3fn} if FLASH3_HAS_FLOAT8 else set())
    SUPPORTED_MAX_K = 256
    SUPPORTED_MIN_K = 32
    SUPPORTED_ATTN_BIAS_TYPES: Iterable[Any] = (
        type(None),
        LowerTriangularMask,
        LowerTriangularFromBottomRightMask,
        LowerTriangularFromBottomRightLocalAttentionMask,
        BlockDiagonalMask,
        BlockDiagonalCausalMask,
        BlockDiagonalCausalLocalAttentionMask,
        BlockDiagonalCausalLocalAttentionFromBottomRightMask,
        BlockDiagonalCausalLocalAttentionPaddedKeysMask,
        BlockDiagonalCausalFromBottomRightMask,
        BlockDiagonalCausalWithOffsetGappyKeysMask,
        BlockDiagonalCausalWithOffsetPaddedKeysMask,
        BlockDiagonalLocalAttentionPaddedKeysMask,
        BlockDiagonalGappyKeysMask,
        BlockDiagonalPaddedKeysMask,
        LocalAttentionFromBottomRightMask,
    ) + (
        (
            PagedBlockDiagonalCausalWithOffsetGappyKeysMask,
            PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
            PagedBlockDiagonalGappyKeysMask,
            PagedBlockDiagonalPaddedKeysMask,
        )
        if FLASH3_HAS_PAGED_ATTENTION
        else tuple()
    )

    SUPPORTS_DROPOUT = False
    SUPPORTS_CUSTOM_SCALE = True
    SUPPORTS_DIFFERENT_VALUE_EMBED = True  # Only hopper
    SUPPORTS_BMGHK = True
    SUPPORTS_PARTIAL = True
    UNPADDED_LSE = True
    NAME = f"fa3F@{FLASH_VERSION}"
    VERSION = FLASH_VERSION

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:
        reasons = super(FwOp, cls).not_supported_reasons(d)
        device_type = d.query.device.type
        if device_type == "cuda" and (torch.version.hip is None):
            device_capability = torch.cuda.get_device_capability(d.device)
            if device_capability > cls.CUDA_MINIMUM_COMPUTE_CAPABILITY:
                reasons.append(
                    f"requires device with capability == {cls.CUDA_MINIMUM_COMPUTE_CAPABILITY} "
                    f"but your GPU has capability {device_capability} (too new)"
                )
        check_lastdim_alignment_stride1(reasons, "query", d.query, 8)
        check_lastdim_alignment_stride1(reasons, "key", d.value, 8)
        check_lastdim_alignment_stride1(reasons, "value", d.value, 8)
        _check_needs_no_topleft(d, reasons)
        _check_different_value_headdim_ampere(d, reasons)
        if (
            _get_blocktables(d.attn_bias) is not None
            and d.query.shape[-1] != d.value.shape[-1]
        ):
            reasons.append(
                f"Q/K head-dim ({d.query.shape[-1]}) must be equal "
                f"to V head-dim ({d.value.shape[-1]}) for paged attention"
            )
        return reasons

    @classmethod
    def apply(
        cls,
        inp: Inputs,
        needs_gradient: bool,
        use_kvsplit: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Context]]:
        original_query_shape = inp.query.shape
        out_shape = [
            *inp.query.shape[:-1],
            inp.value.shape[-1],
        ]

        def unpack_func(x) -> Tuple[torch.Tensor, Any]:
            return x.unpack() if isinstance(x, ScaledTensor) else (x, None)

        inp.query, descale_q = unpack_func(inp.query)
        inp.key, descale_k = unpack_func(inp.key)
        inp.value, descale_v = unpack_func(inp.value)
        (
            inp,
            cu_seqlens_q,
            max_seqlen_q,
            cu_seqlens_k,
            max_seqlen_k,
            seqused_k,
        ) = _convert_input_format(inp, supports_mqa=True, use_kvsplit=use_kvsplit)

        q = inp.query
        k = inp.key
        v = inp.value

        if inp.query.numel() > 0 and inp.key.numel() > 0:
            win_left, win_right = _window_size(inp.attn_bias)
            block_tables = _get_blocktables(inp.attn_bias)
            leftpad_k = None
            if isinstance(inp.attn_bias, PagedBlockDiagonalGappyKeysMask):
                assert cu_seqlens_q is not None
                assert cu_seqlens_k is not None
                if len(cu_seqlens_q) == len(cu_seqlens_k):
                    # case #1: len(cu_seqlens_k) = batch_size + 1
                    leftpad_k = cu_seqlens_k[:-1]
                else:
                    # case #2: len(cu_seqlens_k) = batch_size
                    assert (
                        len(cu_seqlens_q) - len(cu_seqlens_k) == 1
                    ), f"{len(cu_seqlens_q)=} {len(cu_seqlens_k)=}"
                    leftpad_k = cu_seqlens_k
            out, softmax_lse = cls.OPERATOR(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                seqused_k=seqused_k,
                leftpad_k=leftpad_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                p=inp.p,
                softmax_scale=inp.scale_float,
                is_causal=_is_causal(inp.attn_bias),
                descale_q=descale_q,
                descale_k=descale_k,
                descale_v=descale_v,
                block_table=block_tables,
                use_kvsplit=use_kvsplit,
                window_left=win_left,
                window_right=win_right,
            )
            out = out.reshape(out_shape)
        else:
            out = torch.zeros(
                inp.query.shape, device=inp.query.device, dtype=inp.query.dtype
            )
            if inp.is_partial:
                softmax_lse = torch.full(
                    [inp.query.shape[0], inp.query.shape[2], inp.query.shape[1]],
                    float("-inf"),
                    device=inp.query.device,
                    dtype=torch.float32,
                )
            else:
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


@register_operator
class BwOp(AttentionBwOpBase):
    __doc__ = FwOp.__doc__

    OPERATOR = get_operator("xformers_flash3", "flash_bwd")
    SUPPORTED_DEVICES = FwOp.SUPPORTED_DEVICES
    CUDA_MINIMUM_COMPUTE_CAPABILITY = FwOp.CUDA_MINIMUM_COMPUTE_CAPABILITY
    CUDA_MAXIMUM_COMPUTE_CAPABILITY = FwOp.CUDA_MAXIMUM_COMPUTE_CAPABILITY
    SUPPORTED_DTYPES = FwOp.SUPPORTED_DTYPES
    SUPPORTED_MAX_K = FwOp.SUPPORTED_MAX_K
    SUPPORTED_MIN_K = 64
    SUPPORTED_ATTN_BIAS_TYPES = (
        # Exclude padded or gappy masks, since seqused_k is not supported by the kernel.
        type(None),
        LowerTriangularMask,
        LowerTriangularFromBottomRightMask,
        LowerTriangularFromBottomRightLocalAttentionMask,
        BlockDiagonalMask,
        BlockDiagonalCausalMask,
        BlockDiagonalCausalLocalAttentionMask,
        BlockDiagonalCausalLocalAttentionFromBottomRightMask,
        BlockDiagonalCausalFromBottomRightMask,
        LocalAttentionFromBottomRightMask,
    )

    SUPPORTS_DROPOUT = FwOp.SUPPORTS_DROPOUT
    SUPPORTS_CUSTOM_SCALE = FwOp.SUPPORTS_CUSTOM_SCALE
    SUPPORTS_DIFFERENT_VALUE_EMBED = FwOp.SUPPORTS_DIFFERENT_VALUE_EMBED
    IS_DETERMINISTIC = FLASH3_HAS_DETERMINISTIC_MODE
    SUPPORTS_BMGHK = False
    SUPPORTS_LSE_FORMATS: Sequence[str] = ["", "varlen_flat"]
    NAME = f"fa3B@{FLASH_VERSION}"
    VERSION = FLASH_VERSION

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:
        reasons = super(BwOp, cls).not_supported_reasons(d)
        device_type = d.query.device.type
        if device_type == "cuda" and (torch.version.hip is None):
            device_capability = torch.cuda.get_device_capability(d.device)
            if device_capability > cls.CUDA_MINIMUM_COMPUTE_CAPABILITY:
                reasons.append(
                    f"requires device with capability == {cls.CUDA_MINIMUM_COMPUTE_CAPABILITY} "
                    f"but your GPU has capability {device_capability} (too new)"
                )
        check_lastdim_alignment_stride1(reasons, "query", d.query, 8)
        check_lastdim_alignment_stride1(reasons, "key", d.value, 8)
        check_lastdim_alignment_stride1(reasons, "value", d.value, 8)
        _check_needs_no_topleft(d, reasons)
        _check_different_value_headdim_ampere(d, reasons)
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
            win_left, win_right = _window_size(inp.attn_bias)
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
                window_left=win_left,
                window_right=win_right,
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


@register_operator
class FwOp_KVSplit(FwOp):
    """Operator that computes memory-efficient attention using \
        `Flash-Attention3 <https://github.com/Dao-AILab/flash-attention/tree/main/hopper>`_ \
        implementation with heuristic rules to dispatch decoding shapes to KVSplit Attention \
    """

    NAME = f"fa3F_splitKV@{FLASH_VERSION}"
    enable_kvsplit_attn: bool = True

    SUPPORTED_ATTN_BIAS_TYPES: Iterable[Any] = (
        type(None),
        BlockDiagonalCausalWithOffsetPaddedKeysMask,
        BlockDiagonalPaddedKeysMask,
        BlockDiagonalCausalWithOffsetGappyKeysMask,
        BlockDiagonalGappyKeysMask,
        BlockDiagonalLocalAttentionPaddedKeysMask,
        BlockDiagonalCausalLocalAttentionPaddedKeysMask,
    ) + (
        (
            PagedBlockDiagonalCausalWithOffsetGappyKeysMask,
            PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
            PagedBlockDiagonalGappyKeysMask,
            PagedBlockDiagonalPaddedKeysMask,
        )
        if FLASH3_HAS_PAGED_ATTENTION
        else tuple()
    )

    @classmethod
    def apply(  # type: ignore[override]
        cls,
        inp: Inputs,
        needs_gradient: bool,
    ) -> Tuple[torch.Tensor, Optional[Context]]:
        use_kvsplit = _heuristic_kvsplit(inp, cls.enable_kvsplit_attn)

        return super().apply(inp, needs_gradient, use_kvsplit)
