# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import importlib.util
import os
from typing import Any, Iterable, List, Optional, Sequence, Set, Tuple

import torch
from torch.utils.flop_counter import (
    _flash_attention_backward_flop,
    _unpack_flash_attention_nested_shapes,
    bmm_flop,
    register_flop_formula,
)

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
    PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
    PagedBlockDiagonalPaddedKeysMask,
)
from .common import (
    AttentionBwOpBase,
    AttentionFwOpBase,
    Context,
    Gradients,
    Inputs,
    ScaledTensor,
    check_lastdim_alignment_stride1,
)
from .flash import (
    _check_needs_no_topleft,
    _convert_input_format,
    _is_causal,
    _post_process_lse,
)

FLASH_VERSION = "0.0.0"


if importlib.util.find_spec("..._C_flashattention3", package=__package__):
    from ... import _C_flashattention3  # type: ignore[attr-defined]
    from ..._cpp_lib import _build_metadata

    if _build_metadata is not None:
        FLASH_VERSION = _build_metadata.flash_version.lstrip("v")

elif importlib.util.find_spec("flash_attn_interface"):
    from flash_attn_interface import flashattn_hopper_cuda as _C_flashattention3

else:
    # We end up here is arch is not 90a
    _C_flashattention3 = None


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def supported_dtypes() -> Set[torch.dtype]:
    types = {
        torch.half,
        torch.bfloat16,
    }
    if os.environ.get("XFORMERS_FLASH3_FP8", "0") == "1":
        types.add(torch.float8_e4m3fn)
    return types


def _paged_attention_filter(attn_bias_types: Iterable[Any]) -> Iterable[Any]:
    if os.environ.get("XFORMERS_FLASH3_PAGED", "0") == "1":
        return attn_bias_types
    return [
        x
        for x in attn_bias_types
        if not issubclass(x, PagedBlockDiagonalPaddedKeysMask)
    ]


# Copied from PyTorch, modified to support MQA/GQA.
# No need to take care of this for the bwd because we don't "unexpand" the keys
# and values (in the fwd we expand to help with the seqlen/headdim swap trick).
def sdpa_flop_count(query_shape, key_shape, value_shape):
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
    total_flops = 0
    # q: [b, h, s_q, d_q] @ k: [b, h, d_q, s_k] -> scores: [b, h, s_q, s_k]
    total_flops += bmm_flop((b * h_q, s_q, d_q), (b * h_q, d_q, s_k))
    # scores: [b, h, s_q, s_k] @ v: [b, h, s_k, d_v] -> out: [b, h, s_q, d_v]
    total_flops += bmm_flop((b * h_q, s_q, s_k), (b * h_q, s_k, d_v))
    return total_flops


if _C_flashattention3 is not None:

    # Compatibility check for FAv3 APIs
    EXPECTED_NUM_OF_ARGS = [
        ("fwd", 31),
        ("bwd", 23),
    ]

    import re

    def count_args_from_doc(docstring) -> int:
        # Use a regular expression to find the argument list inside parentheses
        match = re.search(r"\((.*?)\)", docstring)
        if match:
            # Extract the argument list and split by commas
            args_list = match.group(1).split(",")
            # Count the number of arguments
            return len(args_list)
        else:
            raise ValueError("No valid argument list found in the docstring.")

    for name, num_of_args in EXPECTED_NUM_OF_ARGS:
        num_of_args_from_doc = count_args_from_doc(
            getattr(_C_flashattention3, name).__doc__
        )
        assert num_of_args_from_doc == num_of_args, (
            f"Found func signature mismatch for {name}. Expected {num_of_args},"
            f"actual: {num_of_args_from_doc} Please update the version of Flash Attention3."
        )

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        win_left = win_right = -1
        query, key = [maybe_contiguous(x) for x in (query, key)]
        if value.stride(-3) != 1:
            # For FP8 it is ok to have stride(-3)==1 instead of stride(-1)
            value = maybe_contiguous(value)
        cu_seqlens_q, cu_seqlens_k, seqused_k = [
            maybe_contiguous(x) for x in (cu_seqlens_q, cu_seqlens_k, seqused_k)
        ]
        block_table = maybe_contiguous(block_table)

        if cu_seqlens_q is None:
            # Fixed-length case
            assert cu_seqlens_k is None
            assert seqused_k is None
            assert (
                block_table is None
            ), "Block table is not supported for fixed-length query yet"

            out, softmax_lse, *rest = _C_flashattention3.fwd(
                query,
                key,
                value,
                None,  # k_new
                None,  # v_new
                None,  # out
                None,  # cu_seqlens_q
                None,  # cu_seqlens_k
                None,  # cu_seqlens_k_new
                None,  # seqused_q
                None,  # seqused_k
                None,  # max_seqlen_q
                None,  # max_seqlen_k
                None,  # page_table
                None,  # kv_batch_idx
                None,  # leftpad_k
                None,  # rotary_cos
                None,  # rotary_sin
                descale_q,
                descale_k,
                descale_v,
                softmax_scale,
                is_causal,
                win_left,
                win_right,
                0,  # sink_token_length
                0.0,  # softcap
                False,  # rotary_interleaved
                1,  # num_splits (not KVSplit Case)
                False,  # pack_gqa
                0,  # sm_margin
            )
            return out, softmax_lse

        else:
            assert (
                descale_q is None and descale_k is None and descale_v is None
            ), "FP8 attention does not yet support variable-length inputs during the forward pass"

            if use_kvsplit:
                # Split KV case
                # Auto-detect if we should use GQA parallel mode
                pack_gqa = False
                if query.shape[1] <= 64 and query.shape[2] != key.shape[2]:
                    pack_gqa = True

                out, softmax_lse, *rest = _C_flashattention3.fwd(
                    query,
                    key,
                    value,
                    None,  # k_new
                    None,  # v_new
                    None,  # out
                    None,  # cu_seqlens_q,
                    cu_seqlens_k,
                    None,  # cu_seqlens_k_new
                    None,  # seqused_q
                    seqused_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    block_table,  # page_table
                    None,  # kv_batch_idx
                    None,  # leftpad_k
                    None,  # rotary_cos
                    None,  # rotary_sin
                    descale_q,
                    descale_k,
                    descale_v,
                    softmax_scale,
                    is_causal,
                    -1,
                    -1,  # window_size_left/right
                    0,  # sink_token_length
                    0.0,  # softcap
                    False,  # rotary_interleaved
                    0,  # num_splits
                    pack_gqa,
                    0,  # sm_margin
                )

                # Reshape softmax_lse to match expected output format
                num_heads_q = query.shape[-2]
                ori_lse_shape = softmax_lse.shape
                softmax_lse = softmax_lse.view(
                    ori_lse_shape[0], num_heads_q, -1, ori_lse_shape[2]
                )
                softmax_lse = softmax_lse.permute(1, 0, 2, 3).reshape(num_heads_q, -1)

                return out, softmax_lse

            else:
                # Variable length case
                out, softmax_lse, *rest = _C_flashattention3.fwd(
                    query,
                    key,
                    value,
                    None,  # k_new
                    None,  # v_new
                    None,  # out
                    cu_seqlens_q,
                    cu_seqlens_k if block_table is None else None,
                    None,  # cu_seqlens_k_new
                    None,  # seqused_q
                    seqused_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    block_table,  # page_table
                    None,  # kv_batch_idx
                    None,  # leftpad_k
                    None,  # rotary_cos
                    None,  # rotary_sin
                    descale_q,
                    descale_k,
                    descale_v,
                    softmax_scale,
                    is_causal,
                    -1,
                    -1,  # window_size_left/right
                    0,  # sink_token_length
                    0.0,  # softcap
                    True,  # rotary_interleaved
                    1,  # num_splits
                    None,  # pack_gqa
                    0,  # sm_margin
                )

                return out, softmax_lse

    @torch.library.register_fake("xformers_flash3::flash_fwd")
    def mha_fwd_fake(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor],
        cu_seqlens_k: Optional[torch.Tensor],
        seqused_k: Optional[torch.Tensor],
        max_seqlen_q: int,
        max_seqlen_k: int,
        p: float,
        softmax_scale: float,
        is_causal: bool,
        descale_q: Optional[torch.Tensor] = None,
        descale_k: Optional[torch.Tensor] = None,
        descale_v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

    @register_flop_formula(torch.ops.xformers_flash3.flash_fwd, get_raw=True)
    def mha_fwd_flops(
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
        res = sum(
            sdpa_flop_count(query_shape, key_shape, value_shape)
            for query_shape, key_shape, value_shape, _ in sizes
        )
        if is_causal:
            res //= 2
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        win_left = win_right = -1
        seqused_q = seqused_k = None
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
                None,  # cu_seqlens_q
                None,  # cu_seqlens_k
                seqused_q,
                seqused_k,
                None,  # max_seqlen_q
                None,  # max_seqlen_k
                softmax_scale,
                is_causal,
                win_left,
                win_right,
                0,  # not used, sink_token_length
                0.0,  # not used, softcap
                is_deterministic,
                0,  # not used, sm_margin
            )
        else:
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
                seqused_q,
                seqused_k,
                max_seqlen_q,
                max_seqlen_k,
                softmax_scale,
                is_causal,
                win_left,
                win_right,
                0,  # not used, sink_token_length
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dq = torch.empty_like(query)
        dk = torch.empty_like(key)
        dv = torch.empty_like(value)
        return dq, dk, dv

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
        # The FLOPs counter might pass more args (out_val, out_shape, ...)
        *args,
        **kwargs,
    ):
        assert 3 <= dout.ndim <= 4
        assert 3 <= query.ndim <= 4
        assert 3 <= key.ndim <= 4
        assert 3 <= value.ndim <= 4
        # See the fwd FLOP formula above for reasoning behind this.
        if os.environ.get("XFORMERS_FLOP_FORMULA_WORST_CASE", "0") == "1":
            cu_seqlens_q = cu_seqlens_k = max_seqlen_q = max_seqlen_k = None  # type: ignore[assignment]
            dout = dout.unsqueeze(0) if dout.ndim == 3 else dout
            query = query.unsqueeze(0) if query.ndim == 3 else query
            key = key.unsqueeze(0) if key.ndim == 3 else key
            value = value.unsqueeze(0) if value.ndim == 3 else value
        res = _flash_attention_backward_flop(
            dout.transpose(-2, -3) if dout.ndim == 4 else dout,
            query.transpose(-2, -3) if query.ndim == 4 else query,
            key.transpose(-2, -3) if key.ndim == 4 else key,
            value.transpose(-2, -3) if value.ndim == 4 else value,
            out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
        )
        if is_causal:
            res //= 2
        return res


@register_operator
class FwOp(AttentionFwOpBase):
    """Operator that computes memory-efficient attention using \
        `Flash-Attention <https://github.com/HazyResearch/flash-attention>`_ \
        implementation.
    """

    OPERATOR = get_operator("xformers_flash3", "flash_fwd")
    SUPPORTED_DEVICES: Set[str] = {"cuda"}
    CUDA_MINIMUM_COMPUTE_CAPABILITY = (9, 0)
    SUPPORTED_DTYPES: Set[torch.dtype] = supported_dtypes()
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
        PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
        PagedBlockDiagonalPaddedKeysMask,
    )

    SUPPORTED_ATTN_BIAS_TYPES = _paged_attention_filter(SUPPORTED_ATTN_BIAS_TYPES)

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
        if d.query.shape[-1] not in [64, 128, 192, 256]:
            reasons.append("only head-dim 64, 128, 192 or 256 is supported")

        _check_needs_no_topleft(d, reasons)

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
        (
            inp,
            cu_seqlens_q,
            max_seqlen_q,
            cu_seqlens_k,
            max_seqlen_k,
            seqused_k,
        ) = _convert_input_format(inp, supports_mqa=True, use_kvsplit=use_kvsplit)

        def unpack_func(x):
            return x.unpack() if isinstance(x, ScaledTensor) else (x, None)

        q, descale_q = unpack_func(inp.query)
        k, descale_k = unpack_func(inp.key)
        v, descale_v = unpack_func(inp.value)

        if inp.query.numel() > 0 and inp.key.numel() > 0:
            block_tables = (
                inp.attn_bias.block_tables
                if isinstance(inp.attn_bias, PagedBlockDiagonalPaddedKeysMask)
                else None
            )
            (out, softmax_lse,) = cls.OPERATOR(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_k,
                max_seqlen_q,
                max_seqlen_k,
                inp.p,
                inp.scale_float,
                _is_causal(inp.attn_bias),
                descale_q,
                descale_k,
                descale_v,
                block_tables,
                use_kvsplit=use_kvsplit,
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
        if d.query.shape[-1] not in [64, 128, 192, 256]:
            reasons.append("only head-dim 64, 128, 192 or 256 is supported")

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


@register_operator
class FwOp_KVSplit(FwOp):
    """Operator that computes memory-efficient attention using \
        `Flash-Attention3 <https://github.com/Dao-AILab/flash-attention/tree/main/hopper>`_ \
        implementation with heuristic rules to dispatch decoding shapes to KVSplit Attention \
    """

    enable_kvsplit_attn: bool = True

    SUPPORTED_ATTN_BIAS_TYPES: Iterable[Any] = (
        BlockDiagonalCausalWithOffsetPaddedKeysMask,
        BlockDiagonalPaddedKeysMask,
    )

    @classmethod
    def apply(
        cls,
        inp: Inputs,
        needs_gradient: bool,
        use_kvsplit: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Context]]:
        attn_bias = inp.attn_bias
        assert isinstance(attn_bias, BlockDiagonalPaddedKeysMask)
        homogeneous_q = attn_bias.q_seqinfo.min_seqlen == attn_bias.q_seqinfo.max_seqlen
        short_q = attn_bias.q_seqinfo.max_seqlen <= 10

        # Note that prefill shouldn't use kvsplit.
        use_kvsplit = (
            use_kvsplit and homogeneous_q and cls.enable_kvsplit_attn and short_q
        )

        return super().apply(inp, needs_gradient, use_kvsplit)
