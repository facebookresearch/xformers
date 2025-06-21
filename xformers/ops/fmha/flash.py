# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import importlib.util
import os
from itertools import zip_longest
from typing import Any, Iterable, List, Optional, Set, Tuple, Union

import torch

from ..common import get_operator, register_operator
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
)
from .torch_attention_compat import is_pt_flash_old

FLASH_VERSION = "0.0.0"
VARLEN_LSE_PACKED = False
pt_flash_is_old = False
_TRY_PT_FLASH_ATTN = (
    torch.version.hip is None and torch.backends.cuda.is_flash_attention_available()
)
_USE_PT_FLASH_ATTN = False


if importlib.util.find_spec("..._C_flashattention", package=__package__):
    from ... import _C_flashattention  # type: ignore[attr-defined]
    from ..._cpp_lib import _build_metadata

    if _build_metadata is not None:
        FLASH_VERSION = _build_metadata.flash_version.lstrip("v")
    VARLEN_LSE_PACKED = True

elif importlib.util.find_spec("flash_attn"):
    import flash_attn
    import flash_attn.flash_attn_interface

    if hasattr(flash_attn.flash_attn_interface, "flash_attn_cuda"):
        _C_flashattention = flash_attn.flash_attn_interface.flash_attn_cuda
    else:
        _C_flashattention = flash_attn.flash_attn_interface.flash_attn_gpu

    FLASH_VERSION = flash_attn.__version__
    FLASH_VER_MIN = (2, 7, 1)
    FLASH_VER_LAST = (2, 8, 0)  # last supported, inclusive
    flash_ver_parsed = tuple(int(s) for s in FLASH_VERSION.split(".")[:3])
    if (
        flash_ver_parsed < FLASH_VER_MIN or flash_ver_parsed > FLASH_VER_LAST
    ) and os.environ.get("XFORMERS_IGNORE_FLASH_VERSION_CHECK", "0") != "1":
        raise ImportError(
            f"Requires Flash-Attention version >={'.'.join([str(i) for i in FLASH_VER_MIN])},"
            f"<={'.'.join([str(i) for i in FLASH_VER_LAST])} "
            f"but got {FLASH_VERSION}."
        )
    VARLEN_LSE_PACKED = True

elif _TRY_PT_FLASH_ATTN:
    pt_flash_is_old = is_pt_flash_old(force=True) is True
    FLASH_VERSION = torch.nn.attention._get_flash_version()  # type: ignore
    VARLEN_LSE_PACKED = not pt_flash_is_old
    _USE_PT_FLASH_ATTN = True


if FLASH_VERSION != "0.0.0":

    @torch.library.custom_op(
        "xformers_flash::flash_fwd",
        mutates_args=(),
        device_types=["cuda"],
    )
    def _flash_fwd(
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
        window_left: int,
        window_right: int,
        return_softmax: bool,
        block_tables: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        softcap = 0.0
        if _USE_PT_FLASH_ATTN:
            ret = torch.ops.aten._flash_attention_forward(
                query,
                key,
                value,
                cu_seqlens_q,  # cum_seq_q
                cu_seqlens_k,  # cum_seq_k
                max_seqlen_q,  # max_q
                max_seqlen_k,  # max_k
                p,  # dropout_p
                is_causal,
                return_debug_mask=False,
                scale=softmax_scale,
                window_size_left=window_left,
                window_size_right=window_right,
                seqused_k=seqused_k,
                alibi_slopes=None,  # alibi_slopes
            )
            if pt_flash_is_old:
                (
                    attention,
                    logsumexp,
                    philox_seed,
                    philox_offset,
                    _,
                ) = ret
                rng_state = torch.stack([philox_seed, philox_offset])
            else:
                attention, logsumexp, rng_state, _, _ = ret
            return attention, logsumexp, rng_state
        else:
            if cu_seqlens_q is None:
                assert cu_seqlens_k is None
                assert seqused_k is None
                out, softmax_lse, p, rng_state = _C_flashattention.fwd(
                    query,
                    key,
                    value,
                    None,  # out
                    None,  # alibi_slopes
                    p,
                    softmax_scale,
                    is_causal,
                    window_left,  # window_size_left
                    window_right,  # window_size_right
                    softcap,
                    return_softmax,
                    None,  # rng
                )
            else:
                out, softmax_lse, p, rng_state = _C_flashattention.varlen_fwd(
                    query,
                    key,
                    value,
                    None,  # out
                    cu_seqlens_q,
                    cu_seqlens_k,
                    seqused_k,
                    None,  # leftpad_k_
                    block_tables,
                    None,  # alibi_slopes
                    max_seqlen_q,
                    max_seqlen_k,
                    p,
                    softmax_scale,
                    False,
                    is_causal,
                    window_left,
                    window_right,
                    softcap,
                    return_softmax,
                    None,  # gen
                )
        return out, softmax_lse, rng_state

    @torch.library.register_fake("xformers_flash::flash_fwd")
    def _flash_fwd_abstract(
        query,
        key,
        value,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        p,
        softmax_scale,
        is_causal,
        window_left,
        window_right,
        return_softmax,
        block_tables,
    ):
        out = torch.empty_like(query)
        if cu_seqlens_q is None:
            B, M, H, K = query.shape
            lse_shape = [B, H, M]
        else:
            M, H, K = query.shape
            B = cu_seqlens_q.shape[0] - 1
            if VARLEN_LSE_PACKED:
                lse_shape = [H, M]
            else:
                lse_shape = [B, H, max_seqlen_q]
        softmax_lse = torch.empty(lse_shape, device=query.device, dtype=torch.float32)
        rng_state = torch.empty([2], device=query.device, dtype=torch.int64)
        return out, softmax_lse, rng_state

    @torch.library.custom_op(
        "xformers_flash::flash_bwd",
        mutates_args=(),
        device_types=["cuda"],
    )
    def _flash_bwd(
        grads_share_storage: bool,
        grad: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        out: torch.Tensor,
        lse: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        p: float,
        softmax_scale: float,
        is_causal: bool,
        window_left: int,
        window_right: int,
        rng_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        softcap = 0.0
        if _USE_PT_FLASH_ATTN:
            assert softcap == 0.0
            if rng_state is not None and pt_flash_is_old:
                rng_state0 = rng_state[0]
                rng_state1 = rng_state[1]
            else:
                rng_state0 = rng_state1 = rng_state
            dq, dk, dv = torch.ops.aten._flash_attention_backward(
                grad,
                query,
                key,
                value,
                out,
                lse,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                p,
                is_causal,
                rng_state0,
                rng_state1,
                scale=softmax_scale,
                window_size_left=window_left,
                window_size_right=window_right,
            )
        else:
            dq, dk, dv = _create_dq_dk_dv(grads_share_storage, query, key, value)
            if cu_seqlens_k is None:
                assert cu_seqlens_q is None
                _C_flashattention.bwd(
                    grad,
                    query,
                    key,
                    value,
                    out,
                    lse,
                    dq,
                    dk,
                    dv,
                    None,  # alibi_slopes
                    p,
                    softmax_scale,
                    is_causal,
                    window_left,
                    window_right,
                    softcap,
                    False,  # deterministic
                    None,
                    rng_state,
                )
            else:
                _C_flashattention.varlen_bwd(
                    grad,
                    query,
                    key,
                    value,
                    out,
                    lse,
                    dq,
                    dk,
                    dv,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    None,  # alibi_slopes
                    max_seqlen_q,
                    max_seqlen_k,
                    p,
                    softmax_scale,
                    False,  # zero_tensors
                    is_causal,
                    window_left,
                    window_right,
                    softcap,
                    False,  # deterministic
                    None,
                    rng_state,
                )
        return dq, dk, dv

    @torch.library.register_fake("xformers_flash::flash_bwd")
    def _flash_bwd_abstract(
        grads_share_storage,
        grad,
        query,
        key,
        value,
        *args,
        **kwargs,
    ):
        return _create_dq_dk_dv(grads_share_storage, query, key, value)

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


def _convert_input_format(
    inp: Inputs,
    supports_mqa: bool,
    use_kvsplit: bool = False,
) -> Tuple[
    Inputs,
    Optional[torch.Tensor],
    int,
    Optional[torch.Tensor],
    int,
    Optional[torch.Tensor],
]:
    assert inp.query.ndim in [4, 5]
    query, key, value = inp.query, inp.key, inp.value
    batch = query.shape[0]
    seqlen_q = query.shape[1]
    seqlen_kv = key.shape[1]
    head_dim_q = query.shape[-1]
    head_dim_v = value.shape[-1]

    attn_bias = inp.attn_bias
    if isinstance(attn_bias, BlockDiagonalMask):
        assert attn_bias.k_seqinfo.seqstart.device == inp.query.device
        cu_seqlen_k = attn_bias.k_seqinfo.seqstart
        cu_seqlen_q = attn_bias.q_seqinfo.seqstart
        max_seqlen_q = attn_bias.q_seqinfo.max_seqlen
        max_seqlen_k = attn_bias.k_seqinfo.max_seqlen
        seqused_k = None
    elif isinstance(
        attn_bias,
        (
            BlockDiagonalGappyKeysMask,
            BlockDiagonalPaddedKeysMask,
            PagedBlockDiagonalGappyKeysMask,
            PagedBlockDiagonalPaddedKeysMask,
        ),
    ):
        assert attn_bias.k_seqinfo.seqstart.device == inp.query.device
        cu_seqlen_k = attn_bias.k_seqinfo.seqstart
        cu_seqlen_q = attn_bias.q_seqinfo.seqstart
        max_seqlen_q = attn_bias.q_seqinfo.max_seqlen
        max_seqlen_k = attn_bias.k_seqinfo.max_seqlen
        seqused_k = attn_bias.k_seqinfo.seqlen
    else:
        cu_seqlen_k = None
        cu_seqlen_q = None
        seqused_k = None
        max_seqlen_q = inp.query.shape[1]
        max_seqlen_k = inp.key.shape[1]

    if query.ndim == 5:  # GQA
        assert supports_mqa

        # Fold the group/head_in_group dimensions together
        def fold(x):
            # Either the head is replicated
            if x.stride(3) == 0:
                return x[:, :, :, 0]
            # Or we reshape
            return x.reshape(
                [
                    x.shape[0],
                    x.shape[1],
                    -1,
                    x.shape[4],
                ]
            )

        query = fold(query)
        key = fold(key)
        value = fold(value)
    # Optimize for MHA
    if supports_mqa and key.ndim == 4 and key.stride(2) == 0 and value.stride(2) == 0:
        key = key[:, :, :1]
        value = value[:, :, :1]
    # Initially we have `query.shape = [batch, seqlen, num_heads, head_dim_q]`
    # We want format `[batch * seqlen, num_heads, head_dim_q]`
    if cu_seqlen_k is not None:
        query = query.reshape([batch * seqlen_q, -1, head_dim_q])
        key = key.reshape([batch * seqlen_kv, -1, head_dim_q])
        value = value.reshape([batch * seqlen_kv, -1, head_dim_v])
        if isinstance(
            attn_bias,
            (PagedBlockDiagonalGappyKeysMask, PagedBlockDiagonalPaddedKeysMask),
        ):
            num_pages = value.shape[0] // attn_bias.page_size
            key = key.view(num_pages, attn_bias.page_size, *key.shape[1:])
            value = value.view(num_pages, attn_bias.page_size, *value.shape[1:])

    new_inp = Inputs(
        query=query,
        key=key,
        value=value,
        attn_bias=attn_bias,
        p=inp.p,
        scale=inp.scale,
        output_dtype=inp.output_dtype,
        is_partial=inp.is_partial,
    )
    return new_inp, cu_seqlen_q, max_seqlen_q, cu_seqlen_k, max_seqlen_k, seqused_k


def _is_causal(attn_bias: Optional[Union[torch.Tensor, AttentionBias]]) -> bool:
    return isinstance(
        attn_bias,
        (
            LowerTriangularMask,
            LowerTriangularFromBottomRightMask,
            LowerTriangularFromBottomRightLocalAttentionMask,
            BlockDiagonalCausalMask,
            BlockDiagonalCausalLocalAttentionMask,
            BlockDiagonalCausalFromBottomRightMask,
            BlockDiagonalCausalLocalAttentionFromBottomRightMask,
            BlockDiagonalCausalLocalAttentionPaddedKeysMask,
            BlockDiagonalCausalWithOffsetGappyKeysMask,
            BlockDiagonalCausalWithOffsetPaddedKeysMask,
            PagedBlockDiagonalCausalWithOffsetGappyKeysMask,
            PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
        ),
    )


def _is_paged_attention_supported(attn_bias_type) -> bool:
    if issubclass(attn_bias_type, PagedBlockDiagonalPaddedKeysMask):
        return not _USE_PT_FLASH_ATTN
    if issubclass(
        attn_bias_type,
        (PagedBlockDiagonalGappyKeysMask, PagedBlockDiagonalPaddedKeysMask),
    ):
        return not torch.mtia.is_available()

    return True


def _window_size(
    attn_bias: Optional[Union[torch.Tensor, AttentionBias]],
) -> Tuple[int, int]:
    win_left = -1
    win_right = -1
    if isinstance(
        attn_bias,
        (
            BlockDiagonalCausalLocalAttentionMask,
            BlockDiagonalCausalLocalAttentionFromBottomRightMask,
            BlockDiagonalCausalLocalAttentionPaddedKeysMask,
            LowerTriangularFromBottomRightLocalAttentionMask,
        ),
    ):
        win_left = attn_bias._window_size - 1
    if isinstance(
        attn_bias,
        (
            BlockDiagonalLocalAttentionPaddedKeysMask,
            LocalAttentionFromBottomRightMask,
        ),
    ):
        win_left = attn_bias.window_left
        win_right = attn_bias.window_right
    return (win_left, win_right)


def _check_needs_no_topleft(d: Inputs, reasons: List[str]) -> None:
    # Flash does not support TopLeft, so only allow causal masks with TopLeft
    # if each batch element has equal number of queries and keys.
    if isinstance(d.attn_bias, BlockDiagonalCausalMask):
        # Flash does not support TopLeft, so only allow BlockDiagonalCausalMask
        # if each batch element has equal number of queries and keys.
        for k_start, q_start in zip_longest(
            d.attn_bias.k_seqinfo.seqstart_py, d.attn_bias.q_seqinfo.seqstart_py
        ):
            if k_start != q_start:
                reasons.append(
                    "Only support BlockDiagonalCausalMask if equal"
                    " numbers of keys and queries"
                )
                break
    elif isinstance(d.attn_bias, LowerTriangularMask):
        if d.query.shape[1] != d.key.shape[1]:
            reasons.append(
                "Only support LowerTriangularMask if equal number of" "keys and queries"
            )


def _check_strides_for_bmghk(x: torch.Tensor, name: str, reasons: List[str]) -> None:
    """
    We want to be able to collapse the G/H dimensions together
    """
    if x.ndim == 5:
        stride_g, stride_h = x.stride(2), x.stride(3)
        if x.shape[2] == 1:
            return
        if x.shape[3] == 1 or stride_h == 0:
            return
        if stride_g != stride_h * x.shape[-2]:
            reasons.append(
                f"GQA is only supported when the G/H dimensions are contiguous\n"
                f"    {name}.stride:  {x.stride()}\n"
                f"    {name}.shape :  {list(x.shape)}"
            )


def _post_process_lse(
    lse: torch.Tensor,
    inp: Inputs,
    original_query_shape: Tuple[int, ...],
    varlen_lse_packed: bool = VARLEN_LSE_PACKED,
) -> torch.Tensor:
    # Easy case: no varlen
    if not isinstance(inp.attn_bias, VARLEN_BIASES):
        if len(original_query_shape) == 5:
            # [B, GH, M] => [B, G, H, M]
            return lse.unflatten(1, original_query_shape[2:4])
        return lse

    # Already packed: just bring back the batch dimension
    if varlen_lse_packed:
        if len(original_query_shape) == 5:
            # (1, G, H, total_q)
            return lse.unflatten(0, original_query_shape[2:4]).unsqueeze(0)
        # (1, H, total_q)
        return lse.unsqueeze(0)

    if not inp.is_partial:
        # (B, H, M)
        return lse

    # reshape from (B, G*H, max_seqlen) to (1, G*H, B*max_seqlen)
    # Unfortunately this flatten is not just a view.
    lse_hkm = lse.permute(1, 0, 2).flatten(start_dim=1)[None]
    if len(original_query_shape) == 5:
        return lse_hkm.unflatten(1, original_query_shape[2:4])
    return lse_hkm


@register_operator
class FwOp(AttentionFwOpBase):
    """Operator that computes memory-efficient attention using \
        `Flash-Attention <https://github.com/HazyResearch/flash-attention>`_ \
        implementation.
    """

    OPERATOR = get_operator("xformers_flash", "flash_fwd")
    SUPPORTED_DEVICES: Set[str] = {"cuda"}
    CUDA_MINIMUM_COMPUTE_CAPABILITY = (8, 0)
    SUPPORTED_DTYPES: Set[torch.dtype] = {torch.half, torch.bfloat16}
    SUPPORTED_MAX_K = 256
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
        BlockDiagonalGappyKeysMask,
        BlockDiagonalPaddedKeysMask,
        LocalAttentionFromBottomRightMask,
        PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
        PagedBlockDiagonalPaddedKeysMask,
    )

    SUPPORTED_ATTN_BIAS_TYPES = [
        b for b in SUPPORTED_ATTN_BIAS_TYPES if _is_paged_attention_supported(b)
    ]

    SUPPORTS_DROPOUT = True
    SUPPORTS_CUSTOM_SCALE = True
    SUPPORTS_DIFFERENT_VALUE_EMBED = False
    SUPPORTS_BMGHK = True
    SUPPORTS_PARTIAL = True
    VARLEN_LSE_PACKED = VARLEN_LSE_PACKED
    NAME = f"fa2F@{FLASH_VERSION}-pt" if _USE_PT_FLASH_ATTN else f"fa2F@{FLASH_VERSION}"
    VERSION = FLASH_VERSION

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:
        reasons = super(FwOp, cls).not_supported_reasons(d)
        check_lastdim_alignment_stride1(reasons, "query", d.query, 8)
        _check_needs_no_topleft(d, reasons)
        _check_strides_for_bmghk(d.query, "query", reasons)
        _check_strides_for_bmghk(d.key, "key", reasons)
        _check_strides_for_bmghk(d.value, "value", reasons)

        if (
            d.is_partial
            and not VARLEN_LSE_PACKED
            and isinstance(d.attn_bias, VARLEN_BIASES)
        ):
            q_seqinfo = d.attn_bias.q_seqinfo
            if q_seqinfo.min_seqlen != q_seqinfo.max_seqlen:
                # Flash provides padded LSE which we don't handle.
                reasons.append("partial attention with heterogeneous queries")
        return reasons

    @classmethod
    def apply(
        cls, inp: Inputs, needs_gradient: bool
    ) -> Tuple[torch.Tensor, Optional[Context]]:
        return_softmax = False
        original_query_shape = inp.query.shape

        out_shape = [
            *inp.query.shape[:-1],
            inp.value.shape[-1],
        ]
        # no cumulative seqlen
        (
            inp,
            cu_seqlens_q,
            max_seqlen_q,
            cu_seqlens_k,
            max_seqlen_k,
            seqused_k,
        ) = _convert_input_format(inp, supports_mqa=True)

        if inp.query.numel() > 0 and inp.key.numel() > 0:
            win_left, win_right = _window_size(inp.attn_bias)
            block_tables = (
                inp.attn_bias.block_tables
                if isinstance(inp.attn_bias, PagedBlockDiagonalPaddedKeysMask)
                else None
            )
            out, softmax_lse, rng_state = cls.OPERATOR(
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
                window_left=win_left,
                window_right=win_right,
                return_softmax=return_softmax,
                block_tables=block_tables,
            )
            out = out.reshape(out_shape)
        else:
            out = torch.zeros(out_shape, device=inp.query.device, dtype=inp.query.dtype)
            rng_state = None
            softmax_lse = torch.empty(
                (
                    [inp.query.shape[2], inp.query.shape[0] * inp.query.shape[1]]
                    if VARLEN_LSE_PACKED and isinstance(inp.attn_bias, VARLEN_BIASES)
                    else [inp.query.shape[0], inp.query.shape[2], inp.query.shape[1]]
                ),
                device=inp.query.device,
                dtype=torch.float32,
            )
        if not needs_gradient:
            return out, None
        ctx = Context(
            out=out,
            lse=_post_process_lse(softmax_lse, inp, original_query_shape),
        )
        if inp.p != 0.0:
            ctx.op_bw = BwOp
            ctx.rng_state = rng_state
        return (out, ctx)


@register_operator
class BwOp(AttentionBwOpBase):
    __doc__ = FwOp.__doc__

    OPERATOR = get_operator("xformers_flash", "flash_bwd")
    SUPPORTED_DEVICES = FwOp.SUPPORTED_DEVICES
    CUDA_MINIMUM_COMPUTE_CAPABILITY = FwOp.CUDA_MINIMUM_COMPUTE_CAPABILITY
    SUPPORTED_DTYPES = FwOp.SUPPORTED_DTYPES
    SUPPORTED_MAX_K = FwOp.SUPPORTED_MAX_K
    SUPPORTED_ATTN_BIAS_TYPES: Iterable[Any] = tuple(
        set(FwOp.SUPPORTED_ATTN_BIAS_TYPES).difference(
            {
                BlockDiagonalCausalLocalAttentionPaddedKeysMask,
                BlockDiagonalCausalWithOffsetGappyKeysMask,
                BlockDiagonalCausalWithOffsetPaddedKeysMask,
                BlockDiagonalGappyKeysMask,
                BlockDiagonalPaddedKeysMask,
                PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
                PagedBlockDiagonalPaddedKeysMask,
            }
        )
    )
    SUPPORTS_DROPOUT = FwOp.SUPPORTS_DROPOUT
    SUPPORTS_CUSTOM_SCALE = FwOp.SUPPORTS_CUSTOM_SCALE
    SUPPORTS_DIFFERENT_VALUE_EMBED = FwOp.SUPPORTS_DIFFERENT_VALUE_EMBED
    IS_DETERMINISTIC = False
    SUPPORTS_BMGHK = False  # NOTE: Don't forget to update fmha doc when changing this!
    VARLEN_LSE_PACKED = VARLEN_LSE_PACKED
    NAME = f"fa2B@{FLASH_VERSION}-pt" if _USE_PT_FLASH_ATTN else f"fa2B@{FLASH_VERSION}"
    VERSION = FLASH_VERSION

    MAX_HEADDIM_DROPOUT_SM8x = 224

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:
        reasons = super(BwOp, cls).not_supported_reasons(d)
        check_lastdim_alignment_stride1(reasons, "query", d.query, 8)
        _check_needs_no_topleft(d, reasons)
        if d.device.type == "cuda":
            # Due to limited shared-memory, some GPUs are limited in head dimension
            device_capability = torch.cuda.get_device_capability(d.device)
            is_sm80_or_sm90 = device_capability in [(8, 0), (9, 0)]
            if (
                max(d.key.shape[-1], d.query.shape[-1]) > cls.MAX_HEADDIM_DROPOUT_SM8x
                and not is_sm80_or_sm90
                and d.p != 0.0
            ):
                reasons.append(
                    "requires a GPU with compute capability 8.0 "
                    f"(A100) or 9.0 (H100) for dropout when 'query.shape[-1] > {cls.MAX_HEADDIM_DROPOUT_SM8x}'"
                )
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
            seqused_k,
        ) = _convert_input_format(inp, supports_mqa=False)
        # assert ctx.lse.is_contiguous()
        assert seqused_k is None
        ctx_lse = ctx.lse
        if isinstance(inp.attn_bias, VARLEN_BIASES) and VARLEN_LSE_PACKED:
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
            grads = Gradients(
                *cls.OPERATOR(
                    ctx.qkv_share_storage,
                    grad.reshape(kernel_out_shape).contiguous(),
                    inp.query,
                    inp.key,
                    inp.value,
                    ctx.out.reshape(kernel_out_shape),
                    ctx_lse,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    inp.p,
                    inp.scale_float,
                    _is_causal(inp.attn_bias),
                    window_left=win_left,
                    window_right=win_right,
                    rng_state=ctx.rng_state if inp.p > 0.0 else None,
                )
            )
        else:
            grads = Gradients(
                dq=torch.zeros_like(inp.query),
                dk=torch.zeros_like(inp.key),
                dv=torch.zeros_like(inp.value),
            )
        if grads.dq.numel() == 0:
            grads.dk.zero_()
            grads.dv.zero_()
        if grads.dv.numel() == 0:
            grads.dq.zero_()
        grads.dq = grads.dq.reshape(dq_shape)
        grads.dk = grads.dk.reshape(dk_shape)
        grads.dv = grads.dv.reshape(dv_shape)
        return grads
