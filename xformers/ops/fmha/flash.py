# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import os
from dataclasses import replace
from itertools import zip_longest
from typing import Any, List, Optional, Set, Tuple, Union

import torch

from ..common import _get_storage_base, get_operator, register_operator
from .attn_bias import (
    AttentionBias,
    BlockDiagonalCausalFromBottomRightMask,
    BlockDiagonalCausalLocalAttentionFromBottomRightMask,
    BlockDiagonalCausalLocalAttentionMask,
    BlockDiagonalCausalMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    BlockDiagonalMask,
    LocalAttentionFromBottomRightMask,
    LowerTriangularFromBottomRightLocalAttentionMask,
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

FLASH_VERSION = "0.0.0"
try:
    try:
        from ... import _C_flashattention  # type: ignore[attr-defined]
        from ..._cpp_lib import _build_metadata

        if _build_metadata is not None:
            FLASH_VERSION = _build_metadata.flash_version
    except ImportError:
        import flash_attn
        from flash_attn.flash_attn_interface import flash_attn_cuda as _C_flashattention

        FLASH_VERSION = flash_attn.__version__
        FLASH_VER_MIN = (2, 5, 2)
        FLASH_VER_LAST = (2, 5, 6)  # last supported, inclusive
        flash_ver_parsed = tuple(int(s) for s in FLASH_VERSION.split(".")[:3])
        if (
            flash_ver_parsed < FLASH_VER_MIN or flash_ver_parsed > FLASH_VER_LAST
        ) and os.environ.get("XFORMERS_IGNORE_FLASH_VERSION_CHECK", "0") != "1":
            raise ImportError(
                f"Requires Flash-Attention version >={'.'.join([str(i) for i in FLASH_VER_MIN])},"
                f"<={'.'.join([str(i) for i in FLASH_VER_LAST])} "
                f"but got {FLASH_VERSION}."
            )

    # create library so that flash-attn goes through the PyTorch Dispatcher
    _flash_lib = torch.library.Library("xformers_flash", "DEF")

    _flash_lib.define(
        "flash_fwd(Tensor query, Tensor key, Tensor value, "
        "Tensor? cu_seqlens_q, Tensor? cu_seqlens_k, Tensor? seqused_k, "
        "int max_seqlen_q, int max_seqlen_k, "
        "float p, float softmax_scale, "
        "bool is_causal, int window_left, "
        "int window_right, bool return_softmax) -> (Tensor, Tensor, Tensor)"
    )

    _flash_lib.define(
        "flash_bwd(Tensor dout, Tensor query, Tensor key, Tensor value, "
        "Tensor out, Tensor softmax_lse_, Tensor dq, Tensor dk, Tensor dv, "
        "Tensor cu_seqlens_q, Tensor cu_seqlens_k, "
        "int max_seqlen_q, int max_seqlen_k, "
        "float p, float softmax_scale, bool is_causal, "
        "int window_left, int window_right, Tensor rng_state) -> (Tensor, Tensor, Tensor)"
    )

    def _flash_fwd(
        query,
        key,
        value,
        cu_seq_lens_q,
        cu_seq_lens_k,
        seqused_k,
        max_seq_len_q,
        max_seq_len_k,
        p,
        softmax_scale,
        is_causal,
        window_left,
        window_right,
        return_softmax,
    ):
        if cu_seq_lens_q is None:
            assert cu_seq_lens_k is None
            assert seqused_k is None
            (
                out,
                q_padded,
                k_padded,
                v_padded,
                out_padded,
                softmax_lse,
                p,
                rng_state,
            ) = _C_flashattention.fwd(
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
                return_softmax,
                None,  # rng
            )
        else:
            (
                out,
                q_padded,
                k_padded,
                v_padded,
                out_padded,
                softmax_lse,
                p,
                rng_state,
            ) = _C_flashattention.varlen_fwd(
                query,
                key,
                value,
                None,  # out
                cu_seq_lens_q,
                cu_seq_lens_k,
                seqused_k,
                None,  # alibi_slopes
                max_seq_len_q,
                max_seq_len_k,
                p,
                softmax_scale,
                False,
                is_causal,
                window_left,
                window_right,
                return_softmax,
                None,
            )
        return out, softmax_lse, rng_state

    def _flash_bwd(
        grad,
        query,
        key,
        value,
        out,
        lse,
        dq,
        dk,
        dv,
        cu_seq_lens_q,
        cu_seq_lens_k,
        max_seq_len_q,
        max_seq_len_k,
        p,
        softmax_scale,
        is_causal,
        window_left,
        window_right,
        rng_state,
    ):
        if cu_seq_lens_k is None:
            assert cu_seq_lens_q is None
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
                cu_seq_lens_q,
                cu_seq_lens_k,
                None,  # alibi_slopes
                max_seq_len_q,
                max_seq_len_k,
                p,
                softmax_scale,
                False,  # zero_tensors
                is_causal,
                window_left,
                window_right,
                False,  # deterministic
                None,
                rng_state,
            )
        return dq, dk, dv

    _flash_lib.impl("flash_fwd", _flash_fwd, "CUDA")
    _flash_lib.impl("flash_bwd", _flash_bwd, "CUDA")
except ImportError:
    pass


def _convert_input_format(
    inp: Inputs,
    supports_mqa: bool,
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
        # BlockDiagonalMask or BlockDiagonalCausalMask
        attn_bias.k_seqinfo.seqstart = attn_bias.k_seqinfo.seqstart.to(
            inp.query.device, non_blocking=True
        )
        attn_bias.q_seqinfo.seqstart = attn_bias.q_seqinfo.seqstart.to(
            inp.query.device, non_blocking=True
        )

        cu_seqlen_k = attn_bias.k_seqinfo.seqstart
        cu_seqlen_q = attn_bias.q_seqinfo.seqstart
        max_seqlen_q = attn_bias.q_seqinfo.max_seqlen
        max_seqlen_k = attn_bias.k_seqinfo.max_seqlen
        seqused_k = None
    elif isinstance(attn_bias, BlockDiagonalCausalWithOffsetPaddedKeysMask):
        attn_bias.k_seqinfo.seqstart = attn_bias.k_seqinfo.seqstart.to(
            inp.query.device, non_blocking=True
        )
        attn_bias.q_seqinfo.seqstart = attn_bias.q_seqinfo.seqstart.to(
            inp.query.device, non_blocking=True
        )
        attn_bias.k_seqinfo.seqlen = attn_bias.k_seqinfo.seqlen.to(
            inp.query.device, non_blocking=True
        )
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
    if key.ndim == 4 and key.stride(2) == 0 and value.stride(2) == 0 and supports_mqa:
        key = key[:, :, :1]
        value = value[:, :, :1]
    # Initially we have `query.shape = [batch, seqlen, num_heads, head_dim_q]`
    # We want format `[batch * seqlen, num_heads, head_dim_q]`
    if cu_seqlen_k is not None:
        query = query.reshape([batch * seqlen_q, -1, head_dim_q])
        key = key.reshape([batch * seqlen_kv, -1, head_dim_q])
        value = value.reshape([batch * seqlen_kv, -1, head_dim_v])
    new_inp = replace(
        inp,
        query=query,
        key=key,
        value=value,
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
            BlockDiagonalCausalWithOffsetPaddedKeysMask,
        ),
    )


def _window_size(
    attn_bias: Optional[Union[torch.Tensor, AttentionBias]]
) -> Tuple[int, int]:
    win_left = -1
    win_right = -1
    if isinstance(
        attn_bias,
        (
            BlockDiagonalCausalLocalAttentionMask,
            BlockDiagonalCausalLocalAttentionFromBottomRightMask,
            LowerTriangularFromBottomRightLocalAttentionMask,
        ),
    ):
        win_left = attn_bias._window_size - 1
    if isinstance(attn_bias, LocalAttentionFromBottomRightMask):
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
    lse: torch.Tensor, inp: Inputs, original_query_shape: Tuple[int, ...]
) -> torch.Tensor:
    if not isinstance(inp.attn_bias, BlockDiagonalCausalWithOffsetPaddedKeysMask):
        if inp.is_partial and len(original_query_shape) == 5:
            # [B, GH, M] => [B, G, H, M]
            return lse.unflatten(1, original_query_shape[2:4])
        return lse
    q_seqinfo = inp.attn_bias.q_seqinfo
    B = len(q_seqinfo.seqstart_py) - 1
    if q_seqinfo.max_seqlen * B != original_query_shape[1]:
        # Heterogeneous batch. We can't fix it.
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
    SUPPORTED_ATTN_BIAS_TYPES: Set[Any] = {
        type(None),
        LowerTriangularMask,
        LowerTriangularFromBottomRightMask,
        LowerTriangularFromBottomRightLocalAttentionMask,
        BlockDiagonalMask,
        BlockDiagonalCausalMask,
        BlockDiagonalCausalLocalAttentionMask,
        BlockDiagonalCausalLocalAttentionFromBottomRightMask,
        BlockDiagonalCausalFromBottomRightMask,
        BlockDiagonalCausalWithOffsetPaddedKeysMask,
        LocalAttentionFromBottomRightMask,
    }
    SUPPORTS_DROPOUT = True
    SUPPORTS_CUSTOM_SCALE = True
    SUPPORTS_DIFFERENT_VALUE_EMBED = False
    SUPPORTS_BMGHK = True
    SUPPORTS_PARTIAL = True
    NAME = f"flshattF@{FLASH_VERSION}"
    VERSION = FLASH_VERSION

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:
        reasons = super(FwOp, cls).not_supported_reasons(d)
        check_lastdim_alignment_stride1(reasons, "query", d.query, 8)
        _check_needs_no_topleft(d, reasons)
        _check_strides_for_bmghk(d.query, "query", reasons)
        _check_strides_for_bmghk(d.key, "key", reasons)
        _check_strides_for_bmghk(d.value, "value", reasons)
        if d.is_partial and isinstance(
            d.attn_bias, BlockDiagonalCausalWithOffsetPaddedKeysMask
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
            )
            out = out.reshape(out_shape)
        else:
            out = torch.zeros(out_shape, device=inp.query.device, dtype=inp.query.dtype)
            rng_state = None
            softmax_lse = torch.empty(
                [inp.query.shape[0], inp.query.shape[2], inp.query.shape[1]],
                device=inp.query.device,
                dtype=torch.float32,
            )
        if not needs_gradient:
            return out, None
        ctx = Context(
            out=out, lse=_post_process_lse(softmax_lse, inp, original_query_shape)
        )
        if inp.p != 0.0:
            ctx.op_bw = BwOp
            ctx.rng_state = rng_state
        return (out, ctx)

    @classmethod
    # type: ignore
    def operator_flop(
        cls,
        query,
        key,
        value,
        cu_seq_lens_q,
        cu_seq_lens_k,
        max_seq_len_q,
        max_seq_len_k,
        p,
        softmax_scale,
        causal,
        return_softmax,
    ) -> int:
        return cls.attn_operator_flop(
            query.unsqueeze(0),
            key.unsqueeze(0),
            value.unsqueeze(0),
            causal=causal,
            seqstart_k=cu_seq_lens_k,
            seqstart_q=cu_seq_lens_q,
        )


@register_operator
class BwOp(AttentionBwOpBase):
    __doc__ = FwOp.__doc__

    OPERATOR = get_operator("xformers_flash", "flash_bwd")
    SUPPORTED_DEVICES = FwOp.SUPPORTED_DEVICES
    CUDA_MINIMUM_COMPUTE_CAPABILITY = FwOp.CUDA_MINIMUM_COMPUTE_CAPABILITY
    SUPPORTED_DTYPES = FwOp.SUPPORTED_DTYPES
    SUPPORTED_MAX_K = FwOp.SUPPORTED_MAX_K
    SUPPORTED_ATTN_BIAS_TYPES = FwOp.SUPPORTED_ATTN_BIAS_TYPES.difference(
        {BlockDiagonalCausalWithOffsetPaddedKeysMask}
    )
    SUPPORTS_DROPOUT = FwOp.SUPPORTS_DROPOUT
    SUPPORTS_CUSTOM_SCALE = FwOp.SUPPORTS_CUSTOM_SCALE
    SUPPORTS_DIFFERENT_VALUE_EMBED = FwOp.SUPPORTS_DIFFERENT_VALUE_EMBED
    IS_DETERMINISTIC = False
    SUPPORTS_BMGHK = False  # NOTE: Don't forget to update fmha doc when changing this!
    NAME = f"flshattB@{FLASH_VERSION}"
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
        assert ctx.lse.is_contiguous()
        assert seqused_k is None
        ctx_lse = ctx.lse
        assert ctx_lse.shape[2] >= max_seqlen_q
        if max_seqlen_q != ctx_lse.shape[2]:
            ctx_lse = ctx_lse[:, :, :max_seqlen_q].contiguous()
        kernel_out_shape = [
            *inp.query.shape[:-1],
            inp.value.shape[-1],
        ]

        # Create dq,dk,dv
        # If Q/K/V come from a single QKV tensor, let's put the gradient in the
        # right strides, so we can avoid a `cat`
        if (
            inp.query.shape[0] == inp.key.shape[0]
            and inp.query.shape[-1] == inp.value.shape[-1]
            and _get_storage_base(inp.query) == _get_storage_base(inp.key)
            and _get_storage_base(inp.query) == _get_storage_base(inp.value)
        ):
            # Create one big contiguous chunk
            # This is because q, k and v usually come from a single
            # output of a linear layer that is chunked.
            # Creating the gradients with the right layout saves us
            # a `torch.cat` call in the backward pass
            chunk = torch.empty(
                (*inp.query.shape[0:-2], 3, inp.query.shape[-2], inp.query.shape[-1]),
                dtype=inp.query.dtype,
                device=inp.device,
            )
            grads = Gradients(
                dq=chunk.select(-3, 0),
                dk=chunk.select(-3, 1),
                dv=chunk.select(-3, 2),
            )
        else:
            grads = Gradients(
                dq=torch.empty_like(inp.query),
                dk=torch.empty_like(inp.key),
                dv=torch.empty_like(inp.value),
            )

        assert grad.dtype in cls.SUPPORTED_DTYPES

        if grads.dq.numel() == 0:
            grads.dk.zero_()
            grads.dv.zero_()
        if grads.dv.numel() == 0:
            grads.dq.zero_()
        if grads.dq.numel() and grads.dk.numel():
            win_left, win_right = _window_size(inp.attn_bias)
            cls.OPERATOR(
                grad.reshape(kernel_out_shape).contiguous(),
                inp.query,
                inp.key,
                inp.value,
                ctx.out.reshape(kernel_out_shape),
                ctx_lse,
                grads.dq,
                grads.dk,
                grads.dv,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                inp.p,
                inp.scale_float,
                _is_causal(inp.attn_bias),
                window_left=win_left,
                window_right=win_right,
                rng_state=ctx.rng_state,
            )
        grads.dq = grads.dq.reshape(dq_shape)
        grads.dk = grads.dk.reshape(dk_shape)
        grads.dv = grads.dv.reshape(dv_shape)
        return grads

    @classmethod
    # type: ignore
    def operator_flop(
        cls,
        grad,
        query,
        key,
        value,
        out,
        lse,
        dq,
        dk,
        dv,
        cu_seq_lens_q,
        cu_seq_lens_k,
        max_seq_len_q,
        max_seq_len_k,
        p,
        softmax_scale,
        causal,
    ) -> int:
        return cls.attn_operator_flop(
            query.unsqueeze(0),
            key.unsqueeze(0),
            value.unsqueeze(0),
            causal=causal,
            seqstart_k=cu_seq_lens_k,
            seqstart_q=cu_seq_lens_q,
        )
