# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Iterable, List, Optional, Set, Tuple, Union

import torch

from ..common import register_operator
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
)

from .common import AttentionBwOpBase, AttentionFwOpBase, Context, Gradients, Inputs


def _get_operator(name: str):
    def no_such_operator(*args, **kwargs):
        raise RuntimeError(
            "No such operator "
            f"fbgemm_gpu.experimental.gen_ai.attention.cutlass_blackwell_fmha.{name} "
            "- did you forget to build xformers with `python setup.py develop`?"
        )

    def no_cuda_environment(*args, **kwargs):
        raise RuntimeError(
            "The operator "
            f"fbgemm_gpu.experimental.gen_ai.attention.cutlass_blackwell_fmha.{name} "
            "cannot run in a non-cuda environment."
        )

    try:
        from fbgemm_gpu.experimental.gen_ai.attention.cutlass_blackwell_fmha import (
            cutlass_blackwell_fmha_interface as fmha,
        )

        return getattr(fmha, name)
    except (RuntimeError, ModuleNotFoundError):
        return no_such_operator
    except OSError as e:
        if torch.cuda.is_available() is False:
            return no_cuda_environment
        raise e


def _convert_input_format(
    inp: Inputs,
) -> Tuple[
    Inputs,
    Optional[torch.Tensor],
    Optional[int],
    Optional[torch.Tensor],
    Optional[int],
    Optional[torch.Tensor],
]:
    assert inp.query.ndim in (4, 5)
    query, key, value = inp.query, inp.key, inp.value

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
            BlockDiagonalPaddedKeysMask,
            BlockDiagonalCausalWithOffsetPaddedKeysMask,
            BlockDiagonalGappyKeysMask,
            BlockDiagonalCausalWithOffsetGappyKeysMask,
            BlockDiagonalLocalAttentionPaddedKeysMask,
            BlockDiagonalCausalLocalAttentionPaddedKeysMask,
        ),
    ):
        assert attn_bias.k_seqinfo.seqstart.device == inp.query.device
        cu_seqlen_k = attn_bias.k_seqinfo.seqstart
        cu_seqlen_q = attn_bias.q_seqinfo.seqstart
        max_seqlen_q = attn_bias.q_seqinfo.max_seqlen
        max_seqlen_k = attn_bias.k_seqinfo.max_seqlen
        # All these mask types inherit from classes that have seqlen attribute
        seqused_k = attn_bias.k_seqinfo.seqlen
        assert seqused_k is not None
    else:
        cu_seqlen_k = None
        cu_seqlen_q = None
        seqused_k = None
        max_seqlen_q = None
        max_seqlen_k = None

    if query.ndim == 5:  # GQA
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

    if cu_seqlen_k is not None and query.ndim == 4:
        # Fold to 3D when using varlen
        def fold(x):
            assert x.shape[0] == 1
            x = x.squeeze(0)
            assert x.ndim == 3
            return x

        query = fold(query)
        key = fold(key)
        value = fold(value)

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


def _is_seqlen_q_le_seqlen_k(
    cu_seqlens_q_py: List[int], cu_seqlens_k_py: List[int]
) -> bool:
    if len(cu_seqlens_q_py) < 2 or len(cu_seqlens_k_py) < 2:
        # The seqlens q and k info does not exist on CPU
        return True
    cu_seqlens_q = torch.as_tensor(cu_seqlens_q_py, dtype=torch.int, device="cpu")
    cu_seqlens_k = torch.as_tensor(cu_seqlens_k_py, dtype=torch.int, device="cpu")
    seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    seqlens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    return bool(torch.all(seqlens_k >= seqlens_q).item())


def _is_causal(attn_bias: Union[torch.Tensor, AttentionBias, None]) -> bool:
    return isinstance(
        attn_bias,
        (
            LowerTriangularMask,
            BlockDiagonalCausalMask,
            LowerTriangularFromBottomRightMask,
            BlockDiagonalCausalFromBottomRightMask,
            LowerTriangularFromBottomRightLocalAttentionMask,
            BlockDiagonalCausalLocalAttentionMask,
            BlockDiagonalCausalLocalAttentionFromBottomRightMask,
            BlockDiagonalCausalLocalAttentionPaddedKeysMask,
            BlockDiagonalCausalWithOffsetGappyKeysMask,
            BlockDiagonalCausalWithOffsetPaddedKeysMask,
        ),
    )


def _is_bottom_right(attn_bias: Union[torch.Tensor, AttentionBias, None]) -> bool:
    return isinstance(
        attn_bias,
        (
            LowerTriangularFromBottomRightMask,
            BlockDiagonalCausalFromBottomRightMask,
            LocalAttentionFromBottomRightMask,
            BlockDiagonalCausalLocalAttentionFromBottomRightMask,
            BlockDiagonalCausalWithOffsetPaddedKeysMask,
            BlockDiagonalLocalAttentionPaddedKeysMask,
            BlockDiagonalCausalWithOffsetGappyKeysMask,
            BlockDiagonalCausalLocalAttentionPaddedKeysMask,
        ),
    )


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
            LowerTriangularFromBottomRightLocalAttentionMask,
            BlockDiagonalCausalLocalAttentionPaddedKeysMask,
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


@register_operator
class FwOp(AttentionFwOpBase):
    OPERATOR = _get_operator("_cutlass_blackwell_fmha_forward")
    SUPPORTED_DEVICES: Set[str] = {"cuda"}
    SUPPORTED_DTYPES: Set[torch.dtype] = {torch.bfloat16, torch.float16}
    SUPPORTED_MAX_K = 128
    SUPPORTED_MIN_K = 64
    SUPPORTED_ATTN_BIAS_TYPES: Iterable[Any] = (
        type(None),
        LowerTriangularMask,
        LowerTriangularFromBottomRightMask,
        BlockDiagonalCausalFromBottomRightMask,
        BlockDiagonalMask,
        BlockDiagonalCausalMask,
        BlockDiagonalPaddedKeysMask,
        BlockDiagonalCausalWithOffsetPaddedKeysMask,
        BlockDiagonalGappyKeysMask,
        BlockDiagonalCausalWithOffsetGappyKeysMask,
        BlockDiagonalLocalAttentionPaddedKeysMask,
        BlockDiagonalCausalLocalAttentionPaddedKeysMask,
        LocalAttentionFromBottomRightMask,
        LowerTriangularFromBottomRightLocalAttentionMask,
        BlockDiagonalCausalLocalAttentionMask,
        BlockDiagonalCausalLocalAttentionFromBottomRightMask,
    )
    SUPPORTS_DROPOUT = False
    SUPPORTS_CUSTOM_SCALE = False
    SUPPORTS_DIFFERENT_VALUE_EMBED = False
    SUPPORTS_BMGHK = True
    VARLEN_LSE_PACKED = True
    SUPPORTS_PARTIAL = False
    CUDA_MINIMUM_COMPUTE_CAPABILITY = (10, 0)
    NAME = "cutlassF-blackwell"

    _TEST_K: List[int] = [64, 128]

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:
        reasons = super(FwOp, cls).not_supported_reasons(d)
        if isinstance(d.attn_bias, BlockDiagonalCausalMask):
            (
                _,
                cu_seqlens_q,
                _,
                cu_seqlens_k,
                _,
                _,
            ) = _convert_input_format(d)
            if not _is_seqlen_q_le_seqlen_k(
                d.attn_bias.q_seqinfo.seqstart_py,
                d.attn_bias.k_seqinfo.seqstart_py,
            ):
                reasons.append("seqlens_k must be >= seqlens_q")

        if d.query.ndim < 4 or d.key.ndim < 4 or d.value.ndim < 4:
            reasons.append("Only supports BMHK or BMGHK")

        return reasons

    @classmethod
    def shape_not_supported_reasons(
        cls, Mq: int, Mkv: int, K: int, Kv: int
    ) -> List[str]:
        reasons = super().shape_not_supported_reasons(Mq, Mkv, K, Kv)
        if K not in [64, 128] or Kv not in [64, 128]:
            reasons.append(f"Embed dim {K} not supported")
        elif Mkv != 0 and Mq > Mkv:
            reasons.append(f"Only support Mq ({Mq}) <= Mk ({Mkv})")
        return reasons

    @classmethod
    def apply(
        cls, inp: Inputs, needs_gradient: bool
    ) -> Tuple[torch.Tensor, Optional[Context]]:
        q_shape = inp.query.shape
        (
            inp,
            cu_seqlens_q,
            max_seq_len_q,
            cu_seqlens_k,
            max_seq_len_k,
            seqused_k,
        ) = _convert_input_format(inp)

        window_left, window_right = _window_size(inp.attn_bias)

        if inp.query.numel() > 0 and inp.key.numel() > 0:
            out, lse = cls.OPERATOR(
                q=inp.query,
                k=inp.key,
                v=inp.value,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                seqlen_kv=seqused_k,
                max_seq_len_q=max_seq_len_q,
                max_seq_len_k=max_seq_len_k,
                softmax_scale=inp.scale,
                causal=_is_causal(inp.attn_bias),
                window_left=window_left,
                window_right=window_right,
                bottom_right=_is_bottom_right(inp.attn_bias),
            )
        else:
            out = torch.zeros_like(inp.query)
            if cu_seqlens_q is None:
                assert inp.query.ndim == 4
                B, M, H, K = inp.query.shape
                lse_shape = [B, H, M]
            else:
                assert inp.query.ndim == 3
                M, H, K = inp.query.shape
                lse_shape = [1, H, M]
            lse = torch.zeros(*lse_shape, dtype=torch.float, device=out.device)
        out = out.reshape(q_shape)
        if not needs_gradient:
            return out, None
        return out, Context(out=out, lse=lse)


@register_operator
class BwOp(AttentionBwOpBase):
    __doc__ = FwOp.__doc__

    OPERATOR = _get_operator("_cutlass_blackwell_fmha_backward")

    SUPPORTED_DEVICES = FwOp.SUPPORTED_DEVICES
    SUPPORTED_DTYPES = FwOp.SUPPORTED_DTYPES
    SUPPORTED_MAX_K = FwOp.SUPPORTED_MAX_K
    SUPPORTED_MIN_K = FwOp.SUPPORTED_MIN_K
    SUPPORTED_ATTN_BIAS_TYPES: Iterable[Any] = (
        type(None),
        LowerTriangularMask,
        LowerTriangularFromBottomRightMask,
        BlockDiagonalCausalFromBottomRightMask,
        BlockDiagonalMask,
        BlockDiagonalCausalMask,
        LocalAttentionFromBottomRightMask,
        LowerTriangularFromBottomRightLocalAttentionMask,
        BlockDiagonalCausalLocalAttentionMask,
        BlockDiagonalCausalLocalAttentionFromBottomRightMask,
    )
    SUPPORTS_ATTN_BIAS_GRAD = False
    SUPPORTS_DROPOUT = FwOp.SUPPORTS_DROPOUT
    SUPPORTS_CUSTOM_SCALE = FwOp.SUPPORTS_CUSTOM_SCALE
    SUPPORTS_DIFFERENT_VALUE_EMBED = False
    SUPPORTS_BMGHK = False
    VARLEN_LSE_PACKED = True
    SUPPORTS_PARTIAL = False
    CUDA_MINIMUM_COMPUTE_CAPABILITY = (10, 0)
    NAME = "cutlassB-blackwell"

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:
        reasons = super(BwOp, cls).not_supported_reasons(d)
        if isinstance(d.attn_bias, BlockDiagonalCausalMask):
            (
                _,
                cu_seqlens_q,
                _,
                cu_seqlens_k,
                _,
                _,
            ) = _convert_input_format(d)
            if not _is_seqlen_q_le_seqlen_k(
                d.attn_bias.q_seqinfo.seqstart_py,
                d.attn_bias.k_seqinfo.seqstart_py,
            ):
                reasons.append("seqlens_k must be >= seqlens_q")

        if d.query.ndim != 4 or d.key.ndim != 4 or d.value.ndim != 4:
            reasons.append("Only supports BMHK format")

        return reasons

    @classmethod
    def shape_not_supported_reasons(
        cls, Mq: int, Mkv: int, K: int, Kv: int
    ) -> List[str]:
        reasons = super().shape_not_supported_reasons(Mq, Mkv, K, Kv)
        if K not in [64, 128]:
            reasons.append(f"Embed dim {K} not supported")
        elif Mkv != 0 and Mq > Mkv:
            reasons.append(f"Only support Mq ({Mq}) <= Mk ({Mkv})")
        elif Mq < 8:
            reasons.append(f"Only support Mq ({Mq}) >= 8")
        return reasons

    @classmethod
    def apply(cls, ctx: Context, inp: Inputs, grad: torch.Tensor) -> Gradients:
        assert inp.query.ndim == 4
        dq_shape, dk_shape, dv_shape = inp.query.shape, inp.key.shape, inp.value.shape
        (
            inp,
            cu_seqlens_q,
            max_seq_len_q,
            cu_seqlens_k,
            max_seq_len_k,
            _,
        ) = _convert_input_format(inp)

        window_left, window_right = _window_size(inp.attn_bias)

        is_varlen = cu_seqlens_q is not None
        if is_varlen:

            def fold(x):
                assert x.shape[0] == 1
                x = x.squeeze(0)
                assert x.ndim == 3
                return x

            grad = fold(grad)
            ctx.out = fold(ctx.out)

        if inp.query.numel() and inp.key.numel():
            grads = Gradients(
                *cls.OPERATOR(
                    dout=grad,
                    q=inp.query,
                    k=inp.key,
                    v=inp.value,
                    out=ctx.out,
                    softmax_lse=ctx.lse,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seq_len_q=max_seq_len_q,
                    max_seq_len_k=max_seq_len_k,
                    causal=_is_causal(inp.attn_bias),
                    window_left=window_left,
                    window_right=window_right,
                    bottom_right=_is_bottom_right(inp.attn_bias),
                )
            )
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
