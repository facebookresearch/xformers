# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import TYPE_CHECKING, Any, List, Optional, Set, Tuple

import torch

from ..common import _has_triton21, register_operator
from .attn_bias import BlockDiagonalCausalWithOffsetPaddedKeysMask
from .common import AttentionFwOpBase, Context, Inputs, check_lastdim_alignment_stride1


def _strides(x: torch.Tensor, *stride_names: str):
    assert x.ndim == len(stride_names)
    return {f"stride_{s}": x.stride(i) for i, s in enumerate(stride_names)}


if TYPE_CHECKING or _has_triton21():
    import triton
    import triton.language as tl

    from xformers.triton.vararg_kernel import VAR_ARGS_ARRAY, unroll_varargs

    @triton.jit
    def _fwd_kernel_splitK(
        Q,
        K,
        V,
        sm_scale,
        Out_splitK,  # [B, H, split_k, Mq, K]
        Metadata,  # [B, H, 2, split_k, M_ceil] contains [mi, li]
        Seq_len,
        stride_qz,
        stride_qm,
        stride_qg,
        stride_qh,
        stride_qk,
        stride_kz,
        stride_kn,
        stride_kg,
        stride_kh,
        stride_kk,
        stride_vz,
        stride_vn,
        stride_vg,
        stride_vh,
        stride_vk,
        stride_osk_zhg,
        stride_osk_s,
        stride_osk_m,
        stride_osk_k,
        stride_mzhg,
        stride_m2,
        stride_ms,
        stride_mm,
        Z,
        N_CTX_Q,
        N_CTX_K,
        BLOCK_N_PER_SPLIT,
        H: tl.constexpr,
        G: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BOUNDS_CHECKS_N: tl.constexpr,
        USE_SEQ_LEN: tl.constexpr,
        PACKED_PER_VAL: tl.constexpr = 1,
        N_GROUPS: tl.constexpr = 1,
    ):
        """This kernel can accept non-quantized or int4-quantized keys/values.
        PACKED_PER_VAL determines the quantization type:
            - PACKED_PER_VAL == 1 means no quantization
            - PACKED_PER_VAL == 8 means 4-bit quantization (8 packed quantized values inside one int32)
        For the quantized case K/V should be int32 tensors.
        Quantization can be row-wise (when N_GROUPS = 1) or group-wise with N_GROUPS = 2, 4, or 8.
        Quantization coefficients are stored at the beginning of the row along the last dimension of K/V
        So K[B, H, M, :] has a form
        [   quant_coef0, quant_coef1, ...|
            group0_quant_value0, group0_quant_value1,... |
            group1_quant_value0, group1_quant_value1,...]
        where each quant_coef is an int32 which should be interpreted as 2 packed float16: scale and offset.

        Note: this kernel needs to be processed by xformers.triton.vararg_kernel.unroll_varargs
        before compilation. That will unroll variables marked with "VAR_ARGS_ARRAY" into lists.
        See how FwOp.apply does it below.
        """
        tl.static_assert(
            (PACKED_PER_VAL == 1 and tl.constexpr(K.dtype.element_ty != tl.int32))
            or (PACKED_PER_VAL == 8 and tl.constexpr(K.dtype.element_ty == tl.int32)),
            f"Only 4-bit quantization is supported, K/V should have dtype int32 in "
            f"the quantized case: {PACKED_PER_VAL=} {tl.constexpr(K.dtype)=} {tl.constexpr(K.dtype.element_ty)=}",
        )
        tl.static_assert(
            (((N_GROUPS == 1 or N_GROUPS == 2) or N_GROUPS == 4) or N_GROUPS == 8),
            "Number of quantization groups can be 1 (row-wise quantization), 2, 4, or 8.",
        )

        QUANTIZED: tl.constexpr = PACKED_PER_VAL > 1
        PACKED_D_PER_GROUP: tl.constexpr = BLOCK_DMODEL // PACKED_PER_VAL // N_GROUPS
        D_PER_GROUP: tl.constexpr = BLOCK_DMODEL // N_GROUPS

        start_m = tl.program_id(0)
        off_zhg = tl.program_id(1)
        off_z = off_zhg // (H * G)
        off_h = (off_zhg // G) % H
        off_g = off_zhg % G
        splitk_idx = tl.program_id(2)

        lo = splitk_idx * BLOCK_N_PER_SPLIT
        if USE_SEQ_LEN:
            kv_len = tl.load(Seq_len + off_z)
        else:
            kv_len = N_CTX_K
        hi = tl.minimum((splitk_idx + 1) * BLOCK_N_PER_SPLIT, kv_len)

        Q_block_ptr = tl.make_block_ptr(
            base=Q + off_h * stride_qh + off_z * stride_qz + off_g * stride_qg,
            shape=(N_CTX_Q, D_PER_GROUP),
            strides=(stride_qm, stride_qk),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, D_PER_GROUP),
            order=(1, 0),
        )

        k_base = K + off_h * stride_kh + off_z * stride_kz + off_g * stride_kg
        # Additional shift by 1 along the last dimension in the quantized case, since
        # the first element along that dim contains packed quantization coefficients.
        K_block_ptr = tl.make_block_ptr(
            base=k_base + stride_kk * QUANTIZED * N_GROUPS,
            shape=(PACKED_D_PER_GROUP, hi),
            strides=(stride_kk, stride_kn),
            offsets=(0, lo),
            block_shape=(PACKED_D_PER_GROUP, BLOCK_N),
            order=(0, 1),
        )
        v_base = V + off_h * stride_vh + off_z * stride_vz + off_g * stride_vg
        V_block_ptr = tl.make_block_ptr(
            base=v_base + stride_vk * QUANTIZED * N_GROUPS,
            shape=(hi, PACKED_D_PER_GROUP),
            strides=(stride_vn, stride_vk),
            offsets=(lo, 0),
            block_shape=(BLOCK_N, PACKED_D_PER_GROUP),
            order=(1, 0),
        )

        if QUANTIZED:
            # Pointers to quantization coefficients. Even those they are 1D,
            # we have to use block pointers, since usual pointers
            # don't support boundary checks
            K_scale_shift_block_ptr = tl.make_block_ptr(
                base=k_base,
                shape=(1, hi),
                strides=(stride_kk, stride_kn),
                offsets=(0, lo),
                block_shape=(1, BLOCK_N),
                order=(0, 1),
            )
            V_scale_shift_block_ptr = tl.make_block_ptr(
                base=v_base,
                shape=(hi, 1),
                strides=(stride_vn, stride_vk),
                offsets=(lo, 0),
                block_shape=(BLOCK_N, 1),
                order=(1, 0),
            )
        else:
            K_scale_shift_block_ptr = None
            V_scale_shift_block_ptr = None

        # initialize pointer to m and l
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

        # Before compilation, this kernel will be processed by xformers.triton.vararg_kernel.unroll_varargs.
        # That turns tensors annotated as the one below into lists of tensors of length N_GROUPS.
        # This is a solution for Triton native lack of support for lists of tensors.
        acc: "VAR_ARGS_ARRAY"  # noqa: F821

        for i in range(len(acc)):  # noqa: F821
            acc[i] = tl.zeros([BLOCK_M, D_PER_GROUP], dtype=tl.float32)  # noqa: F821
        # scale sm_scale by log_2(e) and use
        # 2^x instead of exp in the loop because CSE and LICM
        # don't work as expected with `exp` in the loop
        qk_scale = sm_scale * 1.44269504
        # load q: it will stay in SRAM throughout
        q: "VAR_ARGS_ARRAY"  # noqa: F821
        for i in range(len(acc)):  # noqa: F821
            q[i] = tl.load(  # noqa: F821
                tl.advance(Q_block_ptr, (0, i * D_PER_GROUP)), boundary_check=(0,)
            )
        # loop over k, v and update accumulator
        for start_n in range(lo, hi, BLOCK_N):
            k: "VAR_ARGS_ARRAY"  # noqa: F821
            v: "VAR_ARGS_ARRAY"  # noqa: F821
            for i in range(len(acc)):  # noqa: F821
                k[i], v[i] = load_dequantize_k_v_group(  # noqa: F821
                    K_block_ptr,
                    V_block_ptr,
                    K_scale_shift_block_ptr,
                    V_scale_shift_block_ptr,
                    BOUNDS_CHECKS_N,
                    PACKED_PER_VAL,
                    PACKED_D_PER_GROUP,
                    Q.dtype.element_ty,
                    i,
                )

            # -- compute qk ---
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            for i in range(len(acc)):  # noqa: F821
                qk += tl.dot(q[i], k[i])  # noqa: F821
            qk *= qk_scale

            # TODO: This is slow, and only needed at the last iteration.
            # Maybe we can unroll the last iteration instead?
            if BOUNDS_CHECKS_N:
                qk = tl.where(tl.arange(0, BLOCK_N) < hi - start_n, qk, float("-inf"))
            # -- compute scaling constant ---
            m_i_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.math.exp2(m_i - m_i_new)
            p = tl.math.exp2(qk - m_i_new[:, None])

            # -- update m_i and l_i --
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_i_new
            p = p.to(Q.dtype.element_ty)

            # -- scale and update acc --
            for i in range(len(acc)):  # noqa: F821
                acc[i] *= alpha[:, None]  # noqa: F821
                acc[i] += tl.dot(p, v[i])  # noqa: F821
            # update pointers
            K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
            V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
            if PACKED_PER_VAL > 1:
                K_scale_shift_block_ptr = tl.advance(
                    K_scale_shift_block_ptr, (0, BLOCK_N)
                )
                V_scale_shift_block_ptr = tl.advance(
                    V_scale_shift_block_ptr, (BLOCK_N, 0)
                )

        # write back O
        O_block_ptr = tl.make_block_ptr(
            base=Out_splitK + off_zhg * stride_osk_zhg + splitk_idx * stride_osk_s,
            shape=(N_CTX_Q, D_PER_GROUP),
            strides=(stride_osk_m, 1),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, D_PER_GROUP),
            order=(1, 0),
        )
        for i in range(len(acc)):  # noqa: F821
            tl.store(
                tl.advance(O_block_ptr, (0, i * D_PER_GROUP)),
                acc[i],  # noqa: F821
                boundary_check=(0,),
            )
        # Write metadata for split-K reduction
        Metadata_ptr = (
            Metadata
            + off_zhg * stride_mzhg
            + splitk_idx * stride_ms
            + start_m * BLOCK_M
            + tl.arange(0, BLOCK_M)
        )
        tl.store(Metadata_ptr, m_i)
        tl.store(Metadata_ptr + stride_m2, l_i)

    @triton.jit
    def load_dequantize_k_v_group(
        K_block_ptr,
        V_block_ptr,
        K_scale_shift_block_ptr,
        V_scale_shift_block_ptr,
        BOUNDS_CHECKS_N: tl.constexpr,
        PACKED_PER_VAL: tl.constexpr,
        PACKED_D_PER_GROUP: tl.constexpr,
        dtype: tl.constexpr,
        group_id: tl.constexpr,
    ):
        """Load K/V for a given block. In case of int4-quantized K/V, dequantize them after loading.
        If quantization is group-wise, use group_id to advance the pointers to the current group.
        """
        # Advance to the current quantization group
        K_block_ptr = tl.advance(K_block_ptr, (PACKED_D_PER_GROUP * group_id, 0))
        V_block_ptr = tl.advance(V_block_ptr, (0, PACKED_D_PER_GROUP * group_id))

        # -- load k, v --
        k = tl.load(K_block_ptr, boundary_check=(1,) if BOUNDS_CHECKS_N else ())
        v = tl.load(V_block_ptr, boundary_check=(0,) if BOUNDS_CHECKS_N else ())

        if PACKED_PER_VAL > 1:
            # K/V are quantized, load quantization coefficients and dequantize

            K_scale_shift_block_ptr = tl.advance(K_scale_shift_block_ptr, (group_id, 0))
            V_scale_shift_block_ptr = tl.advance(V_scale_shift_block_ptr, (0, group_id))

            k_scale_shift = tl.load(
                K_scale_shift_block_ptr, boundary_check=(1,) if BOUNDS_CHECKS_N else ()
            )
            v_scale_shift = tl.load(
                V_scale_shift_block_ptr, boundary_check=(0,) if BOUNDS_CHECKS_N else ()
            )

            k_scale, k_shift = cast_uint32_to_half2(k_scale_shift)
            v_scale, v_shift = cast_uint32_to_half2(v_scale_shift)
            v = dequantize(v, v_scale, v_shift, PACKED_PER_VAL).to(dtype)
            k_t = dequantize(
                tl.trans(k),
                tl.trans(k_scale),
                tl.trans(k_shift),
                PACKED_PER_VAL,
            ).to(dtype)
            k = tl.trans(k_t)
        return k, v

    @triton.jit
    def cast_uint32_to_half2(scale_shift):
        """Extract two float16 packed into one int32"""
        scale = scale_shift & 0xFFFF
        shift = scale_shift >> 16
        scale = scale.to(tl.uint16).to(tl.float16, bitcast=True)
        shift = shift.to(tl.uint16).to(tl.float16, bitcast=True)
        return scale, shift

    @triton.jit
    def dequantize(
        x_,
        scale,
        shift,
        PACKED_PER_VAL: tl.constexpr = 8,
    ):
        """PACKED_PER_VAL is the number of values packed into each element x_.
        For example, for int4 quantization and x_ of type int32, PACKED_PER_VAL is 8.
        """

        # Axis along which offsets are applied matters here
        # It would be natural to have offsets in shape (BLOCK_N, D // PACKED_PER_VAL, PACKED_PER_VAL)
        # and expand K/V to that shape before applying offsets
        # However, Triton for some reason considers dim=1 as contiguous when doing tl.view below, and not dim=2
        # Note that tl.view doesn't guarantee the order of elements in the result - thus the code below depends
        # on the implementation details which might change in the future.
        # Ideally we would like to use tl.reshape, but it's not implemented yet.
        # See https://github.com/openai/triton/blob/9055af1a5dadc576804b38dd77ee91dc42af0bf7/python/triton/language/semantic.py#L541 # noqa: E501

        # x_ : (BLOCK_N, D // PACKED_PER_VAL)
        # scale: (BLOCK_N, 1)
        # offsets: (PACKED_PER_VAL,)
        BLOCK_N: tl.constexpr = x_.shape[0]
        BLOCK_DMODEL_PACKED: tl.constexpr = x_.shape[1]
        offsets = tl.arange(0, PACKED_PER_VAL) * 4
        quant_offset = (
            x_[:, None, :] >> offsets[None, :, None]
        )  # (BLOCK_N, PACKED_PER_VAL, D // PACKED_PER_VAL)

        quant_offset = tl.view(
            quant_offset, (BLOCK_N, BLOCK_DMODEL_PACKED * PACKED_PER_VAL)
        )
        # Trick - instead of converting int4 to float16 we view it as float16
        # and then multiply by 32768 * 512 == 2**24
        quant_offset = (quant_offset & 0xF).to(tl.uint16).to(tl.float16, bitcast=True)
        quant_offset = (quant_offset * 32768.0).to(tl.float16)
        scale_512 = scale * 512

        dequant = quant_offset * scale_512 + shift
        return dequant

    @triton.jit
    def _splitK_reduce(
        Out_splitK,  # [B, H, split_k, Mq, K]
        Metadata,  # [B, H, 2, split_k, M_ceil] contains [mi, li]
        Out,  # [B, H, M, K]
        LSE,  # [B, H, M]
        split_k,
        stride_osk_zhg,
        stride_osk_s,
        stride_osk_m,
        stride_osk_k,
        stride_mzhg,
        stride_m2,
        stride_ms,
        stride_mm,
        stride_oz,
        stride_oh,
        stride_og,
        stride_om,
        stride_ok,
        stride_lse_zhg,
        stride_lse_m,
        BLOCK_SIZE: tl.constexpr,
        H: tl.constexpr,
        G: tl.constexpr,
    ):
        off_zhg = tl.program_id(0)
        off_z = off_zhg // (H * G)
        off_h = (off_zhg // G) % H
        off_g = off_zhg % G
        off_m = tl.program_id(1)

        Out_splitK_ptr = (
            Out_splitK
            + stride_osk_zhg * off_zhg
            + stride_osk_m * off_m
            + tl.arange(0, BLOCK_SIZE)
        )
        Metadata_ptr = Metadata + stride_mzhg * off_zhg + off_m
        m = tl.load(Metadata_ptr)
        l_sum = tl.load(Metadata_ptr + stride_m2)
        acc = tl.load(Out_splitK_ptr)

        for split_k_idx in range(1, split_k):
            Metadata_ptr = Metadata_ptr + stride_ms
            Out_splitK_ptr = Out_splitK_ptr + stride_osk_s

            m_k = tl.load(Metadata_ptr)
            l_k = tl.load(Metadata_ptr + stride_m2)
            acc_k = tl.load(Out_splitK_ptr)

            m_new = tl.maximum(m, m_k)
            if m_k < m:
                # Scale incoming values
                alpha = tl.math.exp2(m_k - m_new)
                acc_k = acc_k * alpha
                l_k = l_k * alpha
            else:
                # Scale our values
                alpha = tl.math.exp2(m - m_new)
                acc = acc * alpha
                l_sum = l_sum * alpha

            m = m_new
            l_sum = l_sum + l_k
            acc = acc + acc_k

        acc = acc / l_sum
        Out_ptr = (
            Out
            + stride_oz * off_z
            + stride_oh * off_h
            + stride_og * off_g
            + stride_om * off_m
            + tl.arange(0, BLOCK_SIZE)
        )
        tl.store(Out_ptr, acc)

        l_ptrs = LSE + off_zhg * stride_lse_zhg + off_m
        tl.store(l_ptrs, (m + tl.math.log2(l_sum)) / 1.44269504)

else:
    _fwd_kernel_splitK = None
    _splitK_reduce = None


@register_operator
class FwOp(AttentionFwOpBase):
    """Flash-Attention with Split-K. Supports fused int-4 K/V quantization.
    Quantized path will be taken if input K/V have type int32.

    Quantization can be row-wise or group-wise (when cls.NUM_GROUPS > 1) along
    the last dimension of K and V. Currently 1, 2, 4, or 8 groups per row are supported.
    Quantization coefficients (scale and shift) are represented as two
    float16 constants per group, packed into int32. Quantization coefficients of
    all groups are placed at the beginning of the row. So, if unquantized K/V have head
    dimension D, the quantized versions have head dimension D // 8 + NUM_GROUPS
    and dtype int32.
    Pseudocode for dequantizing one row can look like:
    group_size = D // 8
    for i in range(NUM_GROUPS):
        group_start = NUM_GROUPS + i * group_size
        group_quant = K[..., group_start: group_start + group_size]
        scale, shift = unpack_int32_into_float16x2(group_quant[0])
        group_dequant = group_quant[..., 1:] * scale + shift
    ...

    """

    OPERATOR = _fwd_kernel_splitK
    SUPPORTED_DEVICES = {"cuda"}
    CUDA_MINIMUM_COMPUTE_CAPABILITY = (8, 0)
    SUPPORTED_DTYPES = {
        torch.half,
        torch.bfloat16,
    }  # Those are dtypes of Q. In the quantized case K/V has dtype int32
    SUPPORTED_MAX_K = 128
    SUPPORTED_ATTN_BIAS_TYPES: Set[Any] = {
        type(None),
        BlockDiagonalCausalWithOffsetPaddedKeysMask,
    }
    SUPPORTS_DROPOUT = False
    SUPPORTS_CUSTOM_SCALE = True
    SUPPORTS_BMGHK = True
    NAME = "triton_splitKF"

    SPLIT_K: Optional[int] = None
    BLOCK_M = 16
    BLOCK_N = 64

    NUM_GROUPS = 1  # Default quantization is row-wise

    @classmethod
    def shape_not_supported_reasons(
        cls, Mq: int, Mkv: int, K: int, Kv: int
    ) -> List[str]:
        reasons = super().shape_not_supported_reasons(Mq, Mkv, K, Kv)
        if K not in {16, 32, 64, 128}:
            reasons.append(f"Embed dim {K} not supported")
        return reasons

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:
        reasons = super(FwOp, cls).not_supported_reasons(d)
        check_lastdim_alignment_stride1(reasons, "query", d.query, 8)
        if d.key.dtype != torch.int32:
            check_lastdim_alignment_stride1(reasons, "key", d.key, 8)
            check_lastdim_alignment_stride1(reasons, "value", d.value, 8)
        if cls.OPERATOR is None:
            reasons.append("triton is not available")
        if d.device.type == "cuda":
            # Has only been tested on 8.0 / 9.0.
            if torch.cuda.get_device_capability(d.device) < (8, 0):
                reasons.append(
                    "requires GPU with sm80 minimum compute capacity, e.g., A100/H100/L4"
                )

        q_len = d.query.shape[1]
        if isinstance(d.attn_bias, BlockDiagonalCausalWithOffsetPaddedKeysMask):
            seqinfo = d.attn_bias.q_seqinfo
            if q_len != seqinfo.seqstart_py[-1]:
                reasons.append(
                    f"Expected total {seqinfo.seqstart_py[-1]} queries not {q_len}"
                )
            q_len = seqinfo.min_seqlen
            if q_len != seqinfo.max_seqlen:
                reasons.append(
                    "Variable query len is not supported in the presence of causal mask."
                )

        if d.key.ndim in [4, 5] and d.key.shape[-2] != 1:
            if d.key.stride(-2) == 0 and d.value.stride(-2) == 0 and q_len > 1:
                reasons.append("multiquery is only supported with query seqlen=1")

        if d.attn_bias is not None and q_len > 1:
            reasons.append(
                "query with seqlen > 1 is not supported in the presence of causal mask"
            )
        return reasons

    @classmethod
    def get_split_k(cls, B: int, H: int, Mk: int) -> int:
        """Heuristic for the number of splits"""
        bh = max(B * H, 1)  # NOTE: Handle B*h=0 case
        split_k = max(Mk, 1024) // bh
        max_chunk_size = 64 if Mk <= 512 and bh <= 64 else 128
        while split_k > 0 and Mk / split_k < max_chunk_size:
            split_k = split_k // 2
        split_k = min(split_k, 64)
        split_k = max(split_k, 1)
        return split_k

    @classmethod
    def apply(
        cls, inp: Inputs, needs_gradient: bool
    ) -> Tuple[torch.Tensor, Optional[Context]]:
        attn_bias = inp.attn_bias
        seq_len = None
        q, k, v = inp.get_qkv_in_bmghk()

        if attn_bias is not None:
            assert isinstance(attn_bias, BlockDiagonalCausalWithOffsetPaddedKeysMask)
            # TODO: do we really need to do this cast? seems fishy but
            # I just copied it from the decoder.py
            attn_bias.k_seqinfo.to(inp.query.device)
            attn_bias.q_seqinfo.to(inp.query.device)
            seq_len = attn_bias.k_seqinfo.seqlen
            B = len(seq_len)
            G, H, Kq = q.shape[-3:]
            Kkv = v.shape[-1]

            # assume kv has been padded
            q = q.reshape(B, -1, G, H, Kq)
            k = k.reshape(B, -1, G, H, Kkv)
            v = v.reshape(B, -1, G, H, Kkv)

        # Transpose in the case of MQA/GQA
        mqa_swap_seqlen_head = False
        if k.shape[3] > 1 and k.stride(3) == 0 and v.stride(3) == 0:
            mqa_swap_seqlen_head = True
            assert q.shape[1] == 1
            q = q.transpose(1, 3)
            k = k[:, :, :, :1]
            v = v[:, :, :, :1]

        if k.dtype == torch.int32:
            # Quantized K/V
            PACKED_PER_VAL = 8
            Lk = (k.shape[-1] - cls.NUM_GROUPS) * 8
        else:
            Lk = k.shape[-1]
            PACKED_PER_VAL = 1

        B, Mk, G, H, Kkv = k.shape
        B, M, G, H, Kq = q.shape
        assert Lk == Kq, f"Keys have head dim {Lk} but queries have head dim {Kq}"

        BLOCK_M = cls.BLOCK_M
        BLOCK_N = cls.BLOCK_N
        if cls.SPLIT_K is not None:
            split_k = cls.SPLIT_K
        else:
            # Use heuristics
            split_k = cls.get_split_k(B, H, Mk)

        M_ceil = (M + BLOCK_M - 1) // BLOCK_M * BLOCK_M
        o_splitk = torch.empty(
            [B * G * H, split_k, M_ceil, Kq], dtype=torch.float32, device=q.device
        )
        metadata = torch.empty(
            [B * G * H, 2, split_k, M_ceil], dtype=torch.float32, device=q.device
        )
        lse = torch.empty((B * G * H, M), device=q.device, dtype=torch.float32)
        grid = (triton.cdiv(M, BLOCK_M), B * G * H, split_k)

        num_warps = 2
        split_size = (Mk + split_k - 1) // split_k
        use_seq_len = seq_len is not None
        _fwd_kernel_splitK_unrolled = unroll_varargs(
            _fwd_kernel_splitK, N=cls.NUM_GROUPS if PACKED_PER_VAL > 1 else 1
        )

        _fwd_kernel_splitK_unrolled[grid](
            Q=q,
            K=k,
            V=v,
            sm_scale=inp.scale_float,
            Out_splitK=o_splitk,
            Metadata=metadata,
            Seq_len=seq_len,
            **_strides(q, "qz", "qm", "qg", "qh", "qk"),
            **_strides(k, "kz", "kn", "kg", "kh", "kk"),
            **_strides(v, "vz", "vn", "vg", "vh", "vk"),
            **_strides(o_splitk, "osk_zhg", "osk_s", "osk_m", "osk_k"),
            **_strides(metadata, "mzhg", "m2", "ms", "mm"),
            Z=B,
            H=H,
            G=G,
            N_CTX_Q=M,
            N_CTX_K=Mk,
            BLOCK_N_PER_SPLIT=split_size,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=Lk,
            BOUNDS_CHECKS_N=(split_size % BLOCK_N) > 0 or use_seq_len,
            USE_SEQ_LEN=use_seq_len,
            num_warps=num_warps,
            num_stages=1,
            PACKED_PER_VAL=PACKED_PER_VAL,
            N_GROUPS=cls.NUM_GROUPS if PACKED_PER_VAL > 1 else 1,
        )

        if mqa_swap_seqlen_head:
            out = torch.empty(
                (B, H, G, M, Kq), device=q.device, dtype=q.dtype
            ).transpose(1, 3)
        else:
            out = torch.empty((B, M, G, H, Kq), device=q.device, dtype=q.dtype)

        # Merge together
        grid = (B * G * H, M, 1)
        _splitK_reduce[grid](
            o_splitk,
            metadata,
            out,
            lse,
            split_k=split_k,
            **_strides(o_splitk, "osk_zhg", "osk_s", "osk_m", "osk_k"),
            **_strides(metadata, "mzhg", "m2", "ms", "mm"),
            **_strides(out, "oz", "om", "og", "oh", "ok"),
            **_strides(lse, "lse_zhg", "lse_m"),
            BLOCK_SIZE=out.shape[-1],
            G=G,
            H=H,
            # TODO: Tune num_warps
        )
        lse = lse.reshape([B, G, H, M])
        if mqa_swap_seqlen_head:
            # H/M dimensions have been swapped
            out = out.transpose(1, 3)
            lse = lse.transpose(2, 3)
        if inp.query.ndim == 4:
            # BMGHK -> BMHK
            assert G == 1
            out = out[:, :, 0]
            lse = lse[:, 0]
        if Mk == 0:
            out.zero_()

        return out, Context(out=out, lse=lse)


class FwOp_S1(FwOp):
    SPLIT_K = 1
    NAME = "triton_splitK1"


class FwOp_S2(FwOp):
    SPLIT_K = 2
    NAME = "triton_splitK2"


class FwOp_S4(FwOp):
    SPLIT_K = 4
    NAME = "triton_splitK4"


class FwOp_S8(FwOp):
    SPLIT_K = 8
    NAME = "triton_splitK8"


class FwOp_S16(FwOp):
    SPLIT_K = 16
    NAME = "triton_splitK16"


class FwOp_S32(FwOp):
    SPLIT_K = 32
    NAME = "triton_splitK32"


class FwOp_S64(FwOp):
    SPLIT_K = 64
    NAME = "triton_splitK64"


class FwOp_S128(FwOp):
    SPLIT_K = 128
    NAME = "triton_splitK128"
