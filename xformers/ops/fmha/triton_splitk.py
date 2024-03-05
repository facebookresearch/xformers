# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import functools
import sys
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Type

import torch

from ..common import _has_triton21, register_operator
from .attn_bias import (
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
)
from .common import AttentionFwOpBase, Context, Inputs, check_lastdim_alignment_stride1


def _strides(x: Optional[torch.Tensor], *stride_names: str):
    if x is None:
        return {f"stride_{name}": None for name in stride_names}
    assert x.ndim == len(stride_names)
    return {f"stride_{name}": s for name, s in zip(stride_names, x.stride())}


AUTOTUNER_KEY = [
    "Z",
    "H",
    "G",
    "N_CTX_Q",
    "N_CTX_K",
    "BLOCK_DMODEL",
    "PACKED_PER_VAL",
    "N_GROUPS",
    "BLOCK_N_PER_SPLIT",
]

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
        LSE_splitk,  # [B, H, split_k, Mq]
        block_tables,
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
        stride_osk_z,
        stride_osk_hg,
        stride_osk_s,
        stride_osk_m,
        stride_osk_k,
        stride_lsek_z,
        stride_lsek_hg,
        stride_lsek_s,
        stride_lsek_m,
        stride_blocktablesz,
        stride_blocktablesl,
        kv_cache_blocks_per_row: tl.constexpr,
        Z: tl.constexpr,
        N_CTX_Q: tl.constexpr,  # The number of queries
        N_CTX_K: tl.constexpr,
        BLOCK_N_PER_SPLIT: tl.constexpr,
        H: tl.constexpr,
        G: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        USE_SEQ_LEN: tl.constexpr,
        PACKED_PER_VAL: tl.constexpr,
        N_GROUPS: tl.constexpr,
        # It's important that BOUNDS_CHECKS_N, BLOCK_M, BLOCK_N come at the end of
        # the argument list, since they are provided by the heuristics/autotune decorator.
        # Otherwise Triton throws IndexError
        BOUNDS_CHECKS_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        IS_SPLITK: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        NUM_QUERIES_CAUSAL: tl.constexpr,  # The N_CTX_Q queries are from this many sequence positions
        USE_PAGED_ATTENTION: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        WRITE_LSE: tl.constexpr,
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

        Set IS_SPLITK=False to indicate the MHA result should be written directly.
        No metadata will be written.
        """
        internal_dtype = (
            tl.float64 if Out_splitK.dtype.element_ty is tl.float64 else tl.float32
        )
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
        off_hg = off_zhg % (H * G)
        off_h = off_hg // G
        off_g = off_hg % G
        splitk_idx = tl.program_id(2)

        if USE_SEQ_LEN:
            kv_len = tl.load(Seq_len + off_z)
        else:
            kv_len = N_CTX_K

        k_base = K + off_h * stride_kh + off_g * stride_kg
        v_base = V + off_h * stride_vh + off_g * stride_vg

        # Boundaries of split-k chunk
        chunk_hi = (splitk_idx + 1) * BLOCK_N_PER_SPLIT
        chunk_lo = splitk_idx * BLOCK_N_PER_SPLIT
        # For paged attention case K/V_block_ptr are defined inside the loop
        # whereas for non-paged case they are defined before the loop.
        if PAGE_SIZE > 0:
            # Page contains several blocks
            BLOCKS_IN_PAGE: tl.constexpr = PAGE_SIZE // BLOCK_N
            # Align boundaries of split-k chunk to page boundaries
            # In the last chunk, shift hi to the right, in the other chunks, shift it to the left
            is_last_chunk = splitk_idx == tl.num_programs(2) - 1
            shift = 0
            if is_last_chunk:
                shift = PAGE_SIZE - 1
            lo = (chunk_lo // PAGE_SIZE) * PAGE_SIZE
            hi = ((chunk_hi + shift) // PAGE_SIZE) * PAGE_SIZE
            hi = tl.minimum(hi, kv_len)
            block_table = block_tables + stride_blocktablesz * off_z
            # Offset in integer blocks
            logical_block_idx = lo // BLOCK_N
        else:
            lo = chunk_lo
            hi = tl.minimum(chunk_hi, kv_len)
            k_base += off_z * stride_kz
            v_base += off_z * stride_vz
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

        Q_block_ptr = tl.make_block_ptr(
            base=Q + off_h * stride_qh + off_z * stride_qz + off_g * stride_qg,
            shape=(N_CTX_Q, D_PER_GROUP),
            strides=(stride_qm, stride_qk),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, D_PER_GROUP),
            order=(1, 0),
        )

        # initialize pointer to m and l
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

        # Before compilation, this kernel will be processed by xformers.triton.vararg_kernel.unroll_varargs.
        # That turns tensors annotated as the one below into lists of tensors of length N_GROUPS.
        # This is a solution for Triton native lack of support for lists of tensors.
        acc: "VAR_ARGS_ARRAY"  # noqa: F821

        for i in range(len(acc)):  # noqa: F821
            acc[i] = tl.zeros(  # noqa: F821
                [BLOCK_M, D_PER_GROUP], dtype=internal_dtype
            )
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

        if IS_CAUSAL:
            # Why does the masking conditon below work as a causal mask?
            # Assuming num_queries <= BLOCK_M:
            # kv_pos = kv_start + range(0, BLOCK_N)
            # q_offset = start_m * BLOCK_M + range(0, BLOCK_M)
            # q_pos = kv_start + kv_len - num_queries + q_offset % num_queries
            # mask = q_pos - kv_pos >= 0
            # So the final masking condition is:
            #   range(0, BLOCK_M) % num_queries - range(0, BLOCK_N) >= num_queries - kv_len

            q_offset = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            diag_idx = (q_offset[:, None] % NUM_QUERIES_CAUSAL) - tl.arange(0, BLOCK_N)[
                None, :
            ]
            diag_idx_shifted = tl.constexpr(diag_idx - NUM_QUERIES_CAUSAL + kv_len)

        # loop over k, v and update accumulator
        for start_n in range(lo, hi, BLOCK_N):
            if PAGE_SIZE > 0:
                # Offset in integer blocks from the beginning of the page
                block_offset_in_page = logical_block_idx % BLOCKS_IN_PAGE
                # Offset in integer pages
                logical_page_idx = logical_block_idx // BLOCKS_IN_PAGE
                physical_page_idx = tl.load(
                    block_table + stride_blocktablesl * logical_page_idx
                ).to(tl.int32)
                offset = physical_page_idx * PAGE_SIZE + block_offset_in_page * BLOCK_N

                current_block_size = min(hi - start_n, BLOCK_N)
                K_block_ptr = tl.make_block_ptr(
                    base=k_base + stride_kk * QUANTIZED * N_GROUPS,
                    shape=(PACKED_D_PER_GROUP, offset + current_block_size),
                    strides=(stride_kk, stride_kn),
                    offsets=(0, offset),
                    block_shape=(PACKED_D_PER_GROUP, BLOCK_N),
                    order=(0, 1),
                )
                V_block_ptr = tl.make_block_ptr(
                    base=v_base + stride_vk * QUANTIZED * N_GROUPS,
                    shape=(offset + current_block_size, PACKED_D_PER_GROUP),
                    strides=(stride_vn, stride_vk),
                    offsets=(offset, 0),
                    block_shape=(BLOCK_N, PACKED_D_PER_GROUP),
                    order=(1, 0),
                )
                if QUANTIZED:
                    # Pointers to quantization coefficients. Even those they are 1D,
                    # we have to use block pointers, since usual pointers
                    # don't support boundary checks
                    K_scale_shift_block_ptr = tl.make_block_ptr(
                        base=k_base,
                        shape=(1, offset + current_block_size),
                        strides=(stride_kk, stride_kn),
                        offsets=(0, offset),
                        block_shape=(1, BLOCK_N),
                        order=(0, 1),
                    )
                    V_scale_shift_block_ptr = tl.make_block_ptr(
                        base=v_base,
                        shape=(offset + current_block_size, 1),
                        strides=(stride_vn, stride_vk),
                        offsets=(offset, 0),
                        block_shape=(BLOCK_N, 1),
                        order=(1, 0),
                    )
                else:
                    K_scale_shift_block_ptr = None
                    V_scale_shift_block_ptr = None
                logical_block_idx += 1

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
            if IS_CAUSAL:
                # -- apply the causal mask --
                p = tl.where(diag_idx_shifted >= start_n, p, 0)

            # -- update m_i and l_i --
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_i_new
            p = p.to(Q.dtype.element_ty)

            # -- scale and update acc --
            for i in range(len(acc)):  # noqa: F821
                acc[i] *= alpha[:, None]  # noqa: F821
                acc[i] += tl.dot(p, v[i])  # noqa: F821

            if not PAGE_SIZE:
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
            base=Out_splitK
            + off_z * stride_osk_z
            + off_hg * stride_osk_hg
            + splitk_idx * stride_osk_s,
            shape=(N_CTX_Q, D_PER_GROUP),
            strides=(stride_osk_m, 1),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, D_PER_GROUP),
            order=(1, 0),
        )
        for i in range(len(acc)):  # noqa: F821
            # If for the current batch element there are no tokens in the current split-k chunk (because
            # seqlen is too short), l_i will be 0, so we need to make sure attention is filled with zeros and not NaNs.
            attn_out = tl.where(
                l_i[:, None] == 0, 0.0, acc[i] / l_i[:, None]  # noqa: F821
            )
            tl.store(
                tl.advance(O_block_ptr, (0, i * D_PER_GROUP)),
                attn_out.to(Out_splitK.dtype.element_ty),  # noqa: F821
                boundary_check=(0,),
            )
        if WRITE_LSE:
            LSE_splitk_ptr = (
                LSE_splitk
                + off_z * stride_lsek_z
                + off_hg * stride_lsek_hg
                + splitk_idx * stride_lsek_s
                + (start_m * BLOCK_M + tl.arange(0, BLOCK_M)) * stride_lsek_m
            )
            mask = start_m * BLOCK_M + tl.arange(0, BLOCK_M) < N_CTX_Q
            # Can be float64 to improve numerics
            lse_dtype = LSE_splitk.dtype.element_ty
            tl.store(
                LSE_splitk_ptr,
                (tl.math.log2(l_i.to(lse_dtype)) + m_i.to(lse_dtype)) / 1.44269504,
                mask=mask,
            )

    def gen_config(
        block_m: int,
        block_n: int,
        stages: int,
        warps: int,
    ) -> triton.Config:
        """A more compact way to define a triton.Config, so it fits on one line"""

        return triton.Config(
            {
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
            },
            num_stages=stages,
            num_warps=warps,
        )

    def _get_splitk_kernel(num_groups):
        """
        Kernel _fwd_kernel_splitK needs to be post-processed by unroll_varargs
        to specialize it for a given number of quantization groups N_GROUPS
        before we can apply triton.heuristics and triton.autotune, so we
        don't do them as decorators.
        """

        _fwd_kernel_splitK_unrolled = unroll_varargs(_fwd_kernel_splitK, N=num_groups)
        kernel = triton.heuristics(
            {
                "BOUNDS_CHECKS_N": lambda args: (
                    args["BLOCK_N_PER_SPLIT"] % args["BLOCK_N"]
                )
                > 0
                or args["USE_SEQ_LEN"]
            }
        )(_fwd_kernel_splitK_unrolled)
        return kernel

    @functools.lru_cache(None)
    def autotune_kernel(kernel: Callable):
        BLOCK_M_VALUES = [16, 32]
        BLOCK_N_VALUES = [32, 64, 128]
        STAGES_VALUES = [1, 2, 3]
        WARPS_VALUES = [1, 2, 4]

        TRITON_CONFIGS = [
            gen_config(block_m, block_n, stages, warps)
            for block_m in BLOCK_M_VALUES
            for block_n in BLOCK_N_VALUES
            for stages in STAGES_VALUES
            for warps in WARPS_VALUES
        ]

        kernel = triton.autotune(
            configs=TRITON_CONFIGS,
            key=AUTOTUNER_KEY,
        )(kernel)
        return kernel

    # This object contains forward kernels wrapped into autotuner for different number
    # of quantization groups.
    _fwd_kernel_splitK_autotune: Dict[int, triton.runtime.Autotuner] = {}
    # The loop below:
    # - transforms the jitted kernel with unroll_varargs producing a new kernel of each value of num_groups
    # - wraps the kernel into triton.heuristics
    # - wraps kernel into Triton autotuner. Autotuning itself happens the first time the kernel is called
    if sys.version_info >= (3, 9):
        # unroll_varargs requires Python 3.9+
        for num_groups in [1, 2, 4, 8]:
            _fwd_kernel_splitK_autotune[num_groups] = autotune_kernel(
                _get_splitk_kernel(num_groups)
            )

        def get_autotuner_cache(num_groups: int) -> Dict[Tuple[int], triton.Config]:
            """Returns a triton.runtime.autotuner.AutoTuner.cache object, which
            represents mappings from kernel autotune keys (tuples describing kernel inputs)
            to triton.Config
            """
            return _fwd_kernel_splitK_autotune[num_groups].cache

        def set_autotuner_cache(
            cache: Dict[Tuple[int], triton.Config], num_groups: int
        ) -> None:
            _fwd_kernel_splitK_autotune[num_groups].cache = cache

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
        # x_ : (BLOCK_N, D // PACKED_PER_VAL)
        # scale: (BLOCK_N, 1)
        # offsets: (PACKED_PER_VAL,)
        BLOCK_N: tl.constexpr = x_.shape[0]
        BLOCK_DMODEL_PACKED: tl.constexpr = x_.shape[1]
        offsets = tl.arange(0, PACKED_PER_VAL) * 4
        quant_offset = (
            x_[:, :, None, :] >> offsets
        )  # (BLOCK_N, D // PACKED_PER_VAL, PACKED_PER_VAL)

        quant_offset = tl.reshape(
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
        LSE_splitK,  # [B, H, split_k, Mq]
        Out,  # [B, H, M, K]
        LSE,  # [B, H, M]
        split_k: tl.constexpr,
        stride_osk_zhg,
        stride_osk_s,
        stride_osk_m,
        stride_osk_k,
        stride_lsek_zhg,
        stride_lsek_s,
        stride_lsek_m,
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
        WRITE_LSE: tl.constexpr,
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

        LSE_splitK_ptr0 = LSE_splitK + stride_lsek_zhg * off_zhg + stride_lsek_m * off_m
        LSE_splitK_ptr = LSE_splitK_ptr0
        lse_max = tl.load(LSE_splitK_ptr)
        for split_k_idx in tl.static_range(1, split_k):
            LSE_splitK_ptr = LSE_splitK_ptr + stride_lsek_s
            lse_splitk = tl.load(LSE_splitK_ptr)
            lse_max = tl.maximum(lse_max, lse_splitk)

        sumexp_normalized = 0.0
        numerator_normalized = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        LSE_splitK_ptr = LSE_splitK_ptr0
        for split_k_idx in tl.static_range(0, split_k):
            out_splitk = tl.load(Out_splitK_ptr)
            lse_splitk = tl.load(LSE_splitK_ptr)
            # Compute denominator
            sumexp_normalized_splitk = tl.math.exp2(
                (lse_splitk - lse_max).to(tl.float32) * 1.44269504
            )
            sumexp_normalized += sumexp_normalized_splitk

            # Compute numerator
            numerator_normalized += out_splitk * sumexp_normalized_splitk
            LSE_splitK_ptr = LSE_splitK_ptr + stride_lsek_s
            Out_splitK_ptr = Out_splitK_ptr + stride_osk_s

        acc = numerator_normalized / sumexp_normalized

        Out_ptr = (
            Out
            + stride_oz * off_z
            + stride_oh * off_h
            + stride_og * off_g
            + stride_om * off_m
            + tl.arange(0, BLOCK_SIZE)
        )
        if acc.dtype is tl.float64 and Out.dtype.element_ty is not tl.float64:
            # must avoid direct cast f64->f16
            acc = acc.to(tl.float32)
        tl.store(Out_ptr, acc)

        if WRITE_LSE:
            l_ptrs = LSE + off_zhg * stride_lse_zhg + off_m * stride_lse_m
            tl.store(l_ptrs, (lse_max + tl.math.log2(sumexp_normalized) / 1.44269504))

else:
    _fwd_kernel_splitK = None
    _splitK_reduce = None


def _is_cuda() -> bool:
    return torch.version.cuda is not None


def _is_cuda_at_least_sm80(device: torch.device) -> bool:
    return _is_cuda() and torch.cuda.get_device_capability(device) >= (
        8,
        0,
    )


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

    This op uses Paged Attention when bias is PagedBlockDiagonalCausalWithOffsetPaddedKeysMask.
    In this case bias has additional fields:
    - block_tables of shape [batch_size, max_num_pages]
    - K/V of shape [1, max_num_pages * page_size, num_heads, head_dim]
      or [1, max_num_pages * page_size, num_groups, num_heads, head_dim]
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
        PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
    }
    SUPPORTS_DROPOUT = False
    SUPPORTS_CUSTOM_SCALE = True
    SUPPORTS_BMGHK = True
    SUPPORTS_OUTPUT_DTYPE = True
    SUPPORTS_PARTIAL = True
    NAME = "triton_splitKF"

    SPLIT_K: Optional[int] = None
    MAX_BLOCK_M = 32

    # Perform kernel-level Triton autotune
    AUTOTUNE = False

    NUM_GROUPS = 1  # Default quantization is row-wise
    NUM_GROUPS_VALUES = [1, 2, 4, 8]

    # values used when autotune=False
    BLOCK_M: int = 16
    BLOCK_N: int = 64
    NUM_STAGES: int = 1
    NUM_WARPS: int = 2

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
        if (sys.version_info.major, sys.version_info.minor) < (3, 9):
            reasons.append("triton_splitk requires python 3.9 or above!")
        check_lastdim_alignment_stride1(reasons, "query", d.query, 8)
        if d.key.dtype != torch.int32:
            check_lastdim_alignment_stride1(reasons, "key", d.key, 8)
            check_lastdim_alignment_stride1(reasons, "value", d.value, 8)
        if cls.OPERATOR is None:
            reasons.append("triton is not available")
        if d.device.type == "cuda":
            # Has only been tested on 8.0 / 9.0.
            if _is_cuda() and not _is_cuda_at_least_sm80(d.device):
                reasons.append(
                    "requires NVidia GPU with sm80 minimum compute capacity, e.g., A100/H100/L4"
                )
            # TODO: AMD GPU support matrix needs to be figured out. MI300X is tested to work.

        q_len = d.query.shape[1]
        is_block_diagonal = isinstance(
            d.attn_bias, BlockDiagonalCausalWithOffsetPaddedKeysMask
        )
        is_paged = isinstance(
            d.attn_bias, PagedBlockDiagonalCausalWithOffsetPaddedKeysMask
        )
        if is_block_diagonal or is_paged:
            seqinfo = d.attn_bias.q_seqinfo  # type: ignore
            if q_len != seqinfo.seqstart_py[-1]:
                reasons.append(
                    f"Expected total {seqinfo.seqstart_py[-1]} queries not {q_len}"
                )
            q_len = seqinfo.min_seqlen
            if q_len != seqinfo.max_seqlen:
                reasons.append(
                    "Variable query len is not supported in the presence of causal mask."
                )
        if q_len > 16:
            # 16 is the minimum BLOCK_M which gets used
            # XXX I don't really understand why this is needed.
            reasons.append("Query length should not be larger than 16")

        if is_paged:
            page_size = d.attn_bias.page_size  # type: ignore
            if d.key.shape[1] % page_size:
                reasons.append(
                    "For paged attention, key.shape[1] should be divisible "
                    "by the page size, "
                    f"but got {d.key.shape[1]=}, {page_size=}."
                )
            if cls.AUTOTUNE:
                reasons.append("Paged attention doesn't support autotuning yet.")
            if page_size % cls.BLOCK_N:
                reasons.append(
                    "For paged attention, page size should be divisible "
                    "by the block size, "
                    f"but got {page_size=}, {cls.BLOCK_N=}."
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
        output_dtype = inp.get_output_dtype()
        attn_bias = inp.attn_bias
        seq_len = None
        q, k, v = inp.get_qkv_in_bmghk()
        IS_CAUSAL = False
        NUM_QUERIES_CAUSAL = 1

        is_block_diagonal = isinstance(
            attn_bias, BlockDiagonalCausalWithOffsetPaddedKeysMask
        )
        is_paged = isinstance(
            attn_bias, PagedBlockDiagonalCausalWithOffsetPaddedKeysMask
        )
        if attn_bias is not None:
            assert is_paged or is_block_diagonal
            # TODO: do we really need to do this cast? seems fishy but
            # I just copied it from the decoder.py
            attn_bias.k_seqinfo.to(inp.query.device)  # type: ignore
            attn_bias.q_seqinfo.to(inp.query.device)  # type: ignore
            seq_len = attn_bias.k_seqinfo.seqlen  # type: ignore
            assert q.shape[0] == 1
            B = len(seq_len)
            G, Hq, Kq = q.shape[-3:]
            Kkv = v.shape[-1]

            # assume kv has been padded
            q = q.reshape(B, -1, G, Hq, Kq)
            if is_paged:
                k = k.view(1, -1, G, Hq, Kkv)
                v = v.view(1, -1, G, Hq, Kkv)
            else:
                k = k.reshape(B, -1, G, Hq, Kkv)
                v = v.reshape(B, -1, G, Hq, Kkv)
            Mq = q.shape[1]
            IS_CAUSAL = Mq > 1
            NUM_QUERIES_CAUSAL = Mq
        else:
            B, Mq, G, Hq, Kq = q.shape

        # In the case of MQA/GQA, we make q have sequence length (H * Mq) and only one "head".
        mqa_swap_seqlen_head = False
        if (
            not needs_gradient
            and k.shape[3] > 1
            and k.stride(3) == 0
            and v.stride(3) == 0
        ):
            mqa_swap_seqlen_head = True
            # This is a copy iff Mq, G and H are all > 1.
            q = q.permute(0, 3, 1, 2, 4).reshape(B, -1, G, 1, Kq)
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

        page_size = inp.attn_bias.page_size if is_paged else 0  # type: ignore
        block_tables = None
        kv_cache_blocks_per_row = 0
        if is_paged:
            block_tables = inp.attn_bias.block_tables  # type: ignore
            kv_cache_blocks_per_row = block_tables.shape[1]
            Mk = block_tables.shape[1] * page_size
        elif attn_bias is not None:
            Mk = min(Mk, attn_bias.k_seqinfo.max_seqlen)  # type: ignore

        if cls.SPLIT_K is not None:
            split_k = cls.SPLIT_K
        else:
            # Use heuristics
            split_k = cls.get_split_k(B, H, Mk)

        if is_paged:
            # Avoid having more than one split per page
            split_k = min(split_k, block_tables.shape[1])  # type: ignore
        # M_ceil = M rounded up to a multiple of MAX_BLOCK_M
        M_ceil = (M + cls.MAX_BLOCK_M - 1) // cls.MAX_BLOCK_M * cls.MAX_BLOCK_M
        IS_SPLITK = split_k > 1  # or cls.autotune?
        if IS_SPLITK:
            o_splitk_dtype = (
                torch.float64 if output_dtype == torch.float64 else torch.float32
            )
            o_splitk = torch.empty(
                [B, G * H, split_k, M_ceil, Kq],
                dtype=o_splitk_dtype,
                device=q.device,
            )
        else:
            o_splitk = torch.empty(
                [B, split_k, M, G * H, Kq],
                dtype=output_dtype,
                device=q.device,
            ).permute(0, 3, 1, 2, 4)
        lse, lse_splitk = None, None
        # LSE may need higher precision than output
        output_f64_lse = output_dtype in (torch.float32, torch.float64)
        if IS_SPLITK and needs_gradient:
            lse_dtype = torch.float64 if output_f64_lse else torch.float32
            lse = torch.empty((B * G * H, M), device=q.device, dtype=lse_dtype)
        if IS_SPLITK or needs_gradient:
            lse_splitk = torch.empty(
                [B, G * H, split_k, M],
                dtype=torch.float64 if IS_SPLITK or output_f64_lse else torch.float32,
                device=q.device,
            )

        def grid(META):
            return triton.cdiv(M, META["BLOCK_M"]), B * G * H, split_k

        split_size = (Mk + split_k - 1) // split_k
        use_seq_len = seq_len is not None

        num_groups = cls.NUM_GROUPS if PACKED_PER_VAL > 1 else 1
        if cls.AUTOTUNE:
            kernel = _fwd_kernel_splitK_autotune[num_groups]
            extra_args = {}
        else:
            kernel = _get_splitk_kernel(num_groups)
            extra_args = {
                "BLOCK_M": cls.BLOCK_M,
                "BLOCK_N": cls.BLOCK_N,
                "num_warps": cls.NUM_WARPS,
                "num_stages": cls.NUM_STAGES,
            }
        kernel[grid](
            Q=q,
            K=k,
            V=v,
            sm_scale=inp.scale_float,
            Out_splitK=o_splitk,
            LSE_splitk=lse_splitk,
            block_tables=block_tables,
            Seq_len=seq_len,
            **_strides(q, "qz", "qm", "qg", "qh", "qk"),
            **_strides(k, "kz", "kn", "kg", "kh", "kk"),
            **_strides(v, "vz", "vn", "vg", "vh", "vk"),
            **_strides(o_splitk, "osk_z", "osk_hg", "osk_s", "osk_m", "osk_k"),
            **_strides(lse_splitk, "lsek_z", "lsek_hg", "lsek_s", "lsek_m"),
            **_strides(block_tables, "blocktablesz", "blocktablesl"),
            kv_cache_blocks_per_row=kv_cache_blocks_per_row,
            Z=B,
            H=H,
            G=G,
            N_CTX_Q=M,
            N_CTX_K=Mk,
            BLOCK_N_PER_SPLIT=split_size,
            BLOCK_DMODEL=Lk,
            USE_SEQ_LEN=use_seq_len,
            PACKED_PER_VAL=PACKED_PER_VAL,
            N_GROUPS=num_groups,
            IS_CAUSAL=IS_CAUSAL,
            NUM_QUERIES_CAUSAL=NUM_QUERIES_CAUSAL,
            IS_SPLITK=IS_SPLITK,
            USE_PAGED_ATTENTION=is_paged,
            PAGE_SIZE=page_size,
            WRITE_LSE=IS_SPLITK or needs_gradient,
            **extra_args,
        )
        if not IS_SPLITK:
            out = o_splitk[:, :, 0].view(B, G, -1, Mq, Kq)
            # This is a copy iff mqa_swap_seqlen_head and Mq, G and Hq are all > 1.
            out = out.permute(0, 3, 1, 2, 4).contiguous()
            if needs_gradient:
                assert lse_splitk is not None
                lse = lse_splitk[:, :, 0].view(B, G, -1, Mq)
                if attn_bias is not None:
                    lse = lse.permute(1, 2, 0, 3).reshape(1, G, H, B * Mq)
            else:
                lse = None

            if inp.query.ndim == 4:
                # BMGHK -> BMHK
                assert G == 1
                if lse is not None:
                    lse = lse[:, 0]
                out = out[:, :, 0]

            if lse is None:
                return out, None
            return out, Context(out=out, lse=lse)

        if mqa_swap_seqlen_head:
            out = torch.empty(
                (B, G, M, 1, Kq), device=q.device, dtype=output_dtype
            ).permute(0, 2, 1, 3, 4)
        else:
            out = torch.empty((B, M, G, H, Kq), device=q.device, dtype=output_dtype)

        # Merge attention and LSE outputs from different split-k chunks
        assert lse_splitk is not None
        merge_attentions(out, lse, o_splitk[:, :, :, :M], lse_splitk)
        if lse is not None:
            lse = lse.reshape([B, G, H, M])
            if attn_bias is not None:
                lse = lse.permute(1, 2, 0, 3).reshape(1, G, H, B * M)

        if mqa_swap_seqlen_head:
            out = out.reshape(B, -1, Mq, G, Kq).permute(0, 2, 3, 1, 4)
            # This is a copy iff Mq, G and Hq are all > 1.
            out = out.contiguous()
        if inp.query.ndim == 4:
            # BMGHK -> BMHK
            assert G == 1
            out = out[:, :, 0]
            if lse is not None:
                lse = lse[:, 0]
        if Mk == 0:
            out.zero_()

        if lse is None:
            return out, None

        return out, Context(out=out, lse=lse)

    @classmethod
    @functools.lru_cache
    def get_operator(
        cls,
        splitk: int,
        *,
        block_m: Optional[int] = None,
        block_n: Optional[int] = None,
        num_warps: Optional[int] = None,
        num_stages: Optional[int] = None,
    ) -> Type[AttentionFwOpBase]:
        kwargs = {
            "NAME": f"triton_splitK{splitk}",
            "SPLIT_K": splitk,
        }
        if block_m is not None:
            kwargs["BLOCK_M"] = block_m
        if block_n is not None:
            kwargs["BLOCK_N"] = block_n
        if num_warps is not None:
            kwargs["NUM_WARPS"] = num_warps
        if num_stages is not None:
            kwargs["NUM_STAGES"] = num_stages
        return type(
            f"FwOp_S{splitk}",
            (cls,),
            kwargs,
        )


def merge_attentions(
    attn_out: torch.Tensor,
    lse_out: Optional[torch.Tensor],
    attn_split: torch.Tensor,
    lse_split: torch.Tensor,
):
    B, M, G, H, Kq = attn_out.shape
    if lse_out is not None:
        B_H_G, M1 = lse_out.shape
    B1, H_G, split_k, M2, Kq1 = attn_split.shape
    B2, H_G1, split_k1, M3 = lse_split.shape

    assert (
        B == B1 == B2 and G * H == H_G == H_G1 and M == M2 == M3 and Kq == Kq1
    ), f"Incompatible shapes: {attn_out.shape=}, {attn_split.shape=}, {lse_split.shape=}"
    assert (
        split_k == split_k1
    ), f"Incompatible shapes: {attn_split.shape=}, {lse_split.shape=}"
    if lse_out is not None:
        assert (
            B * G * H == B_H_G and M == M1
        ), f"Incompatible shapes: {attn_out.shape=}, {lse_out.shape=}"

    # TODO: avoid this copy in more cases
    attn_split_ = attn_split.flatten(end_dim=1)
    lse_split_ = lse_split.flatten(end_dim=1)

    grid = (B * G * H, M, 1)
    _splitK_reduce[grid](
        attn_split_,
        lse_split_,
        attn_out,
        lse_out,
        split_k=split_k,
        **_strides(attn_split_, "osk_zhg", "osk_s", "osk_m", "osk_k"),
        **_strides(lse_split_, "lsek_zhg", "lsek_s", "lsek_m"),
        **_strides(attn_out, "oz", "om", "og", "oh", "ok"),
        **_strides(lse_out, "lse_zhg", "lse_m"),
        BLOCK_SIZE=attn_out.shape[-1],
        G=G,
        H=H,
        WRITE_LSE=lse_out is not None,
        num_warps=2 if B * G * H >= 32 else 4,
    )


FwOp_Map = {
    k: FwOp.get_operator(k) for k in [1, 2, 4, 8, 16, 32, 48, 64, 72, 80, 96, 112, 128]
}
FwOp_S1 = FwOp_Map[1]
FwOp_S2 = FwOp_Map[2]
FwOp_S4 = FwOp_Map[4]
FwOp_S8 = FwOp_Map[8]
FwOp_S16 = FwOp_Map[16]
FwOp_S32 = FwOp_Map[32]
FwOp_S64 = FwOp_Map[64]
FwOp_S128 = FwOp_Map[128]
