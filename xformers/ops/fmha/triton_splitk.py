# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import functools
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)

import torch

from ... import _is_triton_available
from ..common import register_operator
from .attn_bias import (
    BlockDiagonalCausalWithOffsetGappyKeysMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    BlockDiagonalGappyKeysMask,
    BlockDiagonalPaddedKeysMask,
    PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
    PagedBlockDiagonalGappyKeysMask,
    PagedBlockDiagonalPaddedKeysMask,
)
from .common import AttentionFwOpBase, Context, Inputs, check_lastdim_alignment_stride1


def _strides(x: Optional[torch.Tensor], *stride_names: str):
    if x is None:
        return {f"stride_{name}": None for name in stride_names}
    assert x.ndim == len(stride_names)
    return {f"stride_{name}": s for name, s in zip(stride_names, x.stride())}


def _is_supported_causal_bias(attn_bias: Any) -> bool:
    return isinstance(
        attn_bias,
        (
            BlockDiagonalCausalWithOffsetPaddedKeysMask,
            BlockDiagonalCausalWithOffsetGappyKeysMask,
            PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
        ),
    )


def _is_supported_gappy_bias(attn_bias: Any) -> bool:
    return isinstance(
        attn_bias,
        (
            BlockDiagonalGappyKeysMask,
            PagedBlockDiagonalGappyKeysMask,
        ),
    )


def _is_supported_paged_bias(attn_bias: Any) -> bool:
    return isinstance(
        attn_bias,
        (
            PagedBlockDiagonalGappyKeysMask,
            PagedBlockDiagonalPaddedKeysMask,
        ),
    )


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

if TYPE_CHECKING or _is_triton_available():
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
        Seq_starts_k,
        Seq_starts_q,
        Seq_starts_q_multiplier,
        additive_bias,
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
        stride_osk_g,
        stride_osk_h,
        stride_osk_s,
        stride_osk_m,
        stride_osk_k,
        stride_lsek_z,
        stride_lsek_g,
        stride_lsek_h,
        stride_lsek_s,
        stride_lsek_m,
        stride_blocktablesz,
        stride_blocktablesl,
        stride_bias_b,
        stride_bias_g,
        stride_bias_h,
        stride_bias_qm,
        stride_bias_km,
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
        SPLIT_K_EARLY_EXIT: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        NUM_QUERIES_CAUSAL: tl.constexpr,  # The N_CTX_Q queries are from this many sequence positions
        USE_PAGED_ATTENTION: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        WRITE_LSE: tl.constexpr,
        HAS_ADDITIVE_BIAS: tl.constexpr,
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
            if SPLIT_K_EARLY_EXIT and kv_len == 0:
                return
        else:
            kv_len = N_CTX_K

        if Seq_starts_k is None:
            start_kv_idx = 0
        else:
            start_kv_idx = tl.load(Seq_starts_k + off_z)

        if Seq_starts_q is None:
            q_len = N_CTX_Q
            queries_use_batch_dim = 1
            off_m = 0
        else:
            queries_use_batch_dim = 0
            off_m = tl.load(Seq_starts_q + off_z) * Seq_starts_q_multiplier
            q_len = tl.load(Seq_starts_q + off_z + 1) * Seq_starts_q_multiplier - off_m
            if q_len == 0:
                return

        k_base = K + off_h * stride_kh + off_g * stride_kg
        v_base = V + off_h * stride_vh + off_g * stride_vg

        # Boundaries of split-k chunk
        chunk_hi = (splitk_idx + 1) * BLOCK_N_PER_SPLIT
        chunk_lo = splitk_idx * BLOCK_N_PER_SPLIT
        ignore_in_first_block = 0
        # For paged attention case K/V_block_ptr are defined inside the loop
        # whereas for non-paged case they are defined before the loop.
        if PAGE_SIZE > 0:
            # Page contains several blocks
            BLOCKS_IN_PAGE: tl.constexpr = PAGE_SIZE // BLOCK_N
            # Align boundaries of split-k chunk to block boundaries
            # In the last chunk, shift hi to the right, in the other chunks, shift it to the left
            is_last_chunk = splitk_idx == tl.num_programs(2) - 1
            shift = BLOCK_N - 1 if is_last_chunk else 0
            lo = (tl.maximum(chunk_lo, start_kv_idx) // BLOCK_N) * BLOCK_N
            ignore_in_first_block = tl.maximum(0, (start_kv_idx - lo))
            hi = ((chunk_hi + shift) // BLOCK_N) * BLOCK_N
            hi = tl.minimum(hi, kv_len + start_kv_idx)
            block_table = block_tables + stride_blocktablesz * off_z
            # Offset in integer blocks
            logical_block_idx = lo // BLOCK_N
        else:
            lo = chunk_lo
            hi = tl.minimum(chunk_hi, kv_len)
            if Seq_starts_k is not None:
                k_base += start_kv_idx * stride_kn
                v_base += start_kv_idx * stride_vn
            else:
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

            if HAS_ADDITIVE_BIAS:
                additive_bias_block_ptr = tl.make_block_ptr(
                    base=additive_bias
                    + off_z * stride_bias_b
                    + off_g * stride_bias_g
                    + off_h * stride_bias_h,
                    shape=(N_CTX_Q, hi),
                    strides=(stride_bias_qm, stride_bias_km),
                    offsets=(start_m * BLOCK_M, lo),
                    block_shape=(BLOCK_M, BLOCK_N),
                    order=(0, 1),
                )

        if SPLIT_K_EARLY_EXIT and lo >= hi:
            return

        Q_block_ptr = tl.make_block_ptr(
            base=Q
            + off_m * stride_qm
            + off_h * stride_qh
            + off_z * stride_qz * queries_use_batch_dim
            + off_g * stride_qg,
            shape=(q_len, BLOCK_DMODEL),
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

            if start_n == lo and ignore_in_first_block > 0:
                qk = tl.where(
                    tl.arange(0, BLOCK_N) < ignore_in_first_block, float("-inf"), qk
                )

            if HAS_ADDITIVE_BIAS:
                loaded_bias = tl.load(
                    additive_bias_block_ptr,
                    boundary_check=(0, 1) if BOUNDS_CHECKS_N else (0,),
                )
                qk += loaded_bias * 1.44269504
                additive_bias_block_ptr = tl.advance(
                    additive_bias_block_ptr, (0, BLOCK_N)
                )

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
            + off_z.to(tl.int64) * stride_osk_z * queries_use_batch_dim
            + off_m * stride_osk_m
            + off_g * stride_osk_g
            + off_h * stride_osk_h
            + splitk_idx * stride_osk_s,
            shape=(q_len, D_PER_GROUP),
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
                + off_z * stride_lsek_z * queries_use_batch_dim
                + off_m * stride_lsek_m
                + off_g * stride_lsek_g
                + off_h * stride_lsek_h
                + splitk_idx * stride_lsek_s
                + (start_m * BLOCK_M + tl.arange(0, BLOCK_M)) * stride_lsek_m
            )
            mask = start_m * BLOCK_M + tl.arange(0, BLOCK_M) < q_len
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
                "BOUNDS_CHECKS_N": lambda args: bool(
                    (args["BLOCK_N_PER_SPLIT"] % args["BLOCK_N"])
                    or (args["N_CTX_K"] % args["BLOCK_N_PER_SPLIT"])
                    or args["USE_SEQ_LEN"]
                )
            }
        )(_fwd_kernel_splitK_unrolled)
        return kernel

    @functools.lru_cache(None)
    def autotune_kernel(kernel: Callable):
        BLOCK_M_VALUES = [16, 32]
        BLOCK_N_VALUES = [32, 64, 128]
        # On AMD num_stages has to be 0 or 1, but 0 sometimes produces NaN or incorrect results.
        STAGES_VALUES = [1] if torch.version.hip else [1, 2, 3]
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
        Out_splitK,  # [B, G, H, split_k, Mq, K]
        LSE_splitK,  # [B, G, H, split_k, Mq]
        Out,  # [B, H, M, K]
        LSE,  # [B, H, M]
        split_k: tl.constexpr,
        splitK_pow2: tl.constexpr,
        stride_osk_z,
        stride_osk_g,
        stride_osk_h,
        stride_osk_s,
        stride_osk_m,
        stride_osk_k,
        stride_lsek_z,
        stride_lsek_g,
        stride_lsek_h,
        stride_lsek_s,
        stride_lsek_m,
        stride_oz,
        stride_og,
        stride_oh,
        stride_om,
        stride_ok,
        stride_lse_z,
        stride_lse_g,
        stride_lse_h,
        stride_lse_m,
        BLOCK_SIZE: tl.constexpr,
        H: tl.constexpr,
        G: tl.constexpr,
        WRITE_LSE: tl.constexpr,
    ):
        # grid = (M, B * G * H, 1)
        off_m = tl.program_id(0).to(tl.int64)
        off_zhg = tl.program_id(1).to(tl.int64)
        off_z = off_zhg // (H * G)
        off_h = (off_zhg // G) % H
        off_g = off_zhg % G

        Out_splitK_ptr = (
            Out_splitK
            + stride_osk_z * off_z
            + stride_osk_g * off_g
            + stride_osk_h * off_h
            + stride_osk_m * off_m
            + tl.arange(0, BLOCK_SIZE)[None, :]
            + stride_osk_s * tl.arange(0, splitK_pow2)[:, None]
        )

        LSE_splitK_ptr0 = (
            LSE_splitK
            + stride_lsek_z * off_z
            + stride_lsek_g * off_g
            + stride_lsek_h * off_h
            + stride_lsek_m * off_m
            + stride_lsek_s * tl.arange(0, splitK_pow2)
        )

        if splitK_pow2 > split_k:
            mask_1d = tl.arange(0, splitK_pow2) < split_k
            mask_2d = mask_1d[:, None]
            lse_splitk = tl.load(LSE_splitK_ptr0, mask=mask_1d, other=float("-inf"))
            lse_max = tl.max(lse_splitk)
            out_splitk = tl.load(
                Out_splitK_ptr, mask=mask_2d, other=0
            )  # (split_k, BLOCK_SIZE)
            lse_splitk = tl.load(
                LSE_splitK_ptr0, mask=mask_1d, other=float("-inf")
            )  # (split_k,)
        else:
            lse_splitk = tl.load(LSE_splitK_ptr0)
            lse_max = tl.max(lse_splitk)
            out_splitk = tl.load(Out_splitK_ptr)
            lse_splitk = tl.load(LSE_splitK_ptr0)

        sumexp_normalized_splitk = tl.math.exp2(
            (lse_splitk - lse_max).to(tl.float32) * 1.44269504
        )  # (split_k,)
        sumexp_normalized = tl.sum(sumexp_normalized_splitk, axis=0)  # scalar
        # Compute numerator
        numerator_normalized = tl.sum(
            out_splitk * sumexp_normalized_splitk[:, None], axis=0
        )
        acc = numerator_normalized / sumexp_normalized
        acc = tl.where(lse_max == float("-inf"), 0.0, acc)

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
            l_ptrs = (
                LSE
                + off_z * stride_lse_z
                + off_g * stride_lse_g
                + off_h * stride_lse_h
                + off_m * stride_lse_m
            )
            to_store = lse_max + tl.math.log2(sumexp_normalized) / 1.44269504
            to_store = tl.where(lse_max == float("-inf"), lse_max, to_store)
            tl.store(l_ptrs, to_store)

    @triton.jit
    def _splitK_reduce_varargs(
        Out_splitK: "VAR_ARGS_ARRAY",  # list of [B, G, H, Mq, K];
        LSE_splitK: "VAR_ARGS_ARRAY",  # list of [B, G, H, Mq]
        Out,  # [B, G, H, M, K]
        LSE,  # [B, G, H, M]
        stride_osk_z: "VAR_ARGS_ARRAY",
        stride_osk_g: "VAR_ARGS_ARRAY",
        stride_osk_h: "VAR_ARGS_ARRAY",
        stride_osk_m: "VAR_ARGS_ARRAY",
        stride_osk_k: "VAR_ARGS_ARRAY",
        stride_lsek_z: "VAR_ARGS_ARRAY",
        stride_lsek_g: "VAR_ARGS_ARRAY",
        stride_lsek_h: "VAR_ARGS_ARRAY",
        stride_lsek_m: "VAR_ARGS_ARRAY",
        stride_oz,
        stride_og,
        stride_oh,
        stride_om,
        stride_ok,
        stride_lse_z,
        stride_lse_g,
        stride_lse_h,
        stride_lse_m,
        BLOCK_SIZE: tl.constexpr,
        H: tl.constexpr,
        G: tl.constexpr,
        WRITE_LSE: tl.constexpr,
    ):
        """
        This version of reduce kernel takes attention and LSE of chunks as lists of tensors,
        as opposed to _splitK_reduce, which takes each as a stacked tensor.
        """
        # grid = (M, B * G * H, 1)
        off_m = tl.program_id(0).to(tl.int64)
        off_zhg = tl.program_id(1).to(tl.int64)
        off_z = off_zhg // (H * G)
        off_h = (off_zhg // G) % H
        off_g = off_zhg % G

        out_splitk_offset: "VAR_ARGS_ARRAY"  # noqa: F821
        for i in range(len(Out_splitK)):
            out_splitk_offset[i] = (  # noqa: F821
                stride_osk_z[i] * off_z  # type: ignore # noqa: F821
                + stride_osk_g[i] * off_g
                + stride_osk_h[i] * off_h
                + stride_osk_m[i] * off_m
                + tl.arange(0, BLOCK_SIZE)
            )
        lse_splitk_offset: "VAR_ARGS_ARRAY"  # noqa: F821
        for i in range(len(Out_splitK)):
            lse_splitk_offset[i] = (  # noqa: F821
                stride_lsek_z[i] * off_z  # type: ignore # noqa: F821
                + stride_lsek_g[i] * off_g
                + stride_lsek_h[i] * off_h
                + stride_lsek_m[i] * off_m
            )

        lse_max = float("-inf")
        for split_k_idx in range(len(Out_splitK)):  # type: ignore # noqa: F821
            LSE_splitK_ptr = LSE_splitK[split_k_idx] + lse_splitk_offset[split_k_idx]  # type: ignore # noqa: F821
            lse_splitk = tl.load(LSE_splitK_ptr)
            lse_max = tl.maximum(lse_max, lse_splitk)

        sumexp_normalized = 0.0
        numerator_normalized = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

        for split_k_idx in range(len(Out_splitK)):  # type: ignore # noqa: F821
            out_splitk = tl.load(Out_splitK[split_k_idx] + out_splitk_offset[split_k_idx])  # type: ignore # noqa: F821
            lse_splitk = tl.load(LSE_splitK[split_k_idx] + lse_splitk_offset[split_k_idx])  # type: ignore # noqa: F821
            # Compute denominator
            sumexp_normalized_splitk = tl.math.exp2(
                (lse_splitk - lse_max).to(tl.float32) * 1.44269504
            )
            sumexp_normalized += sumexp_normalized_splitk

            # Compute numerator
            numerator_normalized += out_splitk * sumexp_normalized_splitk

        acc = numerator_normalized / sumexp_normalized
        acc = tl.where(lse_max == float("-inf"), 0.0, acc)

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
            l_ptrs = (
                LSE
                + off_z * stride_lse_z
                + off_g * stride_lse_g
                + off_h * stride_lse_h
                + off_m * stride_lse_m
            )
            to_store = lse_max + tl.math.log2(sumexp_normalized) / 1.44269504
            to_store = tl.where(lse_max == float("-inf"), lse_max, to_store)
            tl.store(l_ptrs, to_store)

    @triton.jit
    def _splitK_reduce_varargs_backward(
        Out_splitK: "VAR_ARGS_ARRAY",  # list of [B, G, H, Mq, K];
        LSE_splitK: "VAR_ARGS_ARRAY",  # list of [B, G, H, Mq]
        Dout_splitK: "VAR_ARGS_ARRAY",  # gradients - same shape as the inputs themselves
        DLSE_splitK: "VAR_ARGS_ARRAY",
        Out,  # [B, G, H, M, K]
        LSE,  # [B, G, H, M]
        DOut,
        DLSE,
        # strides of chunked inputs: attention and LSE
        stride_osk_z: "VAR_ARGS_ARRAY",
        stride_osk_g: "VAR_ARGS_ARRAY",
        stride_osk_h: "VAR_ARGS_ARRAY",
        stride_osk_m: "VAR_ARGS_ARRAY",
        stride_osk_k: "VAR_ARGS_ARRAY",
        stride_lsek_z: "VAR_ARGS_ARRAY",
        stride_lsek_g: "VAR_ARGS_ARRAY",
        stride_lsek_h: "VAR_ARGS_ARRAY",
        stride_lsek_m: "VAR_ARGS_ARRAY",
        # strides of merged outputs: attention and LSE
        stride_oz,
        stride_og,
        stride_oh,
        stride_om,
        stride_ok,
        stride_lse_z,
        stride_lse_g,
        stride_lse_h,
        stride_lse_m,
        # strides of gradients
        stride_doz,
        stride_dog,
        stride_doh,
        stride_dom,
        stride_dok,
        stride_dlse_z,
        stride_dlse_g,
        stride_dlse_h,
        stride_dlse_m,
        BLOCK_SIZE: tl.constexpr,
        H: tl.constexpr,
        G: tl.constexpr,
    ):
        """
        Backward for _splitK_reduce_varargs. Similar to forward, it takes
        attention and LSE of chunks as lists of tensors,
        and outputs the corresponding gradients in the same format.
        """

        # grid = (M, B * G * H, 1)
        off_m = tl.program_id(0).to(tl.int64)
        off_zhg = tl.program_id(1).to(tl.int64)
        off_z = off_zhg // (H * G)
        off_h = (off_zhg // G) % H
        off_g = off_zhg % G

        # Compute offsets inside each attention/LSE chunk.
        # Note that each chunk can have different strides, so offsets can also be different.
        out_splitk_offset: "VAR_ARGS_ARRAY"  # noqa: F821
        for i in range(len(Out_splitK)):
            out_splitk_offset[i] = (  # type: ignore # noqa: F821
                stride_osk_z[i] * off_z
                + stride_osk_g[i] * off_g
                + stride_osk_h[i] * off_h
                + stride_osk_m[i] * off_m
                + tl.arange(0, BLOCK_SIZE)
            )
        lse_splitk_offset: "VAR_ARGS_ARRAY"  # noqa: F821
        for i in range(len(Out_splitK)):
            lse_splitk_offset[i] = (  # type: ignore # noqa: F821
                stride_lsek_z[i] * off_z
                + stride_lsek_g[i] * off_g
                + stride_lsek_h[i] * off_h
                + stride_lsek_m[i] * off_m
            )

        lse_max = float("-inf")
        for split_k_idx in range(len(Out_splitK)):  # type: ignore # noqa: F821
            LSE_splitK_ptr = LSE_splitK[split_k_idx] + lse_splitk_offset[split_k_idx]  # type: ignore # noqa: F821
            lse_splitk = tl.load(LSE_splitK_ptr)
            lse_max = tl.maximum(lse_max, lse_splitk)

        # Load attention and the corresponding gradient
        offset_out = (
            stride_oz * off_z
            + stride_oh * off_h
            + stride_og * off_g
            + stride_om * off_m
            + tl.arange(0, BLOCK_SIZE)
        )
        offset_dout = (
            stride_doz * off_z
            + stride_doh * off_h
            + stride_dog * off_g
            + stride_dom * off_m
            + tl.arange(0, BLOCK_SIZE)
        )
        out = tl.load(Out + offset_out)
        dattn = tl.load(DOut + offset_dout)

        # Load LSE and the corresponding gradient
        offset_lse = (
            stride_lse_z * off_z
            + stride_lse_h * off_h
            + stride_lse_g * off_g
            + stride_lse_m * off_m
        )
        offset_dlse = (
            stride_dlse_z * off_z
            + stride_dlse_h * off_h
            + stride_dlse_g * off_g
            + stride_dlse_m * off_m
        )
        lse = tl.load(LSE + offset_lse)
        dlse = tl.load(DLSE + offset_dlse)

        for split_k_idx in range(len(Out_splitK)):  # type: ignore # noqa: F821
            # Load attention and LSE of chunks
            out_splitk = tl.load(Out_splitK[split_k_idx] + out_splitk_offset[split_k_idx])  # type: ignore # noqa: F821
            lse_splitk = tl.load(LSE_splitK[split_k_idx] + lse_splitk_offset[split_k_idx])  # type: ignore # noqa: F821

            # Pointers to save gradients of attention and LSE of chunks
            dout_splitk_ptr = Dout_splitK[split_k_idx] + out_splitk_offset[split_k_idx]  # type: ignore # noqa: F821
            dlse_splitk_ptr = DLSE_splitK[split_k_idx] + lse_splitk_offset[split_k_idx]  # type: ignore # noqa: F821

            # dX/dattn_i = dX/dattn * dattn/dattn_i + dX/dlse * dlse/dattn_i, and dlse/dattn_i == 0
            dattn_dattn_i = tl.exp(lse_splitk - lse_max) / tl.exp(lse - lse_max)
            dX_dattn_i = dattn_dattn_i * dattn
            tl.store(dout_splitk_ptr, dX_dattn_i)

            dattn_dlse_i = (out_splitk - out) * dattn_dattn_i

            # dX/dlse_i = dX/dattn * dattn/dlse_i + dX/dlse * dlse/dlse_i
            dlse_dlse_i = dattn_dattn_i
            dX_dlse_i = dlse_dlse_i * dlse + tl.sum(
                dattn_dlse_i * dattn
            )  # Sum is over the hidden dimension
            tl.store(dlse_splitk_ptr, dX_dlse_i)

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

    This op uses Paged Attention when bias is one of the Paged* classes.
    In this case bias has additional fields:
    - block_tables of shape [batch_size, max_num_pages]
    - K/V of shape [1, max_num_pages * page_size, num_heads, head_dim]
      or [1, max_num_pages * page_size, num_groups, num_heads, head_dim]

    The shape which the kernel takes the queries and the output
    is quite different from the user interface. There are three
    types of input (a) no bias / tensor bias, (b) variable q_len
    (which is only for non causal) and (c) other bias objects.
    From the interface to the kernel the following changes happen.

    (0) In all cases, a group dimension may need to be added.

    (1) For (c), a batch dimension is created, reshaping from (1, B*Mq, G, Hq, K)
        to (B, Mq, G, Hq, K)

    (2) For (a) and (c), in the case of multiquery (i.e. the head dimension
        of keys and values is expanded), the head-swapping trick
        reshaping from (B, Mq, G, Hq, K) to (B, M=Hq*Mq, G, H=1, K)

    (3) For (b), in the case of multiquery, the head-swapping trick
        trick, reshaping from (1, Mq, G, Hq, K) to (1, Mq*Hq, G, H=1, K)
        Note here that Mq is a single long dimension which spans all the queries
        in the batch, unlike in case (C). Also that Hq has to run faster than
        Mq in order that the queries in a batch element remain evenly spaced.

    In all cases, the shape as seen by the kernel is called (Bqq, Mqq, G, H, K).
    The kernel operates on B batch elements and M queries per batch element.
    """

    OPERATOR = _fwd_kernel_splitK
    SUPPORTED_DEVICES = {"cuda"}
    CUDA_MINIMUM_COMPUTE_CAPABILITY = (8, 0)
    SUPPORTED_DTYPES = {
        torch.half,
        torch.bfloat16,
    }  # Those are dtypes of Q. In the quantized case K/V has dtype int32
    SUPPORTED_MAX_K = 512
    SUPPORTED_ATTN_BIAS_TYPES: Iterable[Any] = (
        type(None),
        torch.Tensor,
        BlockDiagonalCausalWithOffsetPaddedKeysMask,
        BlockDiagonalGappyKeysMask,
        BlockDiagonalCausalWithOffsetGappyKeysMask,
        BlockDiagonalPaddedKeysMask,
        PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
        PagedBlockDiagonalGappyKeysMask,
        PagedBlockDiagonalPaddedKeysMask,
    )
    SUPPORTS_DROPOUT = False
    SUPPORTS_CUSTOM_SCALE = True
    SUPPORTS_BMGHK = True
    SUPPORTS_OUTPUT_DTYPE = True
    SUPPORTS_PARTIAL = True
    NAME = "triton_splitKF"

    SPLIT_K: Optional[int] = None
    MAX_BLOCK_M = 32

    # Whether blocks attending to no part of a variable sequence length
    # should exit early. This requires extra kernels to run beforehand
    # to initialise the outputs.
    # TODO: avoid these by making the reduce kernel work out it doesn't need
    # to look at the irrelevant places.
    SPLIT_K_EARLY_EXIT: bool = False

    # Perform kernel-level Triton autotune
    AUTOTUNE = False

    NUM_GROUPS = 1  # Default quantization is row-wise
    NUM_GROUPS_VALUES = [1, 2, 4, 8]

    # values used when autotune=False
    BLOCK_M: int = 16
    BLOCK_N: int = 64
    # On AMD these two values are overwritten depending on input shapes, see the code just before the kernel launch
    # This might change once we get autotuning working on AMD
    NUM_STAGES: int = 1
    NUM_WARPS: int = 2

    @classmethod
    def shape_not_supported_reasons(
        cls, Mq: int, Mkv: int, K: int, Kv: int
    ) -> List[str]:
        reasons = super().shape_not_supported_reasons(Mq, Mkv, K, Kv)
        if K not in {16, 32, 64, 128, 256, 512}:
            reasons.append(f"Embed dim {K} not supported")
        if Mkv == 0:
            # Other ops support this; but here, triton compilation
            # crashes on A100
            reasons.append("Query length is 0")
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
            d.attn_bias, (BlockDiagonalPaddedKeysMask, BlockDiagonalGappyKeysMask)
        )
        is_paged = _is_supported_paged_bias(d.attn_bias)
        is_causal = _is_supported_causal_bias(d.attn_bias)
        if is_block_diagonal or is_paged:
            seqinfo = d.attn_bias.q_seqinfo  # type: ignore
            if q_len != seqinfo.seqstart_py[-1]:
                reasons.append(
                    f"Expected total {seqinfo.seqstart_py[-1]} queries not {q_len}"
                )
            q_len = seqinfo.max_seqlen
            if is_causal and q_len != seqinfo.min_seqlen:
                reasons.append("Variable query len is not supported for causal masks.")
        if q_len > 16 and is_causal:
            # 16 is the minimum BLOCK_M which gets used
            # XXX I don't really understand why this is needed.
            reasons.append(
                "Query length should not be larger than 16 for causal attention biases"
            )

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

        if isinstance(d.attn_bias, torch.Tensor):
            if d.attn_bias.ndim not in (4, 5):
                reasons.append(
                    "Additive attention bias has to have shape (B, G, H, Mq, Mkv) "
                    f"or (B, H, Mq, Mkv), but got {d.attn_bias.shape}."
                )

        return reasons

    @classmethod
    def get_split_k(cls, B: int, G: int, H: int, Mk: int) -> int:
        """Heuristic for the number of splits"""
        bh = max(B * H, 1)  # NOTE: Handle B*h=0 case
        if torch.version.hip:
            split_k = max(Mk + bh - 1, 1024) // bh
            max_chunk_size = 64
            split_k_stop_val = 1024 / (B * G * H)
            while split_k > 1 and Mk / (split_k - 1) < max_chunk_size:
                split_k = split_k - 1

            while split_k > split_k_stop_val:
                split_k = split_k // 2

            split_size = (Mk + split_k - 1) // split_k

            chunk_size = split_size // max_chunk_size * max_chunk_size
            if chunk_size < split_size:
                split_k += 1

            split_k_upper_bound = 512
        else:
            split_k = max(Mk, 1024) // bh
            max_chunk_size = 64 if Mk <= 512 and bh <= 64 else 128
            split_k_stop_val = Mk / max_chunk_size
            split_k_upper_bound = 64

            while split_k > split_k_stop_val:
                split_k = split_k // 2

        split_k = min(split_k, split_k_upper_bound)
        split_k = max(split_k, 1)

        return split_k

    @classmethod
    def apply(
        cls, inp: Inputs, needs_gradient: bool
    ) -> Tuple[torch.Tensor, Optional[Context]]:
        output_dtype = inp.get_output_dtype()
        if not isinstance(inp.attn_bias, torch.Tensor):
            attn_bias_tensor = None
            attn_bias = cast(
                Optional[
                    Union[
                        BlockDiagonalCausalWithOffsetPaddedKeysMask,
                        BlockDiagonalGappyKeysMask,
                        BlockDiagonalCausalWithOffsetGappyKeysMask,
                        BlockDiagonalPaddedKeysMask,
                        PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
                        PagedBlockDiagonalGappyKeysMask,
                        PagedBlockDiagonalPaddedKeysMask,
                    ]
                ],
                inp.attn_bias,
            )
        else:
            attn_bias_tensor = inp.attn_bias
            attn_bias = None

        seq_len = None
        seq_starts_k = None
        seq_starts_q = None
        seq_starts_q_multiplier = None
        q, k, v = inp.get_qkv_in_bmghk()
        IS_CAUSAL = False
        NUM_QUERIES_CAUSAL = 1
        variable_q = False

        is_block_diagonal = isinstance(attn_bias, BlockDiagonalPaddedKeysMask)
        is_gappy = _is_supported_gappy_bias(attn_bias)
        is_paged = _is_supported_paged_bias(attn_bias)
        if attn_bias is not None:
            assert is_paged or is_block_diagonal or is_gappy
            assert attn_bias.k_seqinfo.seqlen.device == inp.query.device
            seq_len = attn_bias.k_seqinfo.seqlen
            assert seq_len.stride(0) == 1
            if is_gappy:
                seq_starts_k = attn_bias.k_seqinfo.seqstart
                assert seq_starts_k.stride(0) == 1
            assert q.shape[0] == 1
            B = len(seq_len)
            G, Hq, Kq = q.shape[-3:]
            # force a bool because triton cannot take np.bool_
            multiple_q = bool(attn_bias.q_seqinfo.max_seqlen > 1)
            IS_CAUSAL = multiple_q and _is_supported_causal_bias(attn_bias)
            variable_q = multiple_q and not IS_CAUSAL
            Kkv = v.shape[-1]

            if variable_q:
                seq_starts_q = attn_bias.q_seqinfo.seqstart
                seq_starts_q_multiplier = 1
                assert seq_starts_q.stride(0) == 1
            else:
                q = q.view(B, -1, G, Hq, Kq)
            if is_paged or is_gappy:
                k = k.view(1, -1, G, Hq, Kkv)
                v = v.view(1, -1, G, Hq, Kkv)
            else:
                k = k.view(B, -1, G, Hq, Kkv)
                v = v.view(B, -1, G, Hq, Kkv)
            Mq = q.shape[1]
            NUM_QUERIES_CAUSAL = Mq
        else:
            B, Mq, G, Hq, Kq = q.shape

        if attn_bias_tensor is not None and attn_bias_tensor.ndim == 4:
            # (B, H, Mq, Mkv) -> (B, G, H, Mq, Mkv)
            attn_bias_tensor = attn_bias_tensor.unsqueeze(1)

        # In the case of MQA/GQA, we make q have sequence length (H * Mq) and only one "head".
        mqa_swap_seqlen_head = False
        if (
            k.shape[3] > 1
            and k.stride(3) == 0
            and v.stride(3) == 0
            and attn_bias_tensor is None
        ):
            mqa_swap_seqlen_head = True
            if variable_q:
                seq_starts_q_multiplier = Hq
                assert q.shape[0] == 1
                # The idea is Hq,Mq are reshaped to (M=Mq*Hq, H=1)
                q = q.permute(0, 1, 3, 2, 4).reshape(1, -1, G, 1, Kq)
            else:
                # This is a copy iff Mq, G and H are all > 1.
                # The idea is Hq,Mq are reshaped to (M=Hq*Mq, H=1)
                q = q.permute(0, 3, 1, 2, 4).reshape(q.shape[0], -1, G, 1, Kq)
            k = k[:, :, :, :1]
            v = v[:, :, :, :1]

        if k.dtype == torch.int32:
            # Quantized K/V
            PACKED_PER_VAL = 8
            Lk = (k.shape[-1] - cls.NUM_GROUPS) * 8
        else:
            Lk = k.shape[-1]
            PACKED_PER_VAL = 1

        _, Mk, G, H, Kkv = k.shape
        Bqq, Mqq, G, H, Kq = q.shape
        assert Lk == Kq, f"Keys have head dim {Lk} but queries have head dim {Kq}"
        if variable_q:
            assert attn_bias is not None
            assert seq_starts_q_multiplier is not None
            M = attn_bias.q_seqinfo.max_seqlen * seq_starts_q_multiplier
        else:
            M = Mqq
        page_size = inp.attn_bias.page_size if is_paged else 0  # type: ignore
        block_tables = None
        kv_cache_blocks_per_row = 0
        if is_paged:
            block_tables = inp.attn_bias.block_tables  # type: ignore
            kv_cache_blocks_per_row = block_tables.shape[1]
            Mk = block_tables.shape[1] * page_size
        elif attn_bias is not None:
            Mk = min(Mk, attn_bias.k_seqinfo.max_seqlen)

        if cls.SPLIT_K is not None:
            split_k = cls.SPLIT_K
        else:
            # Use heuristics
            split_k = cls.get_split_k(B, G, H, Mk)

        # M_ceil = Mqq rounded up to a multiple of MAX_BLOCK_M
        M_ceil = (Mqq + cls.MAX_BLOCK_M - 1) // cls.MAX_BLOCK_M * cls.MAX_BLOCK_M
        IS_SPLITK = split_k > 1  # or cls.autotune?
        output_shape = (Bqq, Mq, G, Hq, Kq)
        if IS_SPLITK:
            o_splitk_dtype = (
                torch.float64 if output_dtype == torch.float64 else torch.float32
            )
            if cls.SPLIT_K_EARLY_EXIT:
                o_splitk = torch.zeros(
                    [Bqq, G, H, split_k, M_ceil, Kq],
                    dtype=o_splitk_dtype,
                    device=q.device,
                )
            else:
                o_splitk = torch.empty(
                    [Bqq, G, H, split_k, M_ceil, Kq],
                    dtype=o_splitk_dtype,
                    device=q.device,
                )
        else:
            o_splitk = torch.empty(
                [Bqq, split_k, Mqq, G, H, Kq],
                dtype=output_dtype,
                device=q.device,
            ).permute(0, 3, 4, 1, 2, 5)
        lse, lse_splitk = None, None
        # LSE may need higher precision than output
        output_f64_lse = output_dtype in (torch.float32, torch.float64)
        if IS_SPLITK or needs_gradient:
            if cls.SPLIT_K_EARLY_EXIT:
                lse_splitk = torch.full(
                    [Bqq, G, H, split_k, Mqq],
                    -float("inf"),
                    dtype=torch.float64
                    if IS_SPLITK or output_f64_lse
                    else torch.float32,
                    device=q.device,
                )
            else:
                lse_splitk = torch.empty(
                    [Bqq, G, H, split_k, Mqq],
                    dtype=torch.float64
                    if IS_SPLITK or output_f64_lse
                    else torch.float32,
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

            # TODO: remove this when autotuning on AMD is working
            num_warps = cls.NUM_WARPS
            num_stages = cls.NUM_STAGES
            if torch.version.hip:
                if B == 1:
                    num_warps = 4
                    num_stages = 1  # TODO num_stages = 0 gives better perf on AMD, but sometimes produces NaNs
                elif B <= 4 and split_k <= 128:
                    num_warps = 2
                    num_stages = 1
                else:
                    num_warps = 1
                    num_stages = 1
            extra_args = {
                "BLOCK_M": cls.BLOCK_M,
                "BLOCK_N": cls.BLOCK_N,
                "num_warps": num_warps,
                "num_stages": num_stages,
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
            Seq_starts_k=seq_starts_k,
            Seq_starts_q=seq_starts_q,
            Seq_starts_q_multiplier=seq_starts_q_multiplier,
            additive_bias=attn_bias_tensor,
            **_strides(q, "qz", "qm", "qg", "qh", "qk"),
            **_strides(k, "kz", "kn", "kg", "kh", "kk"),
            **_strides(v, "vz", "vn", "vg", "vh", "vk"),
            **_strides(o_splitk, "osk_z", "osk_g", "osk_h", "osk_s", "osk_m", "osk_k"),
            **_strides(lse_splitk, "lsek_z", "lsek_g", "lsek_h", "lsek_s", "lsek_m"),
            **_strides(block_tables, "blocktablesz", "blocktablesl"),
            **_strides(
                attn_bias_tensor, "bias_b", "bias_g", "bias_h", "bias_qm", "bias_km"
            ),
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
            SPLIT_K_EARLY_EXIT=cls.SPLIT_K_EARLY_EXIT,
            USE_PAGED_ATTENTION=is_paged,
            PAGE_SIZE=page_size,
            WRITE_LSE=IS_SPLITK or needs_gradient,
            HAS_ADDITIVE_BIAS=attn_bias_tensor is not None,
            **extra_args,
        )
        if not IS_SPLITK:
            out = o_splitk[:, :, :, 0]  # Bqq, G, H, Mqq, Kq
            if variable_q and mqa_swap_seqlen_head:
                out = out.view(1, G, Mq, Hq, Kq).permute(0, 2, 1, 3, 4).contiguous()
            else:
                out = out.view(Bqq, G, Hq, Mq, Kq)
                # This is a copy iff mqa_swap_seqlen_head and Mq, G and Hq are all > 1.
                out = out.permute(0, 3, 1, 2, 4).contiguous()
            if needs_gradient:
                assert lse_splitk is not None
                lse = lse_splitk[:, :, :, 0]  # Bqq, G, H, Mqq
                if variable_q and mqa_swap_seqlen_head:
                    lse = lse.view(1, G, Mq, Hq).permute(0, 1, 3, 2)
                else:
                    lse = lse.view(Bqq, G, Hq, Mq)
                    if attn_bias is not None and not variable_q:
                        lse = lse.permute(1, 2, 0, 3).reshape(1, G, Hq, B * Mq)
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

        out = torch.empty(output_shape, device=q.device, dtype=output_dtype)

        # Merge attention and LSE outputs from different split-k chunks
        assert lse_splitk is not None
        output_lse = None
        if needs_gradient:
            lse_dtype = torch.float64 if output_f64_lse else torch.float32
            if attn_bias is None or variable_q:
                output_lse = torch.empty(
                    (Bqq, G, Hq, Mq), device=q.device, dtype=lse_dtype
                )
                lse = output_lse
            else:
                output_lse = torch.empty(
                    (1, G, Hq, B * Mq), device=q.device, dtype=lse_dtype
                )
                lse = output_lse.view(G, Hq, B, Mq).permute(2, 0, 1, 3)

        o_splitk = o_splitk[:, :, :, :, :Mqq]

        if mqa_swap_seqlen_head:
            if variable_q:
                o_splitk = o_splitk.view(Bqq, G, split_k, Mq, Hq, Kq).permute(
                    0, 1, 4, 2, 3, 5
                )
                lse_splitk = lse_splitk.view(Bqq, G, split_k, Mq, Hq).permute(
                    0, 1, 4, 2, 3
                )
            else:
                o_splitk = o_splitk.view(Bqq, G, split_k, Hq, Mq, Kq).permute(
                    0, 1, 3, 2, 4, 5
                )
                lse_splitk = lse_splitk.view(Bqq, G, split_k, Hq, Mq).permute(
                    0, 1, 3, 2, 4
                )

        merge_attentions(out, lse, o_splitk, lse_splitk)

        if inp.query.ndim == 4:
            # BMGHK -> BMHK
            assert G == 1
            out = out[:, :, 0]
            if output_lse is not None:
                output_lse = output_lse[:, 0]
        if Mk == 0:
            out.zero_()

        if attn_bias is not None and not variable_q:
            out = out.view(1, B * Mq, G, Hq, Kq)

        if output_lse is None:
            return out, None

        return out, Context(out=out, lse=output_lse)

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
        split_k_early_exit: Optional[bool] = None,
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
        if split_k_early_exit is not None:
            kwargs["SPLIT_K_EARLY_EXIT"] = split_k_early_exit
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
    B1, G1, H1, split_k, M1, Kq1 = attn_split.shape
    B2, G2, H2, split_k1, M2 = lse_split.shape

    assert (
        B == B1 == B2
        and G == G1 == G2
        and H == H1 == H2
        and M == M1 == M2
        and Kq == Kq1
    ), f"Incompatible shapes: {attn_out.shape=}, {attn_split.shape=}, {lse_split.shape=}"
    assert (
        split_k == split_k1
    ), f"Incompatible shapes: {attn_split.shape=}, {lse_split.shape=}"
    if lse_out is not None:
        B3, G3, H3, M3 = lse_out.shape
        assert (
            B == B3 and G == G3 and H == H3 and M == M3
        ), f"Incompatible shapes: {attn_out.shape=}, {lse_out.shape=}"

    num_warps = 4 if B * G * H < 32 or torch.version.hip else 2
    splitK_pow2 = triton.next_power_of_2(split_k)
    grid = (M, B * G * H, 1)
    _splitK_reduce[grid](
        attn_split,
        lse_split,
        attn_out,
        lse_out,
        split_k=split_k,
        splitK_pow2=splitK_pow2,
        **_strides(attn_split, "osk_z", "osk_g", "osk_h", "osk_s", "osk_m", "osk_k"),
        **_strides(lse_split, "lsek_z", "lsek_g", "lsek_h", "lsek_s", "lsek_m"),
        **_strides(attn_out, "oz", "om", "og", "oh", "ok"),
        **_strides(lse_out, "lse_z", "lse_g", "lse_h", "lse_m"),
        BLOCK_SIZE=attn_out.shape[-1],
        G=G,
        H=H,
        WRITE_LSE=lse_out is not None,
        num_warps=num_warps,
    )


def merge_attentions_varargs(
    attn_out: torch.Tensor,
    lse_out: Optional[torch.Tensor],
    attn_split: Sequence[torch.Tensor],
    lse_split: Sequence[torch.Tensor],
):
    kernel_args, grid = _prepare_reduce_kernel_params(
        attn_out, lse_out, attn_split, lse_split
    )
    reduce_kernel = unroll_varargs(_splitK_reduce_varargs, N=len(attn_split))
    reduce_kernel[grid](
        *attn_split,
        *lse_split,
        Out=attn_out,
        LSE=lse_out,
        **kernel_args,
        BLOCK_SIZE=attn_out.shape[-1],
        WRITE_LSE=lse_out is not None,
    )


def merge_attentions_varargs_backward(
    attn_split: List[torch.Tensor],
    lse_split: List[torch.Tensor],
    attn_out: torch.Tensor,
    lse_out: torch.Tensor,
    grad_attn: torch.Tensor,
    grad_lse: torch.Tensor,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

    dattn_splitk = [torch.empty_like(x) for x in attn_split]
    dlse_splitk = [torch.empty_like(x) for x in lse_split]

    kernel_args, grid = _prepare_reduce_kernel_params(
        attn_out, lse_out, attn_split, lse_split, grad_attn, grad_lse
    )

    reduce_kernel_backward = unroll_varargs(
        _splitK_reduce_varargs_backward, N=len(attn_split)
    )
    reduce_kernel_backward[grid](
        *attn_split,
        *lse_split,
        *dattn_splitk,
        *dlse_splitk,
        Out=attn_out,
        LSE=lse_out,
        DOut=grad_attn,
        DLSE=grad_lse,
        **kernel_args,
        BLOCK_SIZE=attn_out.shape[-1],
    )

    return dattn_splitk, dlse_splitk


def _prepare_reduce_kernel_params(
    attn_out: torch.Tensor,
    lse_out: Optional[torch.Tensor],
    attn_split: Sequence[torch.Tensor],
    lse_split: Sequence[torch.Tensor],
    grad_attn: Optional[torch.Tensor] = None,
    grad_lse: Optional[torch.Tensor] = None,
) -> Tuple[Dict[str, int], Tuple[int, int, int]]:

    B, M, G, H, Kq = attn_out.shape
    B1, G1, H1, M1, Kq1 = attn_split[0].shape
    B2, G2, H2, M2 = lse_split[0].shape

    assert (
        B == B1 == B2
        and G == G1 == G2
        and H == H1 == H2
        and M == M1 == M2
        and Kq == Kq1
    ), f"Incompatible shapes: {attn_out.shape=}, {attn_split[0].shape=}, {lse_split[0].shape=}"
    if lse_out is not None:
        B3, G3, H3, M3 = lse_out.shape
        assert (
            B == B3 and G == G3 and H == H3 and M == M3
        ), f"Incompatible shapes: {attn_out.shape=}, {lse_out.shape=}"

    attn_split_strides = {}
    lse_split_strides = {}
    for i in range(len(attn_split)):
        attn_split_strides.update(
            _strides(
                attn_split[i],
                "osk_z" + str(i),
                "osk_g" + str(i),
                "osk_h" + str(i),
                "osk_m" + str(i),
                "osk_k" + str(i),
            )
        )
        lse_split_strides.update(
            _strides(
                lse_split[i],
                "lsek_z" + str(i),
                "lsek_g" + str(i),
                "lsek_h" + str(i),
                "lsek_m" + str(i),
            )
        )

    num_warps = 4 if B * G * H < 32 or torch.version.hip else 2
    grid = (M, B * G * H, 1)

    kernel_args = {
        "G": G,
        "H": H,
        "num_warps": num_warps,
        **attn_split_strides,
        **lse_split_strides,
    }
    kernel_args.update(_strides(attn_out, "oz", "om", "og", "oh", "ok"))
    kernel_args.update(_strides(lse_out, "lse_z", "lse_g", "lse_h", "lse_m"))
    if grad_attn is not None:
        kernel_args.update(_strides(grad_attn, "doz", "dom", "dog", "doh", "dok"))
        kernel_args.update(_strides(grad_lse, "dlse_z", "dlse_g", "dlse_h", "dlse_m"))
    return kernel_args, grid


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
