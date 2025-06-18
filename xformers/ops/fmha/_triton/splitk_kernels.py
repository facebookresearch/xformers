# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import functools
import sys
from typing import Callable, Dict, Tuple, Union

import torch
import triton
import triton.language as tl

from xformers.triton.vararg_kernel import unroll_varargs, VAR_ARGS_ARRAY

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
    "PAGE_SIZE",
]


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
    K_fp8_scale_shift,
    V_fp8_scale_shift,
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
    stride_k_fp8_scale_shift_z: tl.constexpr,
    stride_k_fp8_scale_shift_n: tl.constexpr,
    stride_k_fp8_scale_shift_g: tl.constexpr,
    stride_k_fp8_scale_shift_h: tl.constexpr,
    stride_v_fp8_scale_shift_z: tl.constexpr,
    stride_v_fp8_scale_shift_n: tl.constexpr,
    stride_v_fp8_scale_shift_g: tl.constexpr,
    stride_v_fp8_scale_shift_h: tl.constexpr,
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
    IS_LOCAL: tl.constexpr,
    NUM_QUERIES_CAUSAL: tl.constexpr,  # The N_CTX_Q queries are from this many sequence positions
    USE_PAGED_ATTENTION: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    WINDOW_RIGHT: tl.constexpr,
    WRITE_LSE: tl.constexpr,
    HAS_ADDITIVE_BIAS: tl.constexpr,
    NUM_PROGRAMS_DIM2_CONST: tl.constexpr,
    IS_HIP: tl.constexpr,
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
        or (
            (PACKED_PER_VAL == 4 or PACKED_PER_VAL == 8)
            and tl.constexpr(K.dtype.element_ty == tl.int32)
        ),
        f"Only int4 and fp8 quantization is supported, K/V should have dtype int32 in "
        f"the quantized case: {PACKED_PER_VAL=} {tl.constexpr(K.dtype)=} {tl.constexpr(K.dtype.element_ty)=}",
    )
    tl.static_assert(
        (((N_GROUPS == 1 or N_GROUPS == 2) or N_GROUPS == 4) or N_GROUPS == 8),
        "Number of quantization groups can be 1 (row-wise quantization), 2, 4, or 8.",
    )
    tl.static_assert(
        N_GROUPS == 1 or K_fp8_scale_shift is None,
        f"Only row-wise fp8 quantization is supported, but got {N_GROUPS=} > 1.",
    )
    FP8_QUANTIZED: tl.constexpr = K_fp8_scale_shift is not None
    INT4_QUANTIZED: tl.constexpr = PACKED_PER_VAL > 1 and not FP8_QUANTIZED
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
        if USE_SEQ_LEN and PAGE_SIZE > 0:
            # gappy with paged attention stores each "end" instead of each "length"
            # because that's what FA3 needs.
            kv_len -= start_kv_idx

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

    if FP8_QUANTIZED:
        k_fp8_scale_shift_base = (
            K_fp8_scale_shift
            + off_h * stride_k_fp8_scale_shift_h
            + off_g * stride_k_fp8_scale_shift_g
        )
        v_fp8_scale_shift_base = (
            V_fp8_scale_shift
            + off_h * stride_v_fp8_scale_shift_h
            + off_g * stride_v_fp8_scale_shift_g
        )
    else:
        k_fp8_scale_shift_base = None
        v_fp8_scale_shift_base = None

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
        # TODO: Replace NUM_PROGRAMS_DIM2_CONST with tl.num_programs(2) after
        # the next Triton upgrade.
        is_last_chunk = splitk_idx == NUM_PROGRAMS_DIM2_CONST - 1
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
            base=k_base + stride_kk * INT4_QUANTIZED * N_GROUPS,
            shape=(PACKED_D_PER_GROUP, hi),
            strides=(stride_kk, stride_kn),
            offsets=(0, lo),
            block_shape=(PACKED_D_PER_GROUP, BLOCK_N),
            order=(0, 1),
        )
        V_block_ptr = tl.make_block_ptr(
            base=v_base + stride_vk * INT4_QUANTIZED * N_GROUPS,
            shape=(hi, PACKED_D_PER_GROUP),
            strides=(stride_vn, stride_vk),
            offsets=(lo, 0),
            block_shape=(BLOCK_N, PACKED_D_PER_GROUP),
            order=(1, 0),
        )

        if INT4_QUANTIZED:
            # Pointers to quantization coefficients. Even those they are 1D,
            # we use block pointers here so the pointer arithmetic is in int64,
            # as otherwise the offsets for V_scale_shift_block_ptr may overflow.
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
        elif FP8_QUANTIZED:
            if Seq_starts_k is not None:
                k_fp8_scale_shift_base += start_kv_idx * stride_k_fp8_scale_shift_n
                v_fp8_scale_shift_base += start_kv_idx * stride_v_fp8_scale_shift_n
            else:
                k_fp8_scale_shift_base += off_z * stride_k_fp8_scale_shift_z
                v_fp8_scale_shift_base += off_z * stride_v_fp8_scale_shift_z
            K_scale_shift_block_ptr = tl.make_block_ptr(
                base=k_fp8_scale_shift_base,
                shape=(1, hi),
                strides=(1, stride_k_fp8_scale_shift_n),
                offsets=(0, lo),
                block_shape=(1, BLOCK_N),
                order=(0, 1),
            )
            V_scale_shift_block_ptr = tl.make_block_ptr(
                base=v_fp8_scale_shift_base,
                shape=(hi, 1),
                strides=(stride_v_fp8_scale_shift_n, 1),
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
        acc[i] = tl.zeros([BLOCK_M, D_PER_GROUP], dtype=internal_dtype)  # noqa: F821
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    #
    # We declare log2e as a constant with a precisely-specified type to guarantee that
    # triton will use the exact same value in all instances below, rather than sometimes
    # using float32 and sometimes using float64.  For more discussion see:
    # https://github.com/triton-lang/triton/issues/5466
    log2e = tl.full((), 1.44269504, tl.float32)
    qk_scale = sm_scale * log2e
    # load q: it will stay in SRAM throughout
    q: "VAR_ARGS_ARRAY"  # noqa: F821
    for i in range(len(acc)):  # noqa: F821
        q[i] = tl.load(  # noqa: F821
            tl.advance(Q_block_ptr, (0, i * D_PER_GROUP)), boundary_check=(0,)
        )

    if IS_CAUSAL or IS_LOCAL:
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
                base=k_base + stride_kk * INT4_QUANTIZED * N_GROUPS,
                shape=(PACKED_D_PER_GROUP, offset + current_block_size),
                strides=(stride_kk, stride_kn),
                offsets=(0, offset),
                block_shape=(PACKED_D_PER_GROUP, BLOCK_N),
                order=(0, 1),
            )
            V_block_ptr = tl.make_block_ptr(
                base=v_base + stride_vk * INT4_QUANTIZED * N_GROUPS,
                shape=(offset + current_block_size, PACKED_D_PER_GROUP),
                strides=(stride_vn, stride_vk),
                offsets=(offset, 0),
                block_shape=(BLOCK_N, PACKED_D_PER_GROUP),
                order=(1, 0),
            )
            if INT4_QUANTIZED:
                # Pointers to quantization coefficients. Even those they are 1D,
                # we use block pointers here so the pointer arithmetic is in int64,
                # as otherwise the offsets for V_scale_shift_block_ptr may overflow.
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
            elif FP8_QUANTIZED:
                K_scale_shift_block_ptr = tl.make_block_ptr(
                    base=k_fp8_scale_shift_base,
                    shape=(1, offset + current_block_size),
                    strides=(1, stride_k_fp8_scale_shift_n),
                    offsets=(0, offset),
                    block_shape=(1, BLOCK_N),
                    order=(0, 1),
                )
                V_scale_shift_block_ptr = tl.make_block_ptr(
                    base=v_fp8_scale_shift_base,
                    shape=(offset + current_block_size, 1),
                    strides=(stride_v_fp8_scale_shift_n, 1),
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
                FP8_QUANTIZED,
                Q.dtype.element_ty,
                i,
                IS_HIP,
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
            qk += loaded_bias.to(tl.float32) * log2e
            additive_bias_block_ptr = tl.advance(additive_bias_block_ptr, (0, BLOCK_N))

        # TODO: This is slow, and only needed at the last iteration.
        # Maybe we can unroll the last iteration instead?
        if BOUNDS_CHECKS_N:
            qk = tl.where(tl.arange(0, BLOCK_N) < hi - start_n, qk, float("-inf"))
        if IS_CAUSAL:
            # -- apply the causal mask --
            qk = tl.where(diag_idx_shifted >= start_n - start_kv_idx, qk, float("-inf"))
        if IS_LOCAL:
            # -- apply the local window size mask --
            qk = tl.where(
                diag_idx_shifted < start_n - start_kv_idx + WINDOW_LEFT + 1,
                qk,
                float("-inf"),
            )
            if not IS_CAUSAL and WINDOW_RIGHT >= 0:
                qk = tl.where(
                    diag_idx_shifted >= start_n - start_kv_idx - WINDOW_RIGHT,
                    qk,
                    float("-inf"),
                )

        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        if HAS_ADDITIVE_BIAS or (IS_CAUSAL or IS_LOCAL):
            # NOTE: It's possible that an entire block is masked out.
            # if this is the case, `m_i_new=nan` and everything becomes nan
            alpha = tl.where(m_i_new == float("-inf"), 0, alpha)
            p = tl.where(m_i_new[:, None] == float("-inf"), 0, p)

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
        attn_out = tl.where(l_i[:, None] == 0, 0.0, acc[i] / l_i[:, None])  # noqa: F821
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
            (tl.math.log2(l_i.to(lse_dtype)) + m_i.to(lse_dtype)) / log2e,
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
                or (
                    args["BLOCK_N_PER_SPLIT"] > 0
                    and args["N_CTX_K"] % args["BLOCK_N_PER_SPLIT"]
                )
                or args["USE_SEQ_LEN"]
            )
        }
    )(_fwd_kernel_splitK_unrolled)
    return kernel


def early_config_prune(configs, named_args, **kwargs):
    use_paged_attention = kwargs["USE_PAGED_ATTENTION"]
    page_size = kwargs["PAGE_SIZE"]
    if use_paged_attention:
        return list(
            filter(lambda config: page_size % config.kwargs["BLOCK_N"] == 0, configs)
        )
    else:
        return configs


@functools.lru_cache(None)
def autotune_kernel(kernel: Callable):
    BLOCK_M_VALUES = [16, 32, 64, 128]
    BLOCK_N_VALUES = [16, 32, 64, 128]
    STAGES_VALUES = [1, 2] if torch.version.hip else [1, 2, 3]
    WARPS_VALUES = [1, 2, 4, 8]

    TRITON_CONFIGS = [
        gen_config(block_m, block_n, stages, warps)
        for block_m in BLOCK_M_VALUES
        for block_n in BLOCK_N_VALUES
        for stages in STAGES_VALUES
        for warps in WARPS_VALUES
        if block_n >= block_m
    ]

    kernel = triton.autotune(
        configs=TRITON_CONFIGS,
        key=AUTOTUNER_KEY,
        use_cuda_graph=True,
        prune_configs_by={
            "early_config_prune": early_config_prune,
        },
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

    def get_autotuner_cache(
        num_groups: int,
    ) -> Dict[Tuple[Union[int, str]], triton.Config]:
        """Returns a triton.runtime.autotuner.AutoTuner.cache object, which
        represents mappings from kernel autotune keys (tuples describing kernel inputs)
        to triton.Config
        """
        return _fwd_kernel_splitK_autotune[num_groups].cache

    def set_autotuner_cache(
        cache: Dict[Tuple[Union[int, str]], triton.Config], num_groups: int
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
    FP8_QUANTIZED: tl.constexpr,
    dtype: tl.constexpr,
    group_id: tl.constexpr,
    IS_HIP: tl.constexpr,
):
    """Load K/V for a given block. In case of int4/fp8-quantized K/V, dequantize them after loading.
    If quantization is group-wise, use group_id to advance the pointers to the current group.
    """
    # Advance to the current quantization group
    K_block_ptr = tl.advance(K_block_ptr, (PACKED_D_PER_GROUP * group_id, 0))
    V_block_ptr = tl.advance(V_block_ptr, (0, PACKED_D_PER_GROUP * group_id))

    # -- load k, v --
    k = tl.load(K_block_ptr, boundary_check=(1,) if BOUNDS_CHECKS_N else ())
    v = tl.load(V_block_ptr, boundary_check=(0,) if BOUNDS_CHECKS_N else ())

    # If K/V are quantized, load quantization coefficients and dequantize.
    if FP8_QUANTIZED:
        v_scale_shift = tl.load(
            V_scale_shift_block_ptr, boundary_check=(0,) if BOUNDS_CHECKS_N else ()
        )
        if IS_HIP:
            # MI300 doesn't have builtin casting instructions for fp8 -> bf16,
            # so casting to f32 is actually more performant on this workload.
            v_scale, v_shift = cast_uint32_to_float(v_scale_shift)
        else:
            v_scale, v_shift = cast_uint32_to_half2(v_scale_shift)
        v = dequantize(v, v_scale, v_shift, PACKED_PER_VAL, IS_HIP).to(dtype)

        k_scale_shift = tl.load(
            K_scale_shift_block_ptr, boundary_check=(1,) if BOUNDS_CHECKS_N else ()
        )
        if IS_HIP:
            k_scale, k_shift = cast_uint32_to_float(k_scale_shift)
            k = dequantize_k_hip(k, k_scale, k_shift, PACKED_PER_VAL).to(dtype)
        else:
            k_scale, k_shift = cast_uint32_to_half2(k_scale_shift)
            k_t = dequantize(
                tl.trans(k),
                tl.trans(k_scale),
                tl.trans(k_shift),
                PACKED_PER_VAL,
                IS_HIP,
            ).to(dtype)
            k = tl.trans(k_t)
    elif PACKED_PER_VAL > 1:
        # Int4 quantization.
        K_scale_shift_block_ptr = tl.advance(K_scale_shift_block_ptr, (group_id, 0))
        V_scale_shift_block_ptr = tl.advance(V_scale_shift_block_ptr, (0, group_id))

        k_scale_shift = tl.load(
            K_scale_shift_block_ptr, boundary_check=(1,) if BOUNDS_CHECKS_N else ()
        )
        v_scale_shift = tl.load(
            V_scale_shift_block_ptr, boundary_check=(0,) if BOUNDS_CHECKS_N else ()
        )
        if IS_HIP:
            k_scale, k_shift = cast_uint32_to_float(k_scale_shift)
            v_scale, v_shift = cast_uint32_to_float(v_scale_shift)
            v = dequantize(v, v_scale, v_shift, PACKED_PER_VAL, IS_HIP).to(dtype)
            k = dequantize_k_hip(k, k_scale, k_shift, PACKED_PER_VAL).to(dtype)
        else:
            k_scale, k_shift = cast_uint32_to_half2(k_scale_shift)
            v_scale, v_shift = cast_uint32_to_half2(v_scale_shift)
            v = dequantize(v, v_scale, v_shift, PACKED_PER_VAL, IS_HIP).to(dtype)
            k_t = dequantize(
                tl.trans(k),
                tl.trans(k_scale),
                tl.trans(k_shift),
                PACKED_PER_VAL,
                IS_HIP,
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
def cast_uint32_to_float(scale_shift):
    """Extract two float16 packed into one int32 as float32"""
    scale = scale_shift & 0xFFFF
    shift = scale_shift >> 16
    scale = scale.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    shift = shift.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    return scale, shift


@triton.jit
def dequantize_k_hip(
    x_,
    scale,
    shift,
    PACKED_PER_VAL: tl.constexpr,
):
    """PACKED_PER_VAL is the number of values packed into each element x_.
    For example, for int4 quantization and x_ of type int32, PACKED_PER_VAL is 8.
    """
    # x_ : (BLOCK_N, D // PACKED_PER_VAL)
    # scale: (BLOCK_N, 1)
    # offsets: (PACKED_PER_VAL,)
    BLOCK_N: tl.constexpr = x_.shape[1]
    BLOCK_DMODEL_PACKED: tl.constexpr = x_.shape[0]
    offsets = tl.arange(0, PACKED_PER_VAL) * (32 // PACKED_PER_VAL)
    quant_offset = (
        x_[:, None, :, :] >> offsets[:, None]
    )  # (BLOCK_N, D // PACKED_PER_VAL, PACKED_PER_VAL)

    quant_offset = tl.reshape(
        quant_offset, (BLOCK_DMODEL_PACKED * PACKED_PER_VAL, BLOCK_N)
    )

    if PACKED_PER_VAL == 4:
        # FP8 quantization.
        fp8_type = tl.float8e4b8 if torch.version.hip is not None else tl.float8e4nv
        dequant = (
            quant_offset.to(tl.uint8).to(fp8_type, bitcast=True).to(scale.dtype) * scale
            + shift
        )
    else:
        # Int4 quantization.
        # Trick - instead of converting int4 to float16 we view it as float16
        # and then multiply by 32768 * 512 == 2**24
        quant_offset = (
            (quant_offset & 0xF)
            .to(tl.uint16)
            .to(tl.float16, bitcast=True)
            .to(tl.float32)
        )
        quant_offset = quant_offset * 32768.0
        scale_512 = scale * 512

        dequant = quant_offset * scale_512 + shift
    return dequant


@triton.jit
def dequantize(
    x_,
    scale,
    shift,
    PACKED_PER_VAL: tl.constexpr,
    IS_HIP: tl.constexpr,
):
    """PACKED_PER_VAL is the number of values packed into each element x_.
    For example, for int4 quantization and x_ of type int32, PACKED_PER_VAL is 8.
    """
    # x_ : (BLOCK_N, D // PACKED_PER_VAL)
    # scale: (BLOCK_N, 1)
    # offsets: (PACKED_PER_VAL,)
    BLOCK_N: tl.constexpr = x_.shape[0]
    BLOCK_DMODEL_PACKED: tl.constexpr = x_.shape[1]
    offsets = tl.arange(0, PACKED_PER_VAL) * (32 // PACKED_PER_VAL)
    quant_offset = (
        x_[:, :, None, :] >> offsets
    )  # (BLOCK_N, D // PACKED_PER_VAL, PACKED_PER_VAL)

    quant_offset = tl.reshape(
        quant_offset, (BLOCK_N, BLOCK_DMODEL_PACKED * PACKED_PER_VAL)
    )
    if PACKED_PER_VAL == 4:
        # FP8 quantization.
        fp8_type = tl.float8e4b8 if torch.version.hip is not None else tl.float8e4nv
        dequant = (
            quant_offset.to(tl.uint8).to(fp8_type, bitcast=True).to(scale.dtype) * scale
            + shift
        )
    else:
        # Int4 quantization.
        # Trick - instead of converting int4 to float16 we view it as float16
        # and then multiply by 32768 * 512 == 2**24
        if IS_HIP:
            # Do final math in float32 to avoid casting to bf16 on MI300. There
            # no direct instructions for this so its less performant on this workload.
            quant_offset = (
                (quant_offset & 0xF)
                .to(tl.uint16)
                .to(tl.float16, bitcast=True)
                .to(tl.float32)
            )
            quant_offset = quant_offset * 32768.0
        else:
            quant_offset = (
                (quant_offset & 0xF).to(tl.uint16).to(tl.float16, bitcast=True)
            )
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
    stride_osk_z: tl.constexpr,
    stride_osk_g: tl.constexpr,
    stride_osk_h: tl.constexpr,
    stride_osk_s: tl.constexpr,
    stride_osk_m: tl.constexpr,
    stride_osk_k: tl.constexpr,
    stride_lsek_z: tl.constexpr,
    stride_lsek_g: tl.constexpr,
    stride_lsek_h: tl.constexpr,
    stride_lsek_s: tl.constexpr,
    stride_lsek_m: tl.constexpr,
    stride_oz: tl.constexpr,
    stride_og: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_ok: tl.constexpr,
    stride_lse_z: tl.constexpr,
    stride_lse_g: tl.constexpr,
    stride_lse_h: tl.constexpr,
    stride_lse_m: tl.constexpr,
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
