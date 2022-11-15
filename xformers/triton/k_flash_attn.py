# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import triton
import triton.language as tl

# CREDITS: Inspired by the Triton tutorial on fused attention.


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    is_causal,
    TMP,
    L,
    M,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
    Out,
    stride_q_batch,
    stride_q_heads,
    stride_qm,
    stride_qk,
    stride_k_batch,
    stride_k_heads,
    stride_kn,
    stride_kk,
    stride_v_batch,
    stride_v_heads,
    stride_vk,
    stride_vn,
    stride_o_batch,
    stride_o_heads,
    stride_om,
    stride_od,
    B,
    H,
    seqlen_q,
    seqlen_k,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):

    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    current_batch_idx = off_hb // H
    current_head_idx = off_hb % H

    # Load current q block
    off_q = (
        current_batch_idx * stride_q_batch
        + current_head_idx * stride_q_heads
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qk
    )
    # Load first k block
    off_k = (
        current_batch_idx * stride_k_batch
        + current_head_idx * stride_k_heads
        + offs_n[:, None] * stride_kn
        + offs_d[None, :] * stride_kk
    )
    # Load first v block
    off_v = (
        current_batch_idx * stride_v_batch
        + current_head_idx * stride_v_heads
        + offs_n[:, None] * stride_qm  # stride_vk
        + offs_d[None, :] * stride_qk  # stride_vn
    )
    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    # initialize pointer to m and l
    t_ptrs = TMP + off_hb * seqlen_q + offs_m
    # keeps track of maxs per row
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    # keeps track of denominator per row
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    # keeps track of output, block of full rows
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs)
    # loop over k, v and update accumulator
    end_n = seqlen_k
    if is_causal:
        end_n = tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        # load current k block
        k = tl.load(k_ptrs + start_n * stride_kn)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k, trans_b=True)
        qk *= sm_scale
        if is_causal:
            qk += tl.where(
                offs_m[:, None] >= (start_n + offs_n[None, :]), 0, float("-inf")
            )
        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        tl.store(t_ptrs, acc_scale)
        acc_scale = tl.load(t_ptrs)  # BUG: have to store and immediately load
        acc = acc * acc_scale[:, None]
        # update acc
        # load current v block
        v = tl.load(v_ptrs + start_n * stride_vk)
        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new
    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m, needed for bwd
    l_ptrs = L + off_hb * seqlen_q + offs_m
    m_ptrs = M + off_hb * seqlen_q + offs_m
    tl.store(l_ptrs, l_i)
    tl.store(m_ptrs, m_i)
    # initialize pointers to output
    off_o = (
        current_batch_idx * stride_o_batch
        + current_head_idx * stride_o_heads
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc)


@triton.jit
def _bwd_preprocess(
    Out,
    DO,
    L,
    NewDO,
    Delta,
    BLOCK_M: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)
    # load
    o = tl.load(Out + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    denom = tl.load(L + off_m).to(tl.float32)
    # compute
    do = do / denom[:, None]
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(NewDO + off_m[:, None] * D_HEAD + off_n[None, :], do)
    tl.store(Delta + off_m, delta)


@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    Out,
    DO,
    DQ,
    DK,
    DV,
    L,
    M,
    D,
    stride_q_batch,
    stride_q_heads,
    stride_qm,
    stride_qk,
    stride_k_batch,
    stride_k_heads,
    stride_kn,
    stride_kk,
    stride_v_batch,
    stride_v_heads,
    stride_vk,
    stride_vn,
    B,
    H,
    seqlen_q,
    num_block,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    is_causal: tl.constexpr,
):
    off_hb = tl.program_id(0)
    off_b = off_hb // H
    off_h = off_hb % H
    # offset pointers for batch/head
    Q += off_b * stride_q_batch + off_h * stride_q_heads
    K += off_b * stride_q_batch + off_h * stride_q_heads
    V += off_b * stride_q_batch + off_h * stride_q_heads
    DO += off_b * stride_q_batch + off_h * stride_q_heads
    DQ += off_b * stride_q_batch + off_h * stride_q_heads
    DK += off_b * stride_q_batch + off_h * stride_q_heads
    DV += off_b * stride_q_batch + off_h * stride_q_heads
    for start_n in range(0, num_block):
        lo = start_n * BLOCK_M if is_causal else 0
        # initialize row/col offsets
        offs_qm = lo + tl.arange(0, BLOCK_M)
        offs_n = start_n * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_DMODEL)
        # initialize pointers to value-like data
        q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        k_ptrs = K + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        v_ptrs = V + (offs_n[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        do_ptrs = DO + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        dq_ptrs = DQ + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        # pointer to row-wise quantities in value-like data
        D_ptrs = D + off_hb * seqlen_q
        m_ptrs = M + off_hb * seqlen_q
        # initialize dv amd dk
        dv = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        dk = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        # k and v stay in SRAM throughout
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        # loop over rows
        for start_m in range(lo, num_block * BLOCK_M, BLOCK_M):
            offs_m_curr = start_m + offs_m
            # load q, k, v, do on-chip
            q = tl.load(q_ptrs)
            # recompute p = softmax(qk, dim=-1).T
            # NOTE: `do` is pre-divided by `l`; no normalization here
            qk = tl.dot(q, k, trans_b=True)
            if is_causal:
                qk = tl.where(
                    offs_m_curr[:, None] >= (offs_n[None, :]), qk, float("-inf")
                )
            m = tl.load(m_ptrs + offs_m_curr)
            p = tl.exp(qk * sm_scale - m[:, None])
            # compute dv
            do = tl.load(do_ptrs)
            dv += tl.dot(p.to(do.dtype), do, trans_a=True)
            # compute dp = dot(v, do)
            Di = tl.load(D_ptrs + offs_m_curr)
            dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
            dp += tl.dot(do, v, trans_b=True)
            # compute ds = p * (dp - delta[:, None])
            # Converting ds to q.dtype here reduces register pressure and makes it much faster
            # for BLOCK_HEADDIM=128
            ds = (p * dp * sm_scale).to(q.dtype)
            # compute dk = dot(ds.T, q)
            dk += tl.dot(ds, q, trans_a=True)
            # # compute dq
            dq = tl.load(dq_ptrs, eviction_policy="evict_last")
            dq += tl.dot(ds, k)
            tl.store(dq_ptrs, dq, eviction_policy="evict_last")
            # # increment pointers
            dq_ptrs += BLOCK_M * stride_qm
            q_ptrs += BLOCK_M * stride_qm
            do_ptrs += BLOCK_M * stride_qm
        # write-back
        dv_ptrs = DV + (offs_n[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        dk_ptrs = DK + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        tl.store(dv_ptrs, dv)
        tl.store(dk_ptrs, dk)
