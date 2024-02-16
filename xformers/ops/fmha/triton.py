# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Triton Flash Attention 2
Based on
https://github.com/openai/triton/blob/293b7fd592a1602f2305c1bd0bc978bbd97337d6/python/tutorials/06-fused-attention.py  # noqa: E501
https://github.com/openai/triton/blob/293b7fd592a1602f2305c1bd0bc978bbd97337d6/python/triton/ops/flash_attention.py  # noqa: E501
https://github.com/Dao-AILab/flash-attention/blob/dd9a6fa45a9b90ff954d2b3f3f44241b9216190e/flash_attn/flash_attn_triton.py  # noqa: E501
https://github.com/ROCmSoftwarePlatform/triton/blob/670ae8054da008424097989a5b6e3816aa601e07/python/perf-kernels/06-fused-attention-transV.py  # noqa: E501
"""

from dataclasses import replace
from typing import Any, List, Mapping, Optional, Set, Tuple

import torch

from xformers import _is_triton_available

from ..common import register_operator
from .attn_bias import (
    BlockDiagonalCausalMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    LowerTriangularMask,
)
from .common import AttentionFwOpBase, Context, Inputs, check_lastdim_alignment_stride1

if _is_triton_available():
    import triton
    import triton.language as tl

    @triton.jit
    def _fwd_kernel_triton_flash_inner(
        acc,
        l_i,
        m_i,
        q,
        K_block_ptr,
        V_block_ptr,
        q_seq_start,
        lo,
        hi,
        start_m,
        qk_scale,
        kv_len,
        offs_m,
        offs_n,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        BOUNDS_CHECKS_N: tl.constexpr,
        CAST_BEFORE_MATMUL: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
        STAGE: tl.constexpr,
        pre_load_v: tl.constexpr,
    ):
        BOUNDS_CHECKS_STAGE: tl.constexpr = BOUNDS_CHECKS_N and STAGE == 2
        # Doesn't seem to make a difference
        if STAGE == 1:
            lo = 0
        else:
            lo = tl.multiple_of(lo, BLOCK_N)
            K_block_ptr = tl.advance(K_block_ptr, (0, lo))
            V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

        # loop over k, v and update accumulator
        for start_n in range(lo, hi, BLOCK_N):
            start_n = tl.multiple_of(
                start_n, BLOCK_N
            )  # doesn't seem to make a difference
            # -- load k, v --
            k = tl.load(K_block_ptr, boundary_check=(1,) if BOUNDS_CHECKS_STAGE else ())
            # Moving masking here seems to introduce num errors,
            # e.g. in test_forward[tritonflashattF-cuda-torch.bfloat16-NoneType-1-256-15-1-32-32-False-BMHK]
            # if BOUNDS_CHECKS_N or USE_SEQ_LEN:
            #     k = tl.where(hi - tl.arange(0, BLOCK_N) > start_n, k, float("-inf"))
            if pre_load_v:
                v = tl.load(
                    V_block_ptr, boundary_check=(0,) if BOUNDS_CHECKS_STAGE else ()
                )
            # -- compute qk ---
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q.to(k.dtype), k, allow_tf32=ALLOW_TF32) * qk_scale
            if CAST_BEFORE_MATMUL:
                k = k.to(tl.float32)
            if STAGE == 2:
                if IS_CAUSAL:
                    # For some reason this is faster than start_n <= q_seq_start + offs_m[:, None] - offs_n[None, :]
                    qk = tl.where(
                        q_seq_start + offs_m[:, None] >= (start_n + offs_n[None, :]),
                        qk,
                        float("-inf"),
                    )
                if BOUNDS_CHECKS_N:
                    qk = tl.where(
                        tl.arange(0, BLOCK_N) < hi - start_n, qk, float("-inf")
                    )

            # -- compute scaling constant ---
            m_i_new = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_i_new[:, None]
            alpha = tl.math.exp2(m_i - m_i_new)
            p = tl.math.exp2(qk)

            # -- scale and update acc --
            acc *= alpha[:, None]
            if not pre_load_v:
                v = tl.load(
                    V_block_ptr, boundary_check=(0,) if BOUNDS_CHECKS_STAGE else ()
                )
            if CAST_BEFORE_MATMUL:
                v = v.to(tl.float32)
            acc += tl.dot(p.to(v.dtype), v, allow_tf32=ALLOW_TF32)
            # -- update m_i and l_i --
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_i_new
            # update pointers
            K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
            V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        return acc, l_i, m_i

    @triton.jit
    def _fwd_kernel_triton_flash(
        Q,
        K,
        V,
        sm_scale,
        L,
        Out,
        Seq_len,
        Seq_pos_q,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vk,
        stride_vn,
        stride_oz,
        stride_oh,
        stride_om,
        stride_on,
        Z,
        H,
        N_CTX,
        Mkv,
        BLOCK_M: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        BOUNDS_CHECKS_N: tl.constexpr,
        BOUNDS_CHECKS_M: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
        CAST_BEFORE_MATMUL: tl.constexpr,
        USE_SEQ_LEN_KV: tl.constexpr,
        USE_SEQ_POS_Q: tl.constexpr,
        IS_KV_PADDED: tl.constexpr,  # Switch between padded and non-padded block-diagonal causal masks
        pre_load_v: tl.constexpr,  # TODO: understand if that matters
    ):
        start_m = tl.program_id(0).to(tl.int64)
        off_hz = tl.program_id(1).to(tl.int64)

        tl.static_assert((IS_KV_PADDED and USE_SEQ_POS_Q) or not IS_KV_PADDED)

        off_z = off_hz // H
        off_h = off_hz % H
        if USE_SEQ_POS_Q:
            seqpos = tl.load(Seq_pos_q + off_z)
            seqpos_next = tl.load(Seq_pos_q + off_z + 1)
            q_len = seqpos_next - seqpos
            q_offset = seqpos * stride_qm + off_h * stride_qh
            out_offset = seqpos * stride_om + off_h * stride_oh
            if not IS_KV_PADDED:
                # BlockDiagonalCausalMask, no padding, use same sequence positions as for Q
                kv_offset = seqpos * stride_kn + off_h * stride_kh
                kv_len = q_len
                q_seq_start = 0
            else:
                # BlockDiagonalCausalWithOffsetPaddedKeysMask
                kv_offset = off_z * stride_kz + off_h * stride_kh
                if USE_SEQ_LEN_KV:
                    kv_len = tl.load(Seq_len + off_z)
                    q_seq_start = kv_len - q_len
                else:
                    # if no variable K/V seqlens are provided, assume full length
                    kv_len = Mkv
                    q_seq_start = 0
        else:
            # No mask or simple causal mask
            q_len = N_CTX
            q_offset = off_z * stride_qz + off_h * stride_qh
            out_offset = off_z * stride_oz + off_h * stride_oh

            kv_len = Mkv
            q_seq_start = 0
            kv_offset = off_z * stride_kz + off_h * stride_kh

        Q_block_ptr = tl.make_block_ptr(
            base=Q + q_offset,
            shape=(q_len, BLOCK_DMODEL),
            strides=(stride_qm, stride_qk),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            base=K + kv_offset,
            shape=(BLOCK_DMODEL, kv_len),
            strides=(stride_kk, stride_kn),
            offsets=(0, 0),
            block_shape=(BLOCK_DMODEL, BLOCK_N),
            order=(0, 1),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V + kv_offset,
            shape=(kv_len, BLOCK_DMODEL),
            strides=(stride_vk, stride_vn),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_DMODEL),
            order=(0, 1),
        )
        # initialize offsets
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)  # For Q
        offs_n = tl.arange(0, BLOCK_N)  # For K/V
        # initialize pointer to m and l
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        # scale sm_scale by log_2(e) and use
        # 2^x instead of exp in the loop because CSE and LICM
        # don't work as expected with `exp` in the loop
        qk_scale = sm_scale * 1.44269504
        # load q: it will stay in SRAM throughout on NV GPUs but in VGPRs on AMD GPUs
        q = tl.load(
            Q_block_ptr, boundary_check=(0,) if BOUNDS_CHECKS_M or USE_SEQ_POS_Q else ()
        )

        # The loop over K/V sequence blocks is divided into two stages:
        # Stage 1: (many) blocks which don't need boundary conditions checks - not touching sequence end or diagonal
        # Stage 2: (few) blocks which need boundary conditions checks
        # Following https://github.com/openai/triton/blob/293b7fd592a1602f2305c1bd0bc978bbd97337d6/python/tutorials/06-fused-attention.py  # noqa: E501

        """
        Iteration doesn't need masking if
            - 1) block doesn't cross the diagonal: max(kv_pos) <= min(q_pos)
            - 2) block doesn't cross the end of the sequence: max(kv_pos) < kv_len
        Find maximum start_n for which condition 1 is satisifed.
        Remember that
            q_pos = q_seq_start + offs_m[:, None]
            kv_pos = start_n + offs_n[None, :]
            offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = tl.arange(0, BLOCK_N)
        min(q_pos) = q_seq_start + start_m * BLOCK_M
        max(kv_pos) = start_n + BLOCK_N - 1
        So the condition becomes
            q_seq_start + start_m * BLOCK_M >= start_n + BLOCK_N - 1
        So:
        1) start_n <= q_seq_start + start_m * BLOCK_M - BLOCK_N + 1
        2) start_n <= kv_len - BLOCK_N

        So the last allowed start_n without masking is min(q_seq_start + start_m * BLOCK_M + 1, kv_len) - BLOCK_N
        """
        # Second stage can only be skipped if no mask is used and K/V length is divisible by the tile size
        TWO_STAGES: tl.constexpr = BOUNDS_CHECKS_N or (
            IS_CAUSAL or (USE_SEQ_LEN_KV or (USE_SEQ_POS_Q and not IS_KV_PADDED))
        )
        if TWO_STAGES:
            # Border between two stages
            hi_stage_1 = min(q_seq_start + start_m * BLOCK_M + 1, kv_len) - BLOCK_N
            hi_stage_1 = (
                hi_stage_1 // BLOCK_N
            ) * BLOCK_N  # Don't understand why it doesn't work without this
        else:
            hi_stage_1 = kv_len

        # Stage 1 - no boundary conditions
        acc, l_i, m_i = _fwd_kernel_triton_flash_inner(
            acc,
            l_i,
            m_i,
            q,
            K_block_ptr,
            V_block_ptr,
            q_seq_start,
            0,
            hi_stage_1,
            start_m,
            qk_scale,
            kv_len,
            offs_m,
            offs_n,
            BLOCK_M,
            BLOCK_N,
            IS_CAUSAL,
            BOUNDS_CHECKS_N,
            CAST_BEFORE_MATMUL,
            ALLOW_TF32,
            STAGE=1,
            pre_load_v=pre_load_v,
        )
        if TWO_STAGES:
            hi = (
                tl.minimum(kv_len, q_seq_start + (start_m + 1) * BLOCK_M)
                if IS_CAUSAL
                else kv_len
            )
            # Do we need this barrier?
            # tl.debug_barrier()
            # Stage 2 - with boundary conditions
            acc, l_i, m_i = _fwd_kernel_triton_flash_inner(
                acc,
                l_i,
                m_i,
                q,
                K_block_ptr,
                V_block_ptr,
                q_seq_start,
                hi_stage_1,
                hi,
                start_m,
                qk_scale,
                kv_len,
                offs_m,
                offs_n,
                BLOCK_M,
                BLOCK_N,
                IS_CAUSAL,
                BOUNDS_CHECKS_N,
                CAST_BEFORE_MATMUL,
                ALLOW_TF32,
                STAGE=2,
                pre_load_v=pre_load_v,
            )

        # write back l and m
        acc1 = acc / l_i[:, None]
        l_ptrs = L + off_hz * N_CTX + offs_m
        # Save LSE, converting from log2 to natural logarithm
        l_mask = (
            start_m * BLOCK_M + tl.arange(0, BLOCK_M) < q_len
            if BOUNDS_CHECKS_M
            else None
        )
        tl.store(l_ptrs, (m_i + tl.math.log2(l_i)) / 1.44269504, mask=l_mask)
        # write back O
        O_block_ptr = tl.make_block_ptr(
            base=Out + out_offset,
            shape=(q_len, BLOCK_DMODEL),
            strides=(stride_om, stride_on),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0),
        )
        tl.store(
            O_block_ptr,
            acc1.to(Out.dtype.element_ty),
            boundary_check=(0,) if BOUNDS_CHECKS_M or USE_SEQ_POS_Q else (),
        )

    _autotuner_config_amd_full = [
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "waves_per_eu": 2, "pre_load_v": False},
            num_stages=1,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "waves_per_eu": 2, "pre_load_v": False},
            num_stages=1,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "waves_per_eu": 2, "pre_load_v": False},
            num_stages=1,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "waves_per_eu": 3, "pre_load_v": True},
            num_stages=1,
            num_warps=4,
        ),  # d64-False
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "waves_per_eu": 3, "pre_load_v": False},
            num_stages=1,
            num_warps=4,
        ),  # d64-True
    ]

    _autotuner_config_amd_dummy = [
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "waves_per_eu": 2, "pre_load_v": False},
            num_stages=1,
            num_warps=8,
        ),
    ]

    _autotuner_config_nvidia_dummy = [
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "pre_load_v": False},
            num_stages=1,
            num_warps=8,
        ),
    ]

    def autotune_kernel(kernel, autotune):

        kernel = triton.heuristics(
            values={
                "BOUNDS_CHECKS_N": lambda args: ((args["Mkv"] % args["BLOCK_N"]) != 0)
                or (args["USE_SEQ_POS_Q"] and not args["IS_KV_PADDED"]),
                "BOUNDS_CHECKS_M": lambda args: (args["N_CTX"] % args["BLOCK_M"]) != 0,
            }
        )(kernel)

        if torch.version.cuda:
            configs = _autotuner_config_nvidia_dummy
        elif autotune:
            configs = _autotuner_config_amd_full
        else:
            configs = _autotuner_config_amd_dummy

        kernel = triton.autotune(
            configs=configs,
            key=["Z", "H", "N_CTX", "IS_CAUSAL", "BLOCK_DMODEL"],
        )(kernel)
        return kernel

    _fwd_kernel_triton_flash_maybe_autotuned = {
        True: autotune_kernel(_fwd_kernel_triton_flash, True),
        False: autotune_kernel(_fwd_kernel_triton_flash, False),
    }
else:
    _fwd_kernel_triton_flash = None
    _fwd_kernel_triton_flash_maybe_autotuned = dict()


def _prepare_inputs(inp: Inputs) -> Inputs:
    attn_bias = inp.attn_bias
    if isinstance(attn_bias, torch.Tensor) and attn_bias.ndim == 3:
        B = inp.query.shape[0]
        h = attn_bias.shape[0] // B
        attn_bias = attn_bias.reshape(B, h, attn_bias.shape[1], attn_bias.shape[2])

    # Make sure that the last dimension is contiguous
    query, key, value = [
        x if x.stride(-1) == 1 else x.contiguous()
        for x in [inp.query, inp.key, inp.value]
    ]
    return replace(inp, attn_bias=attn_bias, query=query, key=key, value=value)


@register_operator
class FwOp(AttentionFwOpBase):
    """Operator that computes memory-efficient attention using \
        `Tri Dao's <https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attn_triton.py>`_ \
        implementation, based on
        `Phil Tillet's code <https://github.com/openai/triton/blob/master/python/tutorials/06-fused-attention.py>`_
    """

    OPERATOR = _fwd_kernel_triton_flash
    SUPPORTED_DEVICES = {"cuda"}
    CUDA_MINIMUM_COMPUTE_CAPABILITY = (8, 0)
    SUPPORTED_DTYPES = {torch.half, torch.bfloat16}
    SUPPORTED_MAX_K = 128
    SUPPORTED_ATTN_BIAS_TYPES: Set[Any] = {
        type(None),
        LowerTriangularMask,
        BlockDiagonalCausalMask,
        BlockDiagonalCausalWithOffsetPaddedKeysMask,
    }
    SUPPORTS_DROPOUT = False
    SUPPORTS_CUSTOM_SCALE = True
    NAME = "tritonflashattF"

    # Off by default to avoid slowing down tests.
    # Needs to be turned on explicitly in benchmarks, in prod, and in a small number of tests
    AUTOTUNE = False

    ERROR_ATOL: Mapping[torch.dtype, float] = {
        torch.half: 2e-2,
        torch.bfloat16: 2e-2,
    }

    ERROR_RTOL: Mapping[torch.dtype, float] = {
        torch.half: 2e-2,
        torch.bfloat16: 2e-2,
    }

    @classmethod
    def shape_not_supported_reasons(
        cls, Mq: int, Mkv: int, K: int, Kv: int
    ) -> List[str]:
        reasons = super().shape_not_supported_reasons(Mq, Mkv, K, Kv)
        if K not in {32, 64, 128}:
            reasons.append(f"Embed dim {K} not supported")
        return reasons

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:
        reasons = super(FwOp, cls).not_supported_reasons(d)
        check_lastdim_alignment_stride1(reasons, "query", d.query, 8)
        check_lastdim_alignment_stride1(reasons, "key", d.key, 8)
        check_lastdim_alignment_stride1(reasons, "value", d.value, 8)

        if isinstance(
            d.attn_bias,
            BlockDiagonalCausalWithOffsetPaddedKeysMask,
        ):
            # Support padded causal block-diagonal mask if the distance between each two consecutive key starts
            # is equal to the padding (key lengths can vary)
            batch_size = len(d.attn_bias.q_seqinfo.seqstart_py) - 1
            B_T = d.key.shape[
                1
            ]  # For these mask types the shapes of Q/K/V are (1, B_T, H, K)
            if B_T % batch_size:
                reasons.append(
                    f"K/V should be padded, but batch size {batch_size} doesn't divide B*T={B_T}"
                )
            else:
                kv_maxlen = d.attn_bias.k_seqinfo.padding
                for i, seqstart in enumerate(d.attn_bias.k_seqinfo.seqstart_py):
                    if seqstart != i * kv_maxlen:
                        reasons.append(
                            "Variable K/V start positions are not supported, they should be determined "
                            f"by kv_maxlen/padding: {d.attn_bias.k_seqinfo.seqstart_py=} {kv_maxlen=} {batch_size=}"
                        )
                        break
        if isinstance(
            d.attn_bias,
            BlockDiagonalCausalMask,
        ):
            # Support padded causal block-diagonal mask if for each batch element number of queries is equal
            # to the number of key/values, i.e. each block is square
            for q_pos, kv_pos in zip(
                d.attn_bias.q_seqinfo.seqstart_py, d.attn_bias.k_seqinfo.seqstart_py
            ):
                if q_pos != kv_pos:
                    reasons.append(
                        f"Position starts of Q and K/V should be the same, but got {q_pos} != {kv_pos}"
                        f"{d.attn_bias.q_seqinfo.seqstart_py=}, {d.attn_bias.k_seqinfo.seqstart_py=}"
                    )

        if d.device.type == "cuda" and torch.version.cuda:
            # Has only been tested on 8.0 / 9.0.
            # Fails on 7.5 with illegal memory access
            if torch.cuda.get_device_capability(d.device) < (8, 0):
                reasons.append(
                    "requires GPU with sm80 minimum compute capacity, e.g., A100/H100/L4"
                )
        return reasons

    @classmethod
    def apply(
        cls, inp: Inputs, needs_gradient: bool
    ) -> Tuple[torch.Tensor, Optional[Context]]:
        inp = _prepare_inputs(inp)
        attn_bias = inp.attn_bias
        seq_len_kv = None
        seqstart_q = None

        q = inp.query
        k = inp.key
        v = inp.value

        if isinstance(
            attn_bias,
            (BlockDiagonalCausalWithOffsetPaddedKeysMask, BlockDiagonalCausalMask),
        ):
            # q ~ [1, B*T, H, K]
            # TODO: do we really need to do this cast? seems fishy but
            # I just copied it from the split-k kernel
            attn_bias.k_seqinfo.to(inp.query.device)
            attn_bias.q_seqinfo.to(inp.query.device)
            seqstart_q = attn_bias.q_seqinfo.seqstart
            B = len(seqstart_q) - 1
            H, Kq = inp.query.shape[-2:]
            H2, Kkv = inp.key.shape[-2:]

            Mq = attn_bias.q_seqinfo.max_seqlen
            if isinstance(attn_bias, BlockDiagonalCausalWithOffsetPaddedKeysMask):
                seq_len_kv = attn_bias.k_seqinfo.seqlen
                # assume kv has been padded
                k = k.reshape(B, -1, H2, Kkv)
                v = v.reshape(B, -1, H2, Kkv)
        else:
            B, Mq, H, _ = q.shape

        # Coded for BHMK format
        q, k, v = (
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )

        out = torch.empty_like(q)

        _, _, Mkv, K = k.shape

        sm_scale = K**-0.5 if inp.scale is None else inp.scale
        L = torch.empty((B * H, Mq), device=q.device, dtype=torch.float32)
        is_causal = inp.attn_bias is not None
        use_seq_len_kv = seq_len_kv is not None
        use_seq_pos_q = seqstart_q is not None
        is_kv_padded = isinstance(
            attn_bias, BlockDiagonalCausalWithOffsetPaddedKeysMask
        )

        grid = lambda META: (triton.cdiv(Mq, META["BLOCK_M"]), B * H, 1)  # noqa: E731
        kernel = _fwd_kernel_triton_flash_maybe_autotuned[cls.AUTOTUNE]
        kernel[grid](
            q,
            k,
            v,
            sm_scale,
            L,
            out,
            seq_len_kv,
            seqstart_q,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            B,
            H,
            Mq,
            Mkv,
            BLOCK_DMODEL=K,
            IS_CAUSAL=is_causal,
            USE_SEQ_LEN_KV=use_seq_len_kv,
            USE_SEQ_POS_Q=use_seq_pos_q,
            IS_KV_PADDED=is_kv_padded,
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
            CAST_BEFORE_MATMUL=False,
        )

        out = out.transpose(1, 2)
        L = L.reshape(B, H, Mq)
        return out, Context(lse=L, out=out)
