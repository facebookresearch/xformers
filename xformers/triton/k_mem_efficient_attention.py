# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional

import torch
import triton
import triton.language as tl

_DEBUG = 0  # 1 to see the kernel PTX assembly
_FUSED_NORMALIZATION = True  # FIXME: rounding error, but should work eventually


# fmt: off
@triton.jit
def k_me_attention_fw(
    OUT, MAXES, WEIGHTS,    # out ptr
    Q, K, V,                # in ptrs
    M, N, L,                # dims
    stride_out_tile, stride_out_m,
    stride_maxes, stride_weights,
    **META,
):
    # fmt: on

    # extract metaparameters
    BLOCK_M = META["BLOCK_M"]
    BLOCK_N, BLOCK_L = META["BLOCK_N"], META["BLOCK_L"]
    FUSED_NORMALIZATION = META["FUSED_NORMALIZATION"]

    scale = META["SCALE"]

    # *within groups*, programs are ordered in a column-major order
    # row-id /col-id of the program in the *launch grid*
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Compute QKt
    # block level matrix multiplication.
    # We fetch a block memory block from both inputs, matmul and accumulate, then repeat
    qkt = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    i = 0
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rl = tl.arange(0, BLOCK_L)

    for _ in range(L, 0, -BLOCK_L):
        rl_i = rl + i * BLOCK_L  # keep track of the masking
        q_ptrs = Q + rm[:, None] * L + rl_i[None, :]    # (BLOCK_M, BLOCK_L)
        k_ptrs = K + rn[None, :] * L + rl_i[:, None]    # (BLOCK_L, BLOCK_N)

        q = tl.load(q_ptrs, mask=((rm[:, None] < M) & (rl_i[None, :] < L)), other=0.0)    # (BLOCK_M, BLOCK_L)
        k = tl.load(k_ptrs, mask=((rl_i[:, None] < L) & (rn[None, :] < N)), other=0.0)    # (BLOCK_L, BLOCK_N)

        q *= scale  # q /= sqrt(dim)
        qkt += tl.dot(q, k).to(tl.float32)              # (BLOCK_M, BLOCK_N)

        # Update the pointers and counter
        i += 1

    # Pick the local max per row, safeguard the incoming exponential
    max_qkt = tl.max(qkt, axis=1)                       # (BLOCK_M)
    max_ptrs = MAXES + pid_n * stride_maxes + rm        # (BLOCK_M)

    # Save so that an eventual mismatch can be fixed post-hoc
    if FUSED_NORMALIZATION is False:
        tl.store(max_ptrs, max_qkt, mask=(rm < M))

    # Exponentiate the neutralized results
    exp_qkt = tl.exp(qkt - max_qkt[:, None])            # (BLOCK_M, BLOCK_N)

    # Softmax normalization constant
    weights = tl.sum(exp_qkt, axis=1)                   # (BLOCK_M)

    if FUSED_NORMALIZATION:
        exp_qkt = exp_qkt / weights[:, None]
    else:
        # Save, global max will be fixed post-hoc
        weights_ptrs = WEIGHTS + pid_n * stride_weights + rm
        tl.store(weights_ptrs, weights, mask=(rm < M))

    # Now pre-compute exp_qkt against V.
    # We proceed per chunk over L, and save as we go
    i = 0
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rl = tl.arange(0, BLOCK_L)

    v_ptrs = V + rn[:, None] * L + rl[None, :]  # (BLOCK_N, BLOCK_L)
    out_ptrs = (
        OUT + pid_n * stride_out_tile + rm[:, None] * stride_out_m + rl[None, :]
    )  # (BLOCK_M, BLOCK_L)

    for _ in range(L, 0, -BLOCK_L):
        rl_i = rl + i * BLOCK_L  # Useful to keep track of the masking

        v = tl.load(v_ptrs, mask=((rn[:, None] < N) & (rl_i[None, :] < L)), other=0.0)
        qkv = tl.dot(exp_qkt, v).to(tl.float32)                # (BLOCK_M, BLOCK_L)

        tl.store(out_ptrs, qkv, mask=(rm[:, None] < M) & (rl_i[None, :] < L))

        i += 1
        v_ptrs += BLOCK_L
        out_ptrs += BLOCK_L


def mem_efficient_fw(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:

    assert q.shape[-1] == k.shape[-1]
    assert v.shape[-1] == k.shape[-1]
    assert k.shape[-2] == v.shape[-2]
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()

    q_shape = q.shape

    if q.ndim == 2:
        # no batch dimension
        q_, k_, v_ = map(lambda x: x.unsqueeze(0), [q, k, v])
    else:
        q_, k_, v_ = q, k, v

    B, M, L = q_.shape
    B, N, L = k_.shape

    BLOCK_M = 4
    BLOCK_N = min(triton.next_power_of_2(N), 1024)  # increase the ceiling to save more memory
    BLOCK_L = 8

    tiles_n = triton.cdiv(N, BLOCK_N)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_M"]),
            tiles_n
        )

    out_n = torch.empty((tiles_n, M, L), dtype=q.dtype, device=q.device)

    if not _FUSED_NORMALIZATION:
        maxes_n = torch.empty((tiles_n, M), dtype=q.dtype, device=q.device)
        weights_n = torch.empty((tiles_n, M), dtype=q.dtype, device=q.device)
    else:
        assert BLOCK_N >= N
        maxes_n = out_n     # placeholder, will not be used
        weights_n = out_n   # placeholder, will not be used

    # FIXME: handle bias
    # FIXME: improve on the batch dimension handling ?
    qkvs = []
    for i_b in range(B):

        # Use a dedicated kernel to process the attention by blocks
        # fmt: off
        bin = k_me_attention_fw[grid](
            out_n, maxes_n, weights_n,          # outputs
            q_[i_b], k_[i_b], v_[i_b],          # inputs
            M, N, L,                            # dimensions
            out_n.stride(0), out_n.stride(1), maxes_n.stride(0), weights_n.stride(0),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_L=BLOCK_L,
            BIAS=False,
            SCALE=1. / math.sqrt(L),
            FUSED_NORMALIZATION=_FUSED_NORMALIZATION,
            num_warps=1
        )
        # fmt: onx

        if _DEBUG:
            print(bin.asm['ptx'])

        # Epilogue
        if tiles_n > 1:
            # There were tiles over the N dimension,
            # so the weights were not correct in real time.

            # Let's fix that:
            # - collect the real overall max per line
            global_max, _ = maxes_n.max(dim=0)

            # - compute the mistake that was done in real time
            mismatch = torch.exp(maxes_n - global_max[None, :])

            # - update the computations to take the consolidated max/weights
            out_n *= mismatch.unsqueeze(-1)
            weights_n *= mismatch

            out = torch.sum(out_n, dim=0)
            weights = torch.sum(weights_n, dim=0)

            qkv = out / weights.unsqueeze(-1)

        else:
            # with fused normalization this should just work
            if _FUSED_NORMALIZATION:
                qkv = out_n.squeeze()
            else:
                qkv = out_n / weights_n.unsqueeze(-1)

        qkvs.append(qkv)

    return torch.cat(qkvs, dim=0).reshape(q_shape)
