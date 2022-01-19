# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import triton
import triton.language as tl

_DEBUG = 0  # 1 to see the kernel PTX assembly


# fmt: off
@triton.jit
def k_me_attention_fw(
    OUT,    # TODO: rename to OUT_PTR etc
    Q, K,            # in ptrs
    M,  # N_QUERY_CTX
    N,  # N_KEY_CTX
    L,  # D_MODEL
    stride_m,
    stride_n,
    **META,
):
    # fmt: on

    # extract metaparameters
    BLOCK_M = META["BLOCK_M"]
    BLOCK_N, BLOCK_L = META["BLOCK_N"], META["BLOCK_L"]

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

    # Final output tile is [BLOCK_M, BLOCK_N]
    m_offset = rm[:, None] * stride_m
    n_offset = rn[None, :] * stride_n
    scores_ptrs = OUT + m_offset + n_offset

    mask = (rm[:, None] < M)
    tl.store(scores_ptrs, qkt, mask=mask)


def mem_efficient_fw(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:

    assert q.ndim == 3
    assert q.shape[-1] == k.shape[-1]
    assert q.is_contiguous() and k.is_contiguous()
    B, M, L = q.shape
    B, N, L = k.shape

    # TODO: autotune these
    BLOCK_M = 8
    BLOCK_N = 8
    # BLOCK_N = min(triton.next_power_of_2(N), 1024)  # increase the ceiling to save more memory
    BLOCK_L = 8

    tiles_n = triton.cdiv(N, BLOCK_N)
    tiles_m = triton.cdiv(M, BLOCK_M)
    # print(f"{tiles_m=}, {tiles_n=}") # 3, 1

    # We put parameters into the grid() function to allow for auto-tuning
    def grid(META):
        # tiles_m = triton.cdiv(M, META["BLOCK_M"])
        return (
            tiles_m,
            tiles_n
        )

    # out_scores = torch.ones((B, M, N), dtype=q.dtype, device=q.device) * 55
    out_scores = torch.empty((B, M, N), dtype=q.dtype, device=q.device)

    # TODO: improve on the batch dimension handling ?

    for i_b in range(B):

        # Use a dedicated kernel to process the attention by blocks
        # fmt: off
        bin = k_me_attention_fw[grid](
            out_scores[i_b],       # outputs
            q[i_b],
            k[i_b],     # inputs
            M, N, L,                            # dimensions
            out_scores.stride(1),
            out_scores.stride(2),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_L=BLOCK_L,
            SCALE=1. / math.sqrt(L),
            num_warps=1
        )
        # fmt: onx

        if _DEBUG:
            print(bin.asm['ptx'])

    return out_scores
