# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import triton
import triton.language as tl


# fmt: off
@triton.jit
def k_ragged_attention_fw(
    OUT_PTR,
    Q_PTR, K_PTR,
    N_QUERY_CTX,
    N_KEY_CTX,
    D_MODEL,
    stride_o_b, stride_o_m,
    stride_q_b, stride_q_m,
    stride_k_b, stride_k_n,
    **META,
):
    # fmt: on

    # extract metaparameters
    BLOCK_M = META["BLOCK_M"]
    BLOCK_N = META["BLOCK_N"]
    BLOCK_L = META["BLOCK_L"]
    scale = META["SCALE"]

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse
    # See above `L2 Cache Optimizations` section for details
    pid_b = tl.program_id(axis=0)
    pid = tl.program_id(axis=1)
    GROUP_SIZE_M = 8

    num_pid_m = tl.cdiv(N_QUERY_CTX, BLOCK_M)
    num_pid_n = tl.cdiv(N_KEY_CTX, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Compute QKt, block level matrix multiplication.
    # We fetch a block memory from both inputs, matmul and accumulate, then repeat
    # This step walks over the model dimension
    qkt = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rl = tl.arange(0, BLOCK_L)

    l_offset = 0
    q_bm_ptrs = Q_PTR + pid_b * stride_q_b + rm[:, None] * stride_q_m
    k_bn_ptrs = K_PTR + pid_b * stride_k_b + rn[None, :] * stride_k_n

    for _ in range(0, D_MODEL, BLOCK_L):
        rl_i = rl + l_offset
        q_ptrs = q_bm_ptrs + rl_i[None, :]    # (BLOCK_M, BLOCK_L)
        k_ptrs = k_bn_ptrs + rl_i[:, None]    # (BLOCK_L, BLOCK_N)

        q_mask = ((rm[:, None] < N_QUERY_CTX) & (rl_i[None, :] < D_MODEL))
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)  # (BLOCK_M, BLOCK_L)

        k_mask = ((rl_i[:, None] < D_MODEL) & (rn[None, :] < N_KEY_CTX))
        kt = tl.load(k_ptrs, mask=k_mask, other=0.0)   # (BLOCK_L, BLOCK_N)

        q *= scale.to(q.dtype)  # q /= sqrt(dim)
        qkt += tl.dot(q, kt).to(tl.float32)              # (BLOCK_M, BLOCK_N)

        # Update the pointers and counter
        l_offset += BLOCK_L

    # Final output tile is [BLOCK_M, BLOCK_N]
    m_offset = rm[:, None] * stride_o_m
    n_offset = rn[None, :]
    scores_ptrs = OUT_PTR + pid_b * stride_o_b + m_offset + n_offset

    tl.store(scores_ptrs, qkt, mask=(rm[:, None] < N_QUERY_CTX) & (rn[None, :] < N_KEY_CTX))


def ragged_qk_dotprod(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:

    assert q.ndim == 3
    assert q.shape[-1] == k.shape[-1] and q.shape[0] == k.shape[0]
    assert q.is_contiguous() and k.is_contiguous()
    B, M, L = q.shape
    B, N, L = k.shape

    # TODO: autotune these
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_L = 32

    # We put parameters into the grid() function to allow for auto-tuning
    def grid(META):
        return (
            B,
            triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        )

    out_scores = torch.empty((B, M, N), dtype=q.dtype, device=q.device)

    # Use a dedicated kernel to process the attention by blocks
    # fmt: off
    k_ragged_attention_fw[grid](
        out_scores,         # outputs
        q, k,               # inputs
        M, N, L,            # dimensions
        out_scores.stride(0), out_scores.stride(1),
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_L=BLOCK_L,
        SCALE=1. / math.sqrt(L),
    )
    # fmt: onx

    return out_scores
