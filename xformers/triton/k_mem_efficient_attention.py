# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional

import torch
import triton
import triton.language as tl


# fmt: off
@triton.jit
def k_me_attention_fw(
    OUT, MAXES, WEIGHTS,    # out ptr
    Q, K, V,                # in ptrs
    M, N, L,                # dims
    stride_out, stride_maxes, stride_weights,
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

    # now compute the ranges that each program will go through
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # initialize and iteratively update accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # block level matrix multiplication.
    # We fetch a block memory block from both inputs, matmul and accumulate, then repeat
    i = 0
    rl = tl.arange(0, BLOCK_L)
    for _ in range(L, 0, -BLOCK_L):
        rl_i = rl + i * BLOCK_L
        q_ptrs = Q + rm[:, None] * L + rl_i[None, :]    # (BLOCK_M, BLOCK_L)
        k_ptrs = K + rn[None, :] * L + rl_i[:, None]    # (BLOCK_L, BLOCK_N)

        q = tl.load(q_ptrs, mask=((rm[:, None] < M) & (rl_i[None, :] < L)), other=0.0)    # (BLOCK_M, BLOCK_L)
        q *= scale  # q /= sqrt(dim)

        k = tl.load(k_ptrs, mask=((rl_i[:, None] < L) & (rn[None, :] < N)), other=0.0)    # (BLOCK_L, BLOCK_N)

        acc += tl.dot(q, k).to(tl.float32)              # (BLOCK_M, BLOCK_N)
        i += 1

    # pick the local max, safeguard the incoming exponential
    # save so that an eventual mismatch can be fixed
    max_acc = tl.max(acc, axis=1)                       # (BLOCK_M)
    max_ptrs = MAXES + pid_n * stride_maxes + rm        # (BLOCK_M)
    tl.store(max_ptrs, max_acc, mask=(rm < M))

    # exponentiate the neutralized results
    exp_acc = tl.exp(acc - max_acc[:, None])            # (BLOCK_M, BLOCK_N)

    # Now pre-compute exp_acc against V.
    # We proceed per chunk over L, and save as we go
    i = 0
    rl = tl.arange(0, BLOCK_L)
    for _ in range(L, 0, -BLOCK_L):
        rl_i = rl + i * BLOCK_L

        v_ptrs = V + rn[:, None] * L + rl_i[None, :]    # (BLOCK_N, BLOCK_L)
        v = tl.load(v_ptrs, mask=((rn[:, None] < N) & (rl_i[None, :] < L)), other=0.0)

        qkv = tl.dot(exp_acc, v).to(tl.float32)         # (BLOCK_M, BLOCK_L)

        out_ptrs = OUT + pid_n * stride_out + rm[:, None] * L + rl_i[None, :]  # (BLOCK_M, BLOCK_L)
        tl.store(out_ptrs, qkv, mask=(rm[:, None] < M) & (rl_i[None, :] < L))
        i += 1

    # save so that an eventual mismatch can be fixed
    weights = tl.sum(exp_acc, axis=1)                   # (BLOCK_M)
    weights_ptrs = WEIGHTS + pid_n * stride_weights + rm
    tl.store(weights_ptrs, weights, mask=(rm < M))


def mem_efficient_fw(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:

    assert q.shape[-1] == k.shape[-1]
    assert v.shape[-1] == k.shape[-1]
    assert k.shape[-2] == v.shape[-2]
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()

    q_shape = q.shape

    if q.ndim == 2:
        # no batch dimension
        q.unsqueeze_(0)
        k.unsqueeze_(0)
        v.unsqueeze_(0)

    B, M, L = q.shape
    B, N, L = k.shape

    BLOCK_M = 8
    BLOCK_N = min(triton.next_power_of_2(N), 1024)  # increase to save more memory
    BLOCK_L = 8

    tiles_n = triton.cdiv(N, BLOCK_N)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_M"]),
            tiles_n
        )

    maxes_n = torch.empty((tiles_n, M), dtype=q.dtype, device=q.device)
    weights_n = torch.empty((tiles_n, M), dtype=q.dtype, device=q.device)

    # FIXME: handle bias
    # FIXME: improve on the batch dimension handling ?
    qkvs = []
    for i_b in range(B):
        out = torch.empty((tiles_n, M, L), dtype=q.dtype, device=q.device)

        # Use a dedicated kernel to process the attention by blocks
        # fmt: off
        k_me_attention_fw[grid](
            out, maxes_n, weights_n,        # outputs
            q[i_b], k[i_b], v[i_b],         # inputs
            M, N, L,                        # dimensions
            out.stride(0), maxes_n.stride(0), weights_n.stride(0),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_L=BLOCK_L,
            BIAS=False,
            SCALE=1. / math.sqrt(L)
        )
        # fmt: onx

        # Epilogue
        if tiles_n > 1:
            # There were tiles over the N dimension,
            # so the weights were not correct in real time.

            # Let's fix that:
            # - collect the real overall max per line
            per_line_max, _ = maxes_n.max(dim=0)

            # - compute the mistake that was done in real time
            mismatch = torch.exp(maxes_n - per_line_max[None, :])

            # - update the computations to take the consolidated max/weights
            out *= mismatch.unsqueeze(-1)
            weights_n *= mismatch

            out = torch.sum(out, dim=0)
            weights = torch.sum(weights_n, dim=0)
        else:
            weights = weights_n

        # TODO: do this in the kernel if it owns the whole line
        qkv = out / weights.unsqueeze(-1)
        qkvs.append(qkv)

    return torch.cat(qkvs, dim=0).reshape(q_shape)
