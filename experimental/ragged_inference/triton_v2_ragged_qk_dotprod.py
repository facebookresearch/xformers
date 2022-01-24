# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import List, Optional

import torch
import triton
import triton.language as tl
from ragged_inference.garbage_pad_ragged_acts import RaggedActivations
from triton.ops.matmul_perf_model import estimate_matmul_time, prune_num_stages

# This implements a ragged attention (batched attention mechanism natively handling different sequence sizes)
# Author: Tom B Brown


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


# TODO: tune these
BLOCK_Q = 16
BLOCK_K = 128
BLOCK_D = 32


def get_fast_dev_configs():
    return [
        triton.Config(
            {"BLOCK_Q": BLOCK_Q, "BLOCK_K": BLOCK_K, "BLOCK_D": BLOCK_D},
            num_stages=5,
            num_warps=2,
        )
    ]


@triton.autotune(
    # configs=get_all_configs(),
    configs=get_fast_dev_configs(),
    key=["max_n_ctx_q_across_seqs", "max_n_ctx_k_across_seqs", "d_head"],
    prune_configs_by={
        "prune_num_stages_by": prune_num_stages,
        "perf_model": estimate_matmul_time,
        "top_k": 10,
    },
)
@triton.jit
def _qk_dotprod_kernel(
    # Pointers to our tensors
    q_ptr,
    k_ptr,
    scores_ptr,  # Rectangular output tensor
    # Pointers to lookup tables (sometimes referred to as a "lut")
    pid_to_in_q_token_offset_ptr,
    pid_to_in_k_token_offset_ptr,
    pid_to_out_q_block_ptr,
    pid_to_out_k_block_ptr,
    pid_to_out_seq_idx_ptr,
    # Integers
    max_n_ctx_q_across_seqs,
    max_n_ctx_k_across_seqs,
    d_head,
    stride_ctx_q,
    stride_ctx_k,
    stride_out_q,
    stride_out_k,
    stride_out_seq,
    total_ctx_q_across_all_seqs,
    total_ctx_k_across_all_seqs,
    # These get populated from the triton.Config
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Adapted from https://github.com/openai/triton/blob/v2.0/python/triton/ops/matmul.py
    """

    # matrix multiplication
    pid = tl.program_id(0)

    out_q_block = tl.load(pid_to_out_q_block_ptr + pid)
    out_k_block = tl.load(pid_to_out_k_block_ptr + pid)
    out_seq_idx = tl.load(pid_to_out_seq_idx_ptr + pid)
    in_q_token_offset = tl.load(pid_to_in_q_token_offset_ptr + pid)
    in_k_token_offset = tl.load(pid_to_in_k_token_offset_ptr + pid)

    # Define indices ranges, we follow the triton convention of prefixing
    # with "r" to denote a range like "rq" is the range for queries below
    rq = in_q_token_offset + tl.arange(0, BLOCK_Q)
    rk = in_k_token_offset + tl.arange(0, BLOCK_K)

    # Prevent out of bounds reads. It's ok to read garbage data for queries and keys that aren't
    # actually used. Their values don't effect any of the outputs.
    q_ctx_in_bounds = rq < total_ctx_q_across_all_seqs
    k_ctx_in_bounds = rk < total_ctx_k_across_all_seqs

    # We will accumulate all the d_head items into acc
    acc_tile = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)

    rd = tl.arange(0, BLOCK_D)  # rd indexes into the d_head dimension

    # We use broadcasting to convert our 1D ranges into 2D tiles
    q_ptr_tile = q_ptr + (rq[:, None] * stride_ctx_q + rd[None, :])
    k_ptr_tile = k_ptr + (rd[:, None] + rk[None, :] * stride_ctx_k)

    # After we have profiling, see if we can rewrite this to be more readable by
    # just updating rd += BLOCK_D
    for d_max_offset in range(d_head, 0, -BLOCK_D):
        q_tile = tl.load(
            q_ptr_tile,
            mask=(rd[None, :] < d_max_offset) & q_ctx_in_bounds[:, None],
            other=0.0,
        )
        k_tile = tl.load(
            k_ptr_tile,
            mask=(rd[:, None] < d_max_offset) & k_ctx_in_bounds[None, :],
            other=0.0,
        )

        # In einsum notation, the tl.dot does: qd,dk->qk
        # This should use tensorcores, so the inputs might be fp16, but the outputs
        # and all the internal accumulators are fp32
        acc_tile += tl.dot(q_tile, k_tile)

        q_ptr_tile += BLOCK_D
        k_ptr_tile += BLOCK_D

    # Figure out the output blocks
    rq_out = out_q_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
    rk_out = out_k_block * BLOCK_K + tl.arange(0, BLOCK_K)

    scores_offset_tile = (
        rq_out[:, None] * stride_out_q
        + rk_out[None, :] * stride_out_k
        + out_seq_idx * stride_out_seq
    )
    scores_ptr_tile = scores_ptr + scores_offset_tile

    mask = (rq_out < max_n_ctx_q_across_seqs)[:, None] & (
        rk_out < max_n_ctx_k_across_seqs
    )[None, :]

    # Cast back to lower precision immediately before storing
    acc_tile = acc_tile.to(scores_ptr.dtype.element_ty)
    tl.store(scores_ptr_tile, acc_tile, mask=mask)


@dataclass
class RaggedQkPidLookupTable:
    # TODO: link to a drawing of what these tensors are
    # All cuda tensors
    pid_to_in_q_token_offset: torch.Tensor
    pid_to_in_k_token_offset: torch.Tensor
    pid_to_out_q_block: torch.Tensor
    pid_to_out_k_block: torch.Tensor
    pid_to_out_seq_idx: torch.Tensor
    n_pids_total: int

    @staticmethod
    def from_single_seq(n_ctx_q: int, n_ctx_k: int) -> "RaggedQkPidLookupTable":
        grid_q = triton.cdiv(n_ctx_q, BLOCK_Q)
        grid_k = triton.cdiv(n_ctx_k, BLOCK_K)
        n_pids_total = grid_q * grid_k

        pid_to_in_q_token_offset = torch.zeros(
            n_pids_total, dtype=torch.int32, device="cuda"
        )
        pid_to_in_k_token_offset = torch.zeros(
            n_pids_total, dtype=torch.int32, device="cuda"
        )
        pid_to_out_q_block = torch.zeros(n_pids_total, dtype=torch.int32, device="cuda")
        pid_to_out_k_block = torch.zeros(n_pids_total, dtype=torch.int32, device="cuda")
        pid_to_out_seq_idx = torch.zeros(n_pids_total, dtype=torch.int32, device="cuda")

        for pid in range(n_pids_total):
            q_block_idx = pid // grid_k
            k_block_idx = pid % grid_k

            in_q_token_offset = q_block_idx * BLOCK_Q
            in_k_token_offset = k_block_idx * BLOCK_K

            pid_to_out_q_block[pid] = q_block_idx
            pid_to_out_k_block[pid] = k_block_idx
            pid_to_in_q_token_offset[pid] = in_q_token_offset
            pid_to_in_k_token_offset[pid] = in_k_token_offset

        return RaggedQkPidLookupTable(
            pid_to_in_q_token_offset=pid_to_in_q_token_offset,
            pid_to_in_k_token_offset=pid_to_in_k_token_offset,
            pid_to_out_q_block=pid_to_out_q_block,
            pid_to_out_k_block=pid_to_out_k_block,
            pid_to_out_seq_idx=pid_to_out_seq_idx,
            n_pids_total=n_pids_total,
        )

    @staticmethod
    def from_query_and_key_tokens_per_seq(
        n_ctx_q_per_seq: List[int],
        n_ctx_k_per_seq: List[int],
        block_q_override: Optional[int] = None,
        block_k_override: Optional[int] = None,
    ) -> "RaggedQkPidLookupTable":
        block_q = block_q_override if block_q_override else BLOCK_Q
        block_k = block_k_override if block_k_override else BLOCK_K

        pid_to_in_q_token_offset = []
        pid_to_in_k_token_offset = []
        pid_to_out_q_block = []
        pid_to_out_k_block = []
        pid_to_out_seq_idx = []

        n_in_q_token_so_far = 0
        n_in_k_token_so_far = 0

        for seq_idx, (n_ctx_q, n_ctx_k) in enumerate(
            zip(n_ctx_q_per_seq, n_ctx_k_per_seq)
        ):
            # Everything below is per sequence
            n_q_ctx_blocks = triton.cdiv(n_ctx_q, block_q)
            n_k_ctx_blocks = triton.cdiv(n_ctx_k, block_k)
            n_pids_in_seq = n_q_ctx_blocks * n_k_ctx_blocks

            for pid in range(n_pids_in_seq):
                q_block_idx = pid // n_k_ctx_blocks
                k_block_idx = pid % n_k_ctx_blocks

                in_q_token_offset = q_block_idx * block_q
                in_k_token_offset = k_block_idx * block_k

                pid_to_out_q_block.append(q_block_idx)
                pid_to_out_k_block.append(k_block_idx)
                pid_to_in_q_token_offset.append(in_q_token_offset + n_in_q_token_so_far)
                pid_to_in_k_token_offset.append(in_k_token_offset + n_in_k_token_so_far)
                pid_to_out_seq_idx.append(seq_idx)

            n_in_q_token_so_far += n_ctx_q
            n_in_k_token_so_far += n_ctx_k

        args = {"dtype": torch.int32, "device": "cuda"}
        return RaggedQkPidLookupTable(
            pid_to_in_q_token_offset=torch.tensor(pid_to_in_q_token_offset, **args),  # type: ignore
            pid_to_in_k_token_offset=torch.tensor(pid_to_in_k_token_offset, **args),  # type: ignore
            pid_to_out_q_block=torch.tensor(pid_to_out_q_block, **args),  # type: ignore
            pid_to_out_k_block=torch.tensor(pid_to_out_k_block, **args),  # type: ignore
            pid_to_out_seq_idx=torch.tensor(pid_to_out_seq_idx, **args),  # type: ignore
            n_pids_total=len(pid_to_in_q_token_offset),
        )


def ragged_single_seq_qk_dotprod(
    query: torch.Tensor, key: torch.Tensor, lut: RaggedQkPidLookupTable
) -> torch.Tensor:
    assert query.ndim == 2 and key.ndim == 2
    device = query.device

    # handle non-contiguous inputs if necessary
    if query.stride(0) > 1 and query.stride(1) > 1:
        query = query.contiguous()
    if key.stride(0) > 1 and key.stride(1) > 1:
        key = key.contiguous()

    # check constraints
    n_ctx_q, d_head = query.shape
    n_ctx_k, d_head_k = key.shape
    assert d_head == d_head_k, f"{query.shape=} {key.shape=}"

    # allocates output
    scores_out = torch.empty((1, n_ctx_q, n_ctx_k), device=device, dtype=query.dtype)

    # Stride along the d_head dimension must be 1
    assert query.stride(1) == 1, f"{query.stride(1)}"
    assert key.stride(1) == 1, f"{key.stride(1)}"

    # pid_to_seq_idx = [0, 0, 1, 2, 2]
    grid = (lut.n_pids_total,)
    _qk_dotprod_kernel[grid](
        q_ptr=query,
        k_ptr=key,
        scores_ptr=scores_out,
        # Lookup tables (sometimes referred to as a "lut")
        pid_to_in_q_token_offset_ptr=lut.pid_to_in_q_token_offset,
        pid_to_in_k_token_offset_ptr=lut.pid_to_in_k_token_offset,
        pid_to_out_q_block_ptr=lut.pid_to_out_q_block,
        pid_to_out_k_block_ptr=lut.pid_to_out_k_block,
        pid_to_out_seq_idx_ptr=lut.pid_to_out_seq_idx,
        # Integers
        max_n_ctx_q_across_seqs=n_ctx_q,
        max_n_ctx_k_across_seqs=n_ctx_k,
        d_head=d_head,
        stride_ctx_q=query.stride(0),
        stride_ctx_k=key.stride(0),
        stride_out_seq=scores_out.stride(0),
        stride_out_q=scores_out.stride(1),
        stride_out_k=scores_out.stride(2),
        total_ctx_q_across_all_seqs=n_ctx_q,
        total_ctx_k_across_all_seqs=n_ctx_k,
    )
    return scores_out.reshape((n_ctx_q, n_ctx_k))


def ragged_qk_dotprod(
    query: RaggedActivations, key: RaggedActivations, lut: RaggedQkPidLookupTable
) -> torch.Tensor:
    device = query.device

    assert query.raw_tensor.is_contiguous()
    assert key.raw_tensor.is_contiguous()

    # check constraints
    total_ctx_q_across_all_seqs, d_head = query.raw_tensor.shape
    total_ctx_k_across_all_seqs, d_head_k = key.raw_tensor.shape
    assert d_head == d_head_k, f"{query.raw_tensor.shape=} {key.raw_tensor.shape=}"

    # allocates output
    # max_n_ctx_q_across_seqs = query.max_n_ctx_per_seq

    assert query.n_seqs == key.n_seqs
    # TODO: flag use zeros for garbage
    scores_out = torch.ones(
        (query.n_seqs, query.max_n_ctx_per_seq, key.max_n_ctx_per_seq),
        device=device,
        dtype=query.dtype,
    )

    # Stride along the d_head dimension must be 1
    assert query.raw_tensor.stride(1) == 1, f"{query.raw_tensor.stride(1)}"
    assert key.raw_tensor.stride(1) == 1, f"{key.raw_tensor.stride(1)}"

    # pid_to_seq_idx = [0, 0, 1, 2, 2]
    grid = (lut.n_pids_total,)
    _qk_dotprod_kernel[grid](
        q_ptr=query.raw_tensor,
        k_ptr=key.raw_tensor,
        scores_ptr=scores_out,
        # Lookup tables (sometimes referred to as a "lut")
        pid_to_in_q_token_offset_ptr=lut.pid_to_in_q_token_offset,
        pid_to_in_k_token_offset_ptr=lut.pid_to_in_k_token_offset,
        pid_to_out_q_block_ptr=lut.pid_to_out_q_block,
        pid_to_out_k_block_ptr=lut.pid_to_out_k_block,
        pid_to_out_seq_idx_ptr=lut.pid_to_out_seq_idx,
        # Integers
        max_n_ctx_q_across_seqs=query.max_n_ctx_per_seq,
        max_n_ctx_k_across_seqs=key.max_n_ctx_per_seq,
        d_head=d_head,
        stride_ctx_q=query.raw_tensor.stride(0),
        stride_ctx_k=key.raw_tensor.stride(0),
        stride_out_seq=scores_out.stride(0),
        stride_out_q=scores_out.stride(1),
        stride_out_k=scores_out.stride(2),
        total_ctx_q_across_all_seqs=total_ctx_q_across_all_seqs,
        total_ctx_k_across_all_seqs=total_ctx_k_across_all_seqs,
    )
    return scores_out
