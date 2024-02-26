# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple

import numpy as np
import torch


def assert_allclose(
    out: Optional[torch.Tensor],
    ref: Optional[torch.Tensor],
    msg: str = "failed",
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> None:
    assert out is not None, f"{msg}: output Tensor is None"
    assert ref is not None, f"{msg}: reference Tensor is None"
    assert out.shape == ref.shape, f"Shape: {out.shape} (expected: {ref.shape})"
    if out.dtype != ref.dtype:
        assert False, f"out dtype: {out.dtype}, ref dtype: {ref.dtype}"
    if out.numel() == 0:
        return
    flatten_diff = ((out - ref).abs() - atol - ref.abs() * rtol).flatten()
    max_pos = flatten_diff.argmax()
    max_location = np.unravel_index(int(max_pos), out.shape)
    max_diff = flatten_diff[max_pos]
    num_different = flatten_diff.numel() - torch.count_nonzero(flatten_diff <= 0)
    percentage = num_different / flatten_diff.numel()
    del flatten_diff
    assert torch.allclose(out, ref, rtol=rtol, atol=atol), (
        f"{msg}: "
        f"out={out.flatten()[max_pos]} and ref={ref.flatten()[max_pos]} (diff={max_diff} > 0)"
        f" at {max_location} of shape {tuple(out.shape)} / atol={atol}, rtol={rtol}"
        f"/ total failing elements: {num_different} ({percentage*100:.3}%)"
    )


def pack_kv_cache(
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    kv_seqlens: List[int],
    BLOCK_N: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create block tables and pages K/V cache for testing paged attention.
    Args:
        cache_k, cache_v: K/V caches, each of shape [B, MAX_T, H_kv, D].
            Note that these tensors are unexpanded,
            i.e. for multiquery case cache_k.shape[2] = 1
        kv_seqlens: list of K/V sequence lengths
        BLOCK_N: number of tokens per per paged attention block
        B: batch size
    Returns:
        block_tables: [B, MAX_BLOCKS]
        packed_cache_k: [1, total_len_rounded, H_kv, D]
        packed_cache_v: [1, total_len_rounded, H_kv, D]
    where total_len_rounded is a sum of K/V seqlens, each rounded up
    to a multiple of BLOCK_N.
    """

    kv_seqlens_rounded = [(x + BLOCK_N - 1) // BLOCK_N * BLOCK_N for x in kv_seqlens]

    total_len_rounded = sum(kv_seqlens_rounded)

    B, MAX_T, H, D = cache_k.shape

    packed_cache_k = torch.empty(
        total_len_rounded, H, D, device=cache_k.device, dtype=cache_k.dtype
    )
    packed_cache_v = torch.empty(
        total_len_rounded, H, D, device=cache_k.device, dtype=cache_k.dtype
    )
    seqstart = 0
    for b in range(B):
        packed_cache_k[seqstart : seqstart + kv_seqlens[b]] = cache_k[
            b, : kv_seqlens[b]
        ].clone()
        packed_cache_v[seqstart : seqstart + kv_seqlens[b]] = cache_v[
            b, : kv_seqlens[b]
        ].clone()
        seqstart += kv_seqlens_rounded[b]

    num_blocks_per_row = (MAX_T + BLOCK_N - 1) // BLOCK_N
    block_tables = (
        torch.arange(num_blocks_per_row, device="cuda", dtype=torch.int32)
        .unsqueeze(0)
        .expand(B, num_blocks_per_row)
    )
    seqstarts = (
        (
            torch.tensor(kv_seqlens_rounded).cumsum(dim=0)
            - torch.tensor(kv_seqlens_rounded)
        )
        .to(device="cuda")
        .unsqueeze(1)
    ) // BLOCK_N
    block_tables = (block_tables + seqstarts).contiguous().to(dtype=torch.int32)
    return (
        block_tables,
        packed_cache_k.unsqueeze(0),
        packed_cache_v.unsqueeze(0),
    )
