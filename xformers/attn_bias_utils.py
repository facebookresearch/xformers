# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
from typing import List, Optional, Sequence, Tuple, Type

import torch

from xformers.ops import fmha
from xformers.ops.fmha.common import AttentionOpBase


def _create_aligned_bias(*shape: int, **kwargs) -> torch.Tensor:
    align_to = 8
    return (
        torch.randn(
            (
                *shape[:-1],
                align_to * ((shape[-1] + align_to - 1) // align_to),
            ),
            **kwargs,
        )
        * 3
    ).narrow(-1, 0, shape[-1])


def create_attn_bias(
    bias_type,
    batch_size: int,
    num_heads: int,
    num_heads_groups: int,
    q_len: int,
    kv_len: int,
    device,
    dtype,
    requires_grad: bool,
    fmt: str,
    op: Type[AttentionOpBase],
):
    if bias_type is None or isinstance(None, bias_type):
        return None
    r = random.Random("-".join(map(str, [batch_size, q_len, kv_len, dtype, fmt])))
    window_size = {0: 3, 1: 128, 2: 300}[r.randint(0, 2)]
    if bias_type is torch.Tensor:
        if fmt == "BMK":
            batch_size *= num_heads
            num_heads = 1
        # `small_k` only supports an expanded 1d bias
        if op in [fmha.small_k.FwOp, fmha.small_k.BwOp]:
            attn_bias = (
                torch.randn(
                    (batch_size, num_heads, 1, kv_len), device=device, dtype=dtype
                )
                * 3
            )
            attn_bias = attn_bias.expand(batch_size, num_heads, q_len, kv_len)
        else:
            attn_bias = _create_aligned_bias(
                batch_size,
                num_heads_groups,
                num_heads,
                q_len,
                kv_len,
                device=device,
                dtype=dtype,
            )

            # make sure it also works if the first columns/rows are partially masked out
            attn_bias[0, 0, 0, : q_len - 1, : kv_len - 1] = -math.inf
            if fmt in ["BMK", "BMHK"]:
                attn_bias = attn_bias[:, 0]

        if requires_grad:
            attn_bias.requires_grad_(True)
        if fmt == "BMK":
            attn_bias = attn_bias[:, 0]
        return attn_bias
    if bias_type is fmha.attn_bias.LowerTriangularMask:
        return bias_type()
    if bias_type is fmha.attn_bias.LowerTriangularFromBottomRightMask:
        return bias_type()
    if bias_type is fmha.attn_bias.LowerTriangularFromBottomRightLocalAttentionMask:
        return bias_type(window_size)
    if bias_type is fmha.attn_bias.LowerTriangularMaskWithTensorBias:
        attn_bias = _create_aligned_bias(
            batch_size,
            num_heads_groups,
            num_heads,
            q_len,
            kv_len,
            device=device,
            dtype=dtype,
        )
        if fmt in ["BMK", "BMHK"]:
            attn_bias = attn_bias[:, 0]
        if fmt == "BMK":
            attn_bias = attn_bias[:, 0]
        if requires_grad:
            attn_bias.requires_grad_(True)
        return fmha.attn_bias.LowerTriangularMaskWithTensorBias(attn_bias)
    if bias_type in [
        fmha.attn_bias.BlockDiagonalMask,
        fmha.attn_bias.BlockDiagonalCausalMask,
        fmha.attn_bias.BlockDiagonalCausalFromBottomRightMask,
        fmha.attn_bias.BlockDiagonalCausalLocalAttentionMask,
        fmha.attn_bias.BlockDiagonalCausalLocalAttentionFromBottomRightMask,
    ]:
        # These bias types are not supported in BMK format
        assert fmt in ["BMGHK", "BMHK"]
        max_q_minus_k = None
        if bias_type in {
            fmha.attn_bias.BlockDiagonalCausalFromBottomRightMask,
            fmha.attn_bias.BlockDiagonalCausalLocalAttentionFromBottomRightMask,
        }:
            max_q_minus_k = 0
        elif bias_type == fmha.attn_bias.BlockDiagonalCausalLocalAttentionMask:
            assert window_size is not None
            max_q_minus_k = window_size - 1

        block_diag = fmha.attn_bias.BlockDiagonalMask.from_seqlens(
            *_rand_seqlens(
                r,
                batch_size,
                q_len,
                kv_len,
                max_q_minus_k=max_q_minus_k,
            )
        )
        if bias_type is fmha.attn_bias.BlockDiagonalCausalMask:
            block_diag = block_diag.make_causal()
        if bias_type in {
            fmha.attn_bias.BlockDiagonalCausalLocalAttentionMask,
            fmha.attn_bias.BlockDiagonalCausalLocalAttentionFromBottomRightMask,
        }:
            block_diag = fmha.attn_bias.BlockDiagonalMask(
                q_seqinfo=block_diag.q_seqinfo,
                k_seqinfo=block_diag.k_seqinfo,
                _batch_sizes=block_diag._batch_sizes,
            )
            assert window_size is not None
            if bias_type is fmha.attn_bias.BlockDiagonalCausalLocalAttentionMask:
                block_diag = block_diag.make_local_attention(window_size)
            else:
                block_diag = block_diag.make_local_attention_from_bottomright(
                    window_size
                )
        if bias_type is fmha.attn_bias.BlockDiagonalCausalFromBottomRightMask:
            block_diag = block_diag.make_causal_from_bottomright()
        return block_diag
    if bias_type in [
        fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask,
        fmha.attn_bias.PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
    ]:
        assert fmt in ["BMHK", "BMGHK"]
        q, k = _rand_seqlens_padded_k(r, batch_size, q_len, kv_len)
        g_block_diag = (
            fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
                q_seqlen=q,
                kv_padding=kv_len,
                kv_seqlen=k,
            )
        )
        if bias_type == fmha.attn_bias.PagedBlockDiagonalCausalWithOffsetPaddedKeysMask:
            page_size = r.choice([64, 128, 256])
            pages_per_row = (kv_len + page_size - 1) // page_size
            block_tables = torch.randperm(
                batch_size * pages_per_row, device=device
            ).reshape(batch_size, pages_per_row)
            return g_block_diag.make_paged(
                block_tables=block_tables, page_size=page_size
            )
        return g_block_diag
    if bias_type == fmha.attn_bias.LocalAttentionFromBottomRightMask:
        return bias_type(
            window_left=r.randint(0, 5),
            window_right=r.randint(0, 5),
        )

    assert False, f"Unsupported bias type: {bias_type}"


def _rand_seqlens(
    r: random.Random,
    bs: int,
    q_len: int,
    kv_len: int,
    max_q_minus_k: Optional[int],
) -> Tuple[Sequence[int], Sequence[int]]:
    """
    Generates lists of lengths of query blocks and corresponding key blocks.
    The total number of queries will be bs * q_len and the
    total number of keys will be bs * kv_len.
    max_q_minus_k: maximum allowed num_queries - num_keys.
        For "bottom-right" masks it's 0, we need to have more keys than
        queries, otherwise some queries have no keys to attend to.
        For BlockDiagonalCausalMask it's None, there is no constraint
        on num_queries - num_keys.
        For BlockDiagonalCausalLocalAttentionMask it's equal
        to the window size.
    """
    if max_q_minus_k == 0:
        # In case max_q_minus_k > 0 the exact condition is
        # kv_len >= q_len - max_q_minus_k * batch_size,
        # but we can't check it without knowing the actual batch size,
        # which is determined in the loop below.
        assert kv_len >= q_len
    q_len *= bs
    kv_len *= bs
    seqlens_q: List[int] = []
    seqlens_k: List[int] = []

    step_q = [max(1, q_len // 10), max(2, q_len // 2)]
    step_k = [max(1, kv_len // 10), max(2, kv_len // 2)]
    while sum(seqlens_q) < q_len and sum(seqlens_k) < kv_len:
        if max_q_minus_k is None:
            # Simple case - no constraint on the number of queries and keys.
            num_queries = r.randrange(*step_q)
            seqlens_q.append(num_queries)
            seqlens_k.append(r.randrange(*step_k))
        else:
            # In this case we need to make sure num_queries - num_keys < max_q_minus_k holds for every batch element.
            # To do this, when choosing num_queries and num_keys at a given step,
            # we ensure two conditions are satisfied:
            # 1) num_queries <= num_keys + max_q_minus_k for the current batch element
            # 2) Same holds for the remaining keys and queries, i.e.
            #    queries_left - num_queries <= keys_left - num_keys + max_q_minus_k
            keys_left = kv_len - sum(seqlens_k, 0)
            queries_left = q_len - sum(seqlens_q, 0)

            assert (
                keys_left >= queries_left - max_q_minus_k
            ), f"{keys_left=} {queries_left=} {max_q_minus_k=} {kv_len=} {q_len=} {seqlens_k=} {seqlens_q=}"
            # Limit num_queries from above: if num_queries > keys_left + max_q_minus_k,
            # condition num_queries <= num_keys + max_q_minus_k can't be satisfied even if we take
            # all the remaining keys
            max_queries_to_take = min(queries_left, keys_left + max_q_minus_k)
            num_queries = r.randrange(1, max_queries_to_take + 1)
            seqlens_q.append(num_queries)

            # Now we know num_queries, let's select num_keys.
            # How many keys can we use for the current batch element so that
            # for the remaining keys and values the constraint
            # num_queries - num_keys < max_q_minus_k holds on the next step?
            extra_keys_available = keys_left - queries_left + max_q_minus_k + 1
            assert extra_keys_available >= 0
            if extra_keys_available > 0:
                seqlens_k.append(num_queries + r.randrange(0, extra_keys_available))
            else:
                seqlens_k.append(num_queries)
    seqlens_q[-1] = q_len - sum(seqlens_q[:-1])
    seqlens_k[-1] = kv_len - sum(seqlens_k[:-1])
    return seqlens_q, seqlens_k


def _rand_maxed_partition(
    r: random.Random, total: int, n: int, mx: int, positive: bool = True
) -> List[int]:
    # returns list of n nonnegative integers less than mx summing to total
    # NB: This is unfortunately biased towards evenly-split bins.
    # If `positive`, outputs are positive
    if positive:
        total -= n
        mx -= 1
    idxs = r.sample(range(n * mx), total)
    y = torch.zeros(n, mx, dtype=torch.int32)
    y.flatten()[idxs] = 1
    z = y.sum(1)
    if positive:
        z += 1
    return z.tolist()


def _rand_seqlens_padded_k(
    r: random.Random, bs: int, q_len: int, kv_len: int
) -> Tuple[Sequence[int], Sequence[int]]:
    # This is for BlockDiagonalCausalWithOffsetPaddedKeysMask.
    # we need q_seqlens and k_seqlens to be of len bsz.
    # For each "batch element" there must be more keys than queries
    # because this bias type is "bottom right" and so any extra queries
    # will attend to nothing and have undefined result.
    # In addition every element of k_seqlens must be <= kv_len
    if q_len > kv_len:
        raise ValueError("need more keys than values")
    if q_len == kv_len:
        # all key slots are needed so we cannot have padding
        q_seqlens = k_seqlens = [kv_len] * bs
    else:
        q_seqlens = _rand_maxed_partition(r, q_len * bs, bs, kv_len)
        k_seqlens = [r.randint(i, kv_len) for i in q_seqlens]
    return q_seqlens, k_seqlens
