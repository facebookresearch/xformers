# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
from typing import List, Optional, Sequence, Tuple, Type

import torch

from xformers.ops import AttentionBias, fmha
from xformers.ops.fmha.attn_bias import AttentionBiasSubTensor
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
    op: Optional[Type[AttentionOpBase]] = None,
    page_size: Optional[int] = None,
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
        elif op is not None and issubclass(op, fmha.triton_splitk.FwOp):
            attn_bias = (
                torch.randn(
                    (batch_size, num_heads_groups, num_heads, q_len, kv_len),
                    device=device,
                    dtype=dtype,
                )
                * 3
            )
            if fmt in ["BMK", "BMHK"]:
                attn_bias = attn_bias[:, 0]
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
        fmha.attn_bias.BlockDiagonalPaddedKeysMask,
        fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask,
        fmha.attn_bias.PagedBlockDiagonalPaddedKeysMask,
        fmha.attn_bias.PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
    ]:
        assert fmt in ["BMHK", "BMGHK"]
        q, k = _rand_seqlens_padded_k(r, batch_size, q_len, kv_len)
        block_diag_type = (
            bias_type._UNPAGED_TYPE
            if issubclass(bias_type, fmha.attn_bias.PagedBlockDiagonalPaddedKeysMask)
            else bias_type
        )
        g_block_diag = block_diag_type.from_seqlens(
            q_seqlen=q,
            kv_padding=kv_len,
            kv_seqlen=k,
        )
        if issubclass(bias_type, fmha.attn_bias.PagedBlockDiagonalPaddedKeysMask):
            assert page_size is not None
            pages_per_row = (kv_len + page_size - 1) // page_size
            block_tables = torch.tensor(
                r.sample(range(batch_size * pages_per_row), batch_size * pages_per_row),
                device=device,
                dtype=torch.int32,
            ).reshape(batch_size, pages_per_row)
            return g_block_diag.make_paged(
                block_tables=block_tables, page_size=page_size, paged_type=bias_type
            )
        return g_block_diag
    if bias_type in [
        fmha.attn_bias.BlockDiagonalCausalWithOffsetGappyKeysMask,
        fmha.attn_bias.BlockDiagonalGappyKeysMask,
    ]:
        assert fmt in ["BMHK", "BMGHK"]
        max_q_minus_k = (
            None if bias_type is fmha.attn_bias.BlockDiagonalGappyKeysMask else 0
        )
        q, k = _rand_seqlens(r, batch_size, q_len, kv_len, max_q_minus_k)
        total_kv_len = kv_len * batch_size
        starts = [r.randint(0, total_kv_len - ki) for ki in k] + [total_kv_len]
        return fmha.attn_bias.BlockDiagonalGappyKeysMask.from_seqlens(
            q_seqlen=q,
            kv_seqstarts=starts,
            kv_seqlen=k,
        )
    if bias_type in [
        fmha.attn_bias.PagedBlockDiagonalGappyKeysMask,
    ]:
        assert fmt in ["BMHK", "BMGHK"]
        assert page_size is not None
        pages_per_row = (kv_len + page_size - 1) // page_size
        total_queries = q_len * batch_size
        q = _rand_maxed_partition(r, total_queries, batch_size, total_queries, False)
        k = [r.randint(1, kv_len) for _ in range(batch_size)]
        row_size = pages_per_row * page_size
        starts = [row_size * i + r.randint(0, row_size - ki) for i, ki in enumerate(k)]
        starts.append(pages_per_row * batch_size * page_size)
        block_diag_type = bias_type._UNPAGED_TYPE  # type: ignore
        g_block_diag = block_diag_type.from_seqlens(
            q_seqlen=q,
            kv_seqstarts=starts,
            kv_seqlen=k,
        )
        block_tables = torch.tensor(
            r.sample(range(batch_size * pages_per_row), batch_size * pages_per_row),
            device=device,
            dtype=torch.int32,
        ).reshape(batch_size, pages_per_row)
        return g_block_diag.make_paged(
            block_tables=block_tables,
            page_size=page_size,
            paged_type=bias_type,
            notional_padding=page_size * pages_per_row,
        )
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


def ref_attention(q, k, v, attn_bias=None, drop_mask=None, p=0.0, scale=None):
    if q.ndim == 5:

        def attn_bias_group(group: int):
            if isinstance(attn_bias, fmha.attn_bias.AttentionBiasSubTensor):
                if attn_bias.HOLDS_DENSE_TENSOR:
                    return attn_bias[:, group]
            elif isinstance(attn_bias, torch.Tensor):
                return attn_bias[:, group]
            return attn_bias

        return torch.stack(
            [
                ref_attention_bmhk(
                    q[:, :, g],
                    k[:, :, g],
                    v[:, :, g],
                    scale=scale,
                    attn_bias=attn_bias_group(g),
                )
                for g in range(q.shape[2])
            ],
            dim=2,
        )
    if q.ndim == 4:
        assert p == 0.0
        return ref_attention_bmhk(q, k, v, scale=scale, attn_bias=attn_bias)
    q = q.float()
    k = k.float()
    v = v.float()

    scale = scale if scale is not None else (1 / q.shape[-1] ** 0.5)
    q = q * scale

    attn = q @ k.transpose(-2, -1)
    if attn_bias is not None:
        if isinstance(attn_bias, (AttentionBias, AttentionBiasSubTensor)):
            # Always create in B,H,Mq,Mk format
            attn_bias_tensor = attn_bias.materialize(
                (q.shape[0], 1, q.shape[1], k.shape[1]),
                device=q.device,
                dtype=torch.float32,
            )
        else:
            attn_bias_tensor = attn_bias
        if attn_bias_tensor.ndim == 4:
            assert q.shape[0] == attn_bias_tensor.shape[0] * attn_bias_tensor.shape[1]
            attn_bias_tensor = attn_bias_tensor.reshape(
                [-1, *attn_bias_tensor.shape[2:]]
            )
        attn = attn + attn_bias_tensor.float()
    attn = attn.softmax(-1)
    if drop_mask is not None:
        attn = attn * (drop_mask / (1 - p))
    return attn @ v


def ref_attention_bmhk(q, k, v, attn_bias, scale=None) -> torch.Tensor:
    assert q.ndim == 4

    def T(t):
        return t.permute((0, 2, 1, 3)).reshape(
            [t.shape[0] * t.shape[2], t.shape[1], t.shape[3]]
        )

    if isinstance(attn_bias, (AttentionBias, AttentionBiasSubTensor)):
        attn_bias = attn_bias.materialize(
            (q.shape[0], q.shape[2], q.shape[1], k.shape[1]),
            device=q.device,
            dtype=torch.float32,
        ).reshape([q.shape[0] * q.shape[2], q.shape[1], k.shape[1]])
    out = ref_attention(T(q), T(k), T(v), attn_bias, scale=scale)
    out = out.reshape([q.shape[0], q.shape[2], q.shape[1], v.shape[3]])
    return out.permute((0, 2, 1, 3))
