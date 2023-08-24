# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
from typing import List, Optional, Sequence, Tuple, Type, TypeVar, Set, Any

import pytest
import torch

## need to FIX
##from scipy.stats import binomtest
from torch.utils.checkpoint import checkpoint

import xformers.ops
from xformers.ops import fmha
from xformers.ops.fmha.common import AttentionOpBase

from tests.utils import assert_allclose

torch.backends.cuda.matmul.allow_tf32 = False
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
_devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]

ALL_FW_OPS: Sequence[Type[fmha.common.AttentionFwOpBase]] = [
    fmha.ck.FwOp,
]

T = TypeVar(
    "T", Type[fmha.common.AttentionFwOpBase], Type[fmha.common.AttentionBwOpBase]
)

def ref_attention(q, k, v, attn_bias=None, drop_mask=None, p=0.0, scale=None):
    if q.ndim == 4:
        assert p == 0.0
        return ref_attention_bmhk(q, k, v, attn_bias=attn_bias)
    q = q.float()
    k = k.float()
    v = v.float()

    scale = scale if scale is not None else (1 / q.shape[-1] ** 0.5)
    q = q * scale

    attn = q @ k.transpose(-2, -1)
    if attn_bias is not None:
        if isinstance(attn_bias, xformers.ops.AttentionBias):
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

    if isinstance(attn_bias, xformers.ops.AttentionBias):
        attn_bias = attn_bias.materialize(
            (q.shape[0], q.shape[2], q.shape[1], k.shape[1]),
            device=q.device,
            dtype=torch.float32,
        ).reshape([q.shape[0] * q.shape[2], q.shape[1], k.shape[1]])
    out = ref_attention(T(q), T(k), T(v), attn_bias, scale=scale)
    out = out.reshape([q.shape[0], q.shape[2], q.shape[1], v.shape[3]])
    return out.permute((0, 2, 1, 3))


def _rand_seqlens(
    r: random.Random,
    bs: int,
    q_len: int,
    kv_len: int,
    more_keys_than_queries_per_block: bool,
) -> Tuple[Sequence[int], Sequence[int]]:
    """
    Generates lists of lengths of query blocks and corresponding key blocks.
    The total number of queries will be bs * q_len and the
    total number of keys will be bs * kv_len.
    """
    if more_keys_than_queries_per_block:
        assert kv_len >= q_len
    q_len *= bs
    kv_len *= bs
    seqlens_q: List[int] = []
    seqlens_k: List[int] = []

    step_q = [max(1, q_len // 10), max(2, q_len // 2)]
    step_k = [max(1, kv_len // 10), max(2, kv_len // 2)]
    while sum(seqlens_q) < q_len and sum(seqlens_k) < kv_len:
        num_queries = r.randrange(*step_q)
        seqlens_q.append(num_queries)

        if more_keys_than_queries_per_block:
            # Must select at least `num_queries` keys
            # But also leave enough keys for later
            keys_left = kv_len - sum(seqlens_k, 0)
            queries_left = q_len - sum(seqlens_q[:-1], 0)
            assert keys_left >= queries_left
            seqlens_k.append(num_queries + r.randrange(0, keys_left - queries_left))
        else:
            seqlens_k.append(r.randrange(*step_k))
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


def _create_aligned_bias(B: int, H: int, Mq: int, Mkv: int, **kwargs) -> torch.Tensor:
    align_to = 8
    return (
        torch.randn(
            (
                B,
                H,
                Mq,
                align_to * ((Mkv + align_to - 1) // align_to),
            ),
            **kwargs,
        )
        * 3
    )[:, :, :, :Mkv]


def create_attn_bias(
    bias_type,
    batch_size: int,
    num_heads: int,
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
    if bias_type is torch.Tensor:
        if fmt == "BMK":
            batch_size *= num_heads
            num_heads = 1
        ##`small_k` only supports an expanded 1d bias
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
                num_heads,
                q_len,
                kv_len,
                device=device,
                dtype=dtype,
            )

            # ToDo: need a fix in ck-flashAttn to avoid divided-by-zero when all-(-inf) occurred
            #       with the data read by one-thread
            # make sure it also works if the first columns are partially masked out
            # attn_bias[0, 0, q_len - 1 :, : num_heads - 2] = -math.inf

        if requires_grad:
            attn_bias.requires_grad_(True)
        return attn_bias
    if bias_type is fmha.attn_bias.LowerTriangularMask:
        return fmha.attn_bias.LowerTriangularMask()
    if bias_type is fmha.attn_bias.LowerTriangularMaskWithTensorBias:
        attn_bias = _create_aligned_bias(
            batch_size,
            num_heads,
            q_len,
            kv_len,
            device=device,
            dtype=dtype,
        )
        if requires_grad:
            attn_bias.requires_grad_(True)
        return fmha.attn_bias.LowerTriangularMaskWithTensorBias(attn_bias)
    if bias_type in [
        fmha.attn_bias.BlockDiagonalMask,
        fmha.attn_bias.BlockDiagonalCausalMask,
        fmha.attn_bias.BlockDiagonalCausalFromBottomRightMask,
    ]:
        # This bias is not supported in BMK format
        assert fmt == "BMHK"
        block_diag = fmha.attn_bias.BlockDiagonalMask.from_seqlens(
            *_rand_seqlens(
                r,
                batch_size,
                q_len,
                kv_len,
                more_keys_than_queries_per_block=bias_type
                is fmha.attn_bias.BlockDiagonalCausalFromBottomRightMask,
            )
        )
        if bias_type is fmha.attn_bias.BlockDiagonalCausalMask:
            block_diag = block_diag.make_causal()
        if bias_type is fmha.attn_bias.BlockDiagonalCausalFromBottomRightMask:
            block_diag = block_diag.make_causal_from_bottomright()
        return block_diag
    if bias_type == fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask:
        assert fmt == "BMHK"
        q, k = _rand_seqlens_padded_k(r, batch_size, q_len, kv_len)
        g_block_diag = (
            fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
                q_seqlen=q,
                kv_padding=kv_len,
                kv_seqlen=k,
            )
        )
        return g_block_diag

    assert False, f"Unsupported bias type: {bias_type}"

def create_tensors(
    op: Type[AttentionOpBase],
    device,
    dtype,
    attn_bias_type,
    B,
    q_len,
    kv_len,
    h,
    k,
    kv,
    *,
    attn_bias_requires_grad: bool = False,
    fmt: str = "BMK",
):
    torch.manual_seed(B * q_len + kv_len * k + kv)
    scale = 3
    if fmt == "BMK":
        query = torch.randn((B * h, q_len, k), device=device, dtype=dtype).mul_(scale)
        key = torch.randn((B * h, kv_len, k), device=device, dtype=dtype).mul_(scale)
        value = torch.randn((B * h, kv_len, kv), device=device, dtype=dtype).mul_(scale)
    else:
        assert fmt == "BMHK"
        query = torch.randn((B, q_len, h, k), device=device, dtype=dtype).mul_(scale)
        key = torch.randn((B, kv_len, h, k), device=device, dtype=dtype).mul_(scale)
        value = torch.randn((B, kv_len, h, kv), device=device, dtype=dtype).mul_(scale)

    if fmt == "BMK" and not fmha.common._is_bias_type_supported_in_BMK(attn_bias_type):
        attn_bias_type = None
    attn_bias = None
    if attn_bias_type is not None:
        attn_bias = create_attn_bias(
            attn_bias_type,
            batch_size=B,
            num_heads=h,
            q_len=q_len,
            kv_len=kv_len,
            dtype=dtype,
            device=device,
            requires_grad=attn_bias_requires_grad,
            fmt=fmt,
            op=op,
        )
        if isinstance(
            attn_bias,
            (
                fmha.attn_bias.BlockDiagonalMask,
                fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask,
            ),
        ):
            query, key, value = [
                x.reshape([1, -1, *x.shape[2:]]) for x in [query, key, value]
            ]

    inputs = fmha.Inputs(query=query, key=key, value=value, attn_bias=attn_bias)
    reasons = op.not_supported_reasons(inputs)
    if reasons:
        err_msg = f"{op.NAME}: unsupported ({'/'.join(reasons)})"
        # Ensure we free memory to avoid OOMs
        del query, key, value, attn_bias, inputs
        pytest.skip(err_msg)
    return query, key, value, attn_bias

## The same set of supported attn_bias types as defined by ck.FwOp
SUPPORTED_ATTN_BIAS_TYPES: Set[Any] = {
        type(None),
        torch.Tensor,
        fmha.attn_bias.LowerTriangularMask,
        fmha.attn_bias.LowerTriangularMaskWithTensorBias,
        fmha.attn_bias.BlockDiagonalMask,
        fmha.attn_bias.BlockDiagonalCausalMask,
        fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask,
        fmha.attn_bias.BlockDiagonalCausalFromBottomRightMask,
        }

@pytest.mark.parametrize("bias_type", SUPPORTED_ATTN_BIAS_TYPES)
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("fmt", ["BMK", "BMHK"])
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
def test_forward(dtype, fmt, packed, bias_type):
    op = fmha.ck.FwOp
    device = torch.device("cuda")
    batch_size = 7
    q_len = 200

    ## BottomRightMask requires generate {m0,m1,...}, {n0,n1,...} where mi <= ni
    if bias_type is fmha.attn_bias.BlockDiagonalCausalFromBottomRightMask:
        kv_len = int(q_len * 1.2) 
    else:
        kv_len = q_len
    h = 3 
    k = 64 
    kv = 64

    if kv > 128:
        pytest.skip("kv > 128 is not supported by CK-FlashAttention-1")

    if packed and not (k == kv and q_len == kv_len):
        pytest.skip(
            f"packed incompatible with `k ({k}) != kv ({kv})` or `q_len ({q_len}) != kv_len ({kv_len})`"
        )
    if fmt == "BMK" and not fmha.common._is_bias_type_supported_in_BMK(bias_type):
        pytest.skip("BMK incompatible with this bias")

    ## packed type always creates the tensors in "BMHK" even the fmt is "BMK", so for packed type, one
    ## should always assume h is already merged in B, and set h to be 1
    if packed and fmt is "BMK" and batch_size > 1 and h > 1:
        pytest.skip("Shape of this is type is skipped")

    query, key, value, attn_bias = create_tensors(
        op, device, dtype, bias_type, batch_size, q_len, kv_len, h, k, kv, fmt="BMHK" if packed else fmt
    )

    ## when packed, the query, key, value is in BMHK format
    if packed:
        c = torch.stack([query, key, value], 2)
        if fmt == "BMK":
            # bm3hk -> 3bhmk -> 3Bmk
            c = c.permute(2, 0, 3, 1, 4).view([3, -1, q_len, k])
            query, key, value = c[0], c[1], c[2]
            # Re-create bias in the right format
            attn_bias = create_attn_bias(
                bias_type=bias_type,
                batch_size=batch_size,
                num_heads=h,
                q_len=q_len,
                kv_len=kv_len,
                device=device,
                dtype=dtype,
                requires_grad=False,
                fmt=fmt,
                op=op,
            )
        else:
            # bm3hk -> 3 x bmhk
            query, key, value = xformers.ops.unbind(c, 2)

        print("The query shaped for packed: ", query.size())
        assert not query.is_contiguous()

    out = xformers.ops.memory_efficient_attention_forward(
        query, key, value, attn_bias, op=op
    )
    assert not out.isnan().any(), ("Output has NaNs", attn_bias)
    out2 = xformers.ops.memory_efficient_attention_forward(
        query, key, value, attn_bias, op=op
    )
    assert torch.allclose(out, out2, atol=0.0, rtol=0.0), (
        "Non-deterministic behavior",
        attn_bias,
    )

    ref = ref_attention(query, key, value, attn_bias)
    assert out.shape == ref.shape, out.shape
    assert_allclose(
        out.float(),
        ref,
        atol=op.ERROR_ATOL[dtype],
        rtol=op.ERROR_RTOL.get(dtype, 1e-5),
    )

