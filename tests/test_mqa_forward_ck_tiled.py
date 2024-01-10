# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
from typing import List, Optional, Sequence, Tuple, Type, TypeVar

import pytest
import torch
from scipy.stats import binomtest
from torch.utils.checkpoint import checkpoint

import xformers.ops
from xformers.ops import fmha
from xformers.ops.fmha.common import AttentionOpBase

from .utils import assert_allclose

torch.backends.cuda.matmul.allow_tf32 = False
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

_devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
_types = [torch.float16, torch.bfloat16]

T = TypeVar(
    "T", Type[fmha.common.AttentionFwOpBase], Type[fmha.common.AttentionBwOpBase]
)

ALL_FW_OPS: Sequence[Type[fmha.common.AttentionFwOpBase]] = [
    fmha.ck.FwOp,
]

ALL_BW_OPS: Sequence[Type[fmha.common.AttentionBwOpBase]] = [
    fmha.ck.BwOp,
]

def sample_random_supported_fw(
    inp: fmha.Inputs, seed: int
) -> Type[fmha.common.AttentionFwOpBase]:
    r = random.Random(seed)
    fw_ops = list(ALL_FW_OPS)
    r.shuffle(fw_ops)
    for op in fw_ops:
        if op.supports(inp):
            return op
    raise NotImplementedError(f"Could not find a FW operator for: {inp}")


def generate_test_shapes_B_Mq_Mkv_H_K_Kv(op):
    shapes = []
    for B in op._TEST_BATCH_SIZES:
        for Mq in [32, 256]:
            for Mkv in [32, 64, 256, 1024]:
                for K in op._TEST_K:
                    shapes.append((B, Mq, Mkv, 1, K, K))
        Mq = 256
        Mkv = 128
        K = 32
        H = 1
        # Weird values of parameters
        for M in [2, 3, 15, 31, 32, 34, 68, 72, 90, 132, 136]:
            shapes.append((B, M, Mkv, H, K, K))
            shapes.append((B, Mq, M, H, K, K))
        for _K in [1, 2, 3, 31, 34, 36, 38, 40, 64, 80, 160, 256 + 2, 256 + 8, 512]:
            if _K <= op.SUPPORTED_MAX_K:
                shapes.append((B, Mq, Mkv, H, _K, _K))
        # Different value for K / Kv
        if op.SUPPORTS_DIFFERENT_VALUE_EMBED:
            for _K in [32, 36, 64, 256 + 8]:
                shapes.append((B, Mq, Mkv, H, K, _K))
                shapes.append((B, Mq, Mkv, H, _K, K))
        # Exotic sizes
        for K in op._TEST_K:
            shapes.append((B, 16, 1024, H, K, K))
            shapes.append((B, 1024, 16, H, K, K))
        # Some number of heads
        for H in [3, 5, 12]:
            shapes.append((max(1, B // H), Mq, Mkv, H, K, K))
    # Filter-out not supported shapes
    shapes = [
        shape
        for shape in shapes
        if len(
            op.shape_not_supported_reasons(
                Mq=shape[1], Mkv=shape[2], K=shape[4], Kv=shape[5]
            )
        )
        == 0
    ]
    # Add some random shapes
    if op in [
        fmha.ck.FwOp,
        fmha.ck.BwOp,
    ]:
        K_CHOICES = [8 * i for i in range(1, 256 // 8)]
        r = random.Random(0)
        found_count = 0
        while found_count < 20:
            B = r.randint(1, 400)
            Mq = r.randint(1, 500)
            Mkv = r.randint(1, 500)
            H = r.randint(2, 11)
            B = max(B // H, 1)
            K = r.choice(K_CHOICES)
            Kv = r.choice(K_CHOICES)
            if not op.SUPPORTS_DIFFERENT_VALUE_EMBED:
                Kv = K
            if len(op.shape_not_supported_reasons(Mq, Mkv, K, Kv)):
                continue
            found_count += 1
            shapes.append((B, Mq, Mkv, H, K, Kv))
    return shapes


def make_id(op, device, dtype, bias_type, *shape):
    return (
        f"{op.NAME}-{device}-{str(dtype)}-{bias_type.__name__}"
        f"-{'-'.join([str(s) for s in shape])}"
    )


def _generate_op_device_dtype_biasT_B_Mq_Mkv_H_K_Kv(
    ops_list: Sequence[Type[fmha.AttentionOpBase]], max_shapes_per_op: int = 65000
):
    r = random.Random(0)
    combination = []
    ids = []
    for op in ops_list:
        op_count = 0
        # Sort list of masks, so it's deterministic across runs
        LIST_MASKS = list(sorted(op.SUPPORTED_ATTN_BIAS_TYPES, key=lambda x: str(x)))
        for shape in generate_test_shapes_B_Mq_Mkv_H_K_Kv(op):
            has_one = False
            for device in _devices:
                if device not in op.SUPPORTED_DEVICES:
                    continue
                for dtype in op.SUPPORTED_DTYPES:
                    bias_type = r.choice(LIST_MASKS)
                    # Avoid using too much memory
                    if bias_type not in [
                        type(None),
                        fmha.attn_bias.LowerTriangularMask,
                    ]:
                        B, Mq, Mkv, H, K, Kv = shape
                        B = min(B, 12)

                        if (
                            bias_type
                            is fmha.attn_bias.BlockDiagonalCausalFromBottomRightMask
                        ):
                            Mq, Mkv = min(Mkv, Mq), max(Mkv, Mq) + 2
                        elif (
                            bias_type
                            is fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask
                        ):
                            Mq, Mkv = min(Mkv, Mq), max(Mkv, Mq)
                        shape = (B, Mq, Mkv, H, K, Kv)
                    combination.append((op, device, dtype, bias_type, *shape))
                    ids.append(
                        f"{op.NAME}-{device}-{str(dtype)}-{bias_type.__name__}"
                        f"-{'-'.join([str(s) for s in shape])}"
                    )
                    has_one = True
            if has_one:
                op_count += 1
            if op_count > max_shapes_per_op:
                break
        # Some specific shapes for which we want to run without any mask
        bias_type = type(None)
        for shape in (
            # Some strides/dims don't fit on an uint16
            (1, 128, 128, 300, 128, 128),
            (13, 1, 67, 200, 8, 8),
            (1, 1 + 2**16, 4, 1, 8, 8),
            (1, 4, 1 + 2**16, 1, 8, 8),
            # TODO: Some strides don't fit on an uint32
            # Crashes on Flash, Errors on Cutlass
            # (1, 1, 64000, 300, 128, 128)
        ):
            for device in _devices:
                if device not in op.SUPPORTED_DEVICES:
                    continue
                for dtype in op.SUPPORTED_DTYPES:
                    combination.append((op, device, dtype, bias_type, *shape))
    return {
        "argvalues": combination,
        "ids": [make_id(*c) for c in combination],
    }


parametrize_opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv = pytest.mark.parametrize(
    "opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv",
    **_generate_op_device_dtype_biasT_B_Mq_Mkv_H_K_Kv(ALL_FW_OPS),
)
parametrize_opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv__xs = pytest.mark.parametrize(
    "opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv",
    **_generate_op_device_dtype_biasT_B_Mq_Mkv_H_K_Kv(ALL_FW_OPS, max_shapes_per_op=1),
)
parametrize_opBW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv = pytest.mark.parametrize(
    "opBW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv",
    **_generate_op_device_dtype_biasT_B_Mq_Mkv_H_K_Kv(ALL_BW_OPS),
)
parametrize_opBW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv__xs = pytest.mark.parametrize(
    "opBW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv",
    **_generate_op_device_dtype_biasT_B_Mq_Mkv_H_K_Kv(ALL_BW_OPS, max_shapes_per_op=1),
)

def ref_attention(q, k, v, attn_bias=None, drop_mask=None, p=0.0, scale=None, dtype=None):
    if q.ndim == 4:
        B, M, Hq, K = q.shape
        _, N, Hkv, Kv = v.shape
        nhead_ratio_qk = Hq // Hkv

        def attn_bias_head(head: int):
            if isinstance(attn_bias, torch.Tensor):
                assert attn_bias.ndim == 4
                _, H, _, _ = attn_bias.shape        
                assert H == Hq
                bias_bghmn = attn_bias.reshape(B, Hkv, nhead_ratio_qk, M, N)
                return bias_bghmn[:, :, head]
            if isinstance(attn_bias, fmha.attn_bias.LowerTriangularMaskWithTensorBias):
                assert attn_bias._bias.ndim == 4
                _, H, _, _ = attn_bias._bias.shape        
                assert H == Hq
                bias_bghmn = attn_bias._bias.reshape(B, Hkv, nhead_ratio_qk, M, N)

                return fmha.attn_bias.LowerTriangularMaskWithTensorBias(
                    bias_bghmn[:, :, head]
                )
            return attn_bias

        q_bmghk = q.reshape((B, M, Hkv, nhead_ratio_qk, K))

        return torch.stack(
            [
                ref_attention_bmhk(
                    q_bmghk[:, :, :, h], k, v, attn_bias=attn_bias_head(h), dtype=dtype
                )
                for h in range(q_bmghk.shape[3])
            ],
            dim=3,
        ).reshape((B, M, Hq, Kv))
     
    assert q.ndim == 3
    if dtype is None:
        dtype = torch.float32
    q = q.to(dtype=dtype)
    k = k.to(dtype=dtype)
    v = v.to(dtype=dtype)

    scale = scale if scale is not None else (q.shape[-1] ** -0.5)
    q = q * scale

    attn = q @ k.transpose(-2, -1)
    if attn_bias is not None:
        if isinstance(attn_bias, xformers.ops.AttentionBias):
            # Always create in B,H,Mq,Mk format
            attn_bias_tensor = attn_bias.materialize(
                (q.shape[0], 1, q.shape[1], k.shape[1]),
                device=q.device,
                dtype=dtype,
            )
        else:
            attn_bias_tensor = attn_bias.to(dtype=dtype)
        if attn_bias_tensor.ndim == 4:
            assert q.shape[0] == attn_bias_tensor.shape[0] * attn_bias_tensor.shape[1]
            attn_bias_tensor = attn_bias_tensor.reshape(
                [-1, *attn_bias_tensor.shape[2:]]
            )
        attn = attn + attn_bias_tensor
    attn = attn.softmax(-1)
    if drop_mask is not None:
        attn = attn * (drop_mask / (1 - p))
    return attn @ v


def ref_attention_bmhk(q, k, v, attn_bias, scale=None, dtype=None) -> torch.Tensor:
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
    out = ref_attention(T(q), T(k), T(v), attn_bias, scale=scale, dtype=dtype)
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


def _rand_partition(r: random.Random, total: int, n: int) -> List[int]:
    # returns list of n nonnegative integers summing to total
    idx = {0, total}
    while len(idx) < n + 1:
        idx.add(r.randint(1, total - 1))
    s = sorted(idx)
    return [e - b for b, e in zip(s[:-1], s[1:])]


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
                num_heads,
                q_len,
                kv_len,
                device=device,
                dtype=dtype,
            )
            # ToDo: need a fix in ck-flashAttn to avoid divided-by-zero when all-(-inf) occurred
            #       with the data read by one-thread
            # make sure it also works if the first columns are partially masked out
            ## attn_bias[0, 0, q_len - 1 :, : num_heads - 2] = -math.inf

        if requires_grad:
            attn_bias.requires_grad_(True)
        if fmt == "BMK":
            attn_bias = attn_bias[:, 0]
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


def get_bias_grad(attn_bias, clear: bool = False) -> Optional[torch.Tensor]:
    tensor_with_grad: Optional[torch.Tensor] = None
    if isinstance(attn_bias, torch.Tensor):
        tensor_with_grad = attn_bias
    if isinstance(attn_bias, fmha.attn_bias.LowerTriangularMaskWithTensorBias):
        tensor_with_grad = attn_bias._bias
    if tensor_with_grad is not None:
        grad = tensor_with_grad.grad
        if clear:
            tensor_with_grad.grad = None
        return grad
    return None


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


def bmhk2bmk(tensor) -> torch.Tensor:
    return (
        tensor.permute((0, 2, 1, 3))
        .contiguous()
        .view([tensor.shape[0] * tensor.shape[2], tensor.shape[1], tensor.shape[3]])
    )


def bmk2bmhk(tensor, num_heads: int) -> torch.Tensor:
    return tensor.reshape([-1, num_heads, tensor.shape[1], tensor.shape[2]]).permute(
        (0, 2, 1, 3)
    )

@pytest.mark.parametrize("hdim_k,hdim_v", [(64, 64), (128, 128)])
@pytest.mark.parametrize("nhead_q,nhead_kv", [(8, 1), (8, 2), (12, 4), (4, 4)])
@pytest.mark.parametrize("seqlen_q,seqlen_kv", [(100, 128), (128, 100), (200, 1000), (400, 300)])
@pytest.mark.parametrize("batches", [100, 64, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("attn_bias_type", [type(None), torch.Tensor, fmha.attn_bias.LowerTriangularMask])
@pytest.mark.parametrize("op", [fmha.ck.FwOp])
def test_mqa_forward(
    op,
    attn_bias_type,
    dtype, 
    batches: int, 
    seqlen_kv: int, 
    seqlen_q: int, 
    nhead_kv: int, 
    nhead_q: int, 
    hdim_v: int, 
    hdim_k: int, 
):
    B = batches
    M = seqlen_q
    N = seqlen_kv
    Hq = nhead_q
    Hkv = nhead_kv
    K = hdim_k
    Kv = hdim_v

    print("Hq=", Hq, "Hkv=", Hkv)

    device = torch.device("cuda")

    if not (K == Kv and (Kv == 64 or Kv == 128)):
        pytest.skip("only head-dim size 64 or 128 supported by ck-tiled!")

    if Kv > 128:
        pytest.skip("kv > 128 is not supported by CK-FlashAttention")

    scale = 3
    query = torch.randn((B, M, Hq, K), device=device, dtype=dtype).mul_(scale)
    key = torch.randn((B, N, Hkv, K), device=device, dtype=dtype).mul_(scale)
    value = torch.randn((B, N, Hkv, Kv), device=device, dtype=dtype).mul_(scale)

    attn_bias = None
    if attn_bias_type is not None:
        attn_bias = create_attn_bias(
            attn_bias_type,
            batch_size=B,
            num_heads=Hq,
            q_len=M,
            kv_len=N,
            dtype=dtype,
            device=device,
            requires_grad=False,
            fmt="BMHK",
            op=op,
        )

    inputs = fmha.Inputs(query=query, key=key, value=value, attn_bias=attn_bias)
    reasons = op.not_supported_reasons(inputs)
    if reasons:
        err_msg = f"{op.NAME}: unsupported ({'/'.join(reasons)})"
        # Ensure we free memory to avoid OOMs
        del query, key, value, attn_bias, inputs

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

