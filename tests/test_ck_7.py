# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import List, Optional, Sequence, Tuple, Type, TypeVar

import pytest
import torch

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
            #
            # attn_bias[0, 0, q_len - 1 :, : num_heads - 2] = -math.inf

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


@pytest.mark.parametrize("fmt", ["BMK", "BMHK"])
@pytest.mark.parametrize("packed", [False, True])
@parametrize_opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv
def test_forward(
    opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv,
    packed,
    fmt,
):
    (
        op,
        device,
        dtype,
        bias_type,
        batch_size,
        q_len,
        kv_len,
        h,
        k,
        kv,
    ) = opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv

    if kv > 128:
        pytest.skip("kv > 128 is not supported by CK-FlashAttention-1")

    if packed and not (k == kv and q_len == kv_len):
        pytest.skip(
            f"packed incompatible with `k ({k}) != kv ({kv})` or `q_len ({q_len}) != kv_len ({kv_len})`"
        )
    if fmt == "BMK" and not fmha.common._is_bias_type_supported_in_BMK(bias_type):
        pytest.skip("BMK incompatible with this bias")

    query, key, value, attn_bias = create_tensors(
        *opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv, fmt="BMHK" if packed else fmt
    )

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


@pytest.mark.parametrize("k_len", [5, 6, 32])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("kv_len", [128, 512])
@pytest.mark.parametrize("q_len", [128, 512])
@pytest.mark.parametrize("device", [torch.device("cuda")])
@pytest.mark.parametrize("dtype", _types)
def test_key_query_all_ones(dtype, device, q_len, kv_len, batch_size, k_len):
    scale = 3
    query = torch.ones((batch_size, q_len, k_len), device=device, dtype=dtype)
    key = torch.ones((batch_size, kv_len, k_len), device=device, dtype=dtype)
    value = torch.randn((batch_size, kv_len, k_len), device=device, dtype=dtype) * scale

    out = xformers.ops.memory_efficient_attention(
        query, key, value, op=(fmha.ck.FwOp, None)
    )
    # this should be equivalent to the average over value
    ref = value.mean(1, keepdim=True).expand_as(query)

    if dtype is torch.float16:
        assert_allclose(out, ref, atol=1e-5)
    else:
        assert_allclose(out, ref, atol=1e-2)


def _block_diag_reshape_lse(
    lse: torch.Tensor, q_seqinfo: fmha.attn_bias._SeqLenInfo
) -> torch.Tensor:
    """LSE can be padded, let's remove the padding"""
    parts = []
    for slice, (start, end) in zip(lse.unbind(0), q_seqinfo.intervals()):
        parts.append(slice[:, : end - start])
    return torch.cat(parts, dim=1).unsqueeze(1)


@parametrize_opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv
def test_logsumexp(opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv):
    (
        op,
        device,
        dtype,
        bias_type,
        batch_size,
        q_len,
        kv_len,
        h,
        k,
        kv,
    ) = opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv
    query, key, value, attn_bias = create_tensors(
        *opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv, fmt="BMK"
    )

    _out, lse = xformers.ops.memory_efficient_attention_forward_requires_grad(
        query,
        key,
        value,
        op=op,
        attn_bias=attn_bias,
    )
    attn = (query.float() / k**0.5) @ key.float().transpose(-2, -1)
    if attn_bias is not None:
        if isinstance(attn_bias, xformers.ops.AttentionBias):
            tensor_bias = attn_bias.materialize(
                (query.shape[0], 1, query.shape[1], key.shape[1]),
                device=query.device,
                dtype=torch.float32,
            )
        else:
            assert isinstance(attn_bias, torch.Tensor)
            tensor_bias = attn_bias
        if tensor_bias.ndim == 4:
            tensor_bias = tensor_bias.reshape([-1, *tensor_bias.shape[2:]])
        attn = attn + tensor_bias.float()
    ref_lse = attn.logsumexp(-1)
    if isinstance(attn_bias, fmha.attn_bias.BlockDiagonalMask):
        lse = _block_diag_reshape_lse(lse, attn_bias.q_seqinfo)
    assert_allclose(lse[:, 0, : ref_lse.shape[1]], ref_lse, atol=2e-4)


@pytest.mark.parametrize("fmt", ["BMK", "BMHK"])
@pytest.mark.parametrize("grad_out_contiguous", [True])
@parametrize_opBW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv
def test_backward(
    opBW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv,
    grad_out_contiguous,
    fmt,
):
    (
        op_bw,
        device,
        dtype,
        bias_type,
        batch_size,
        q_len,
        kv_len,
        h,
        k,
        kv,
    ) = opBW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv

    if k > 128 or kv > 128:
        pytest.skip(
            "head-dim length bigger than 128 is not supported by CK-FlashAttention-1"
        )

    if k % 8 != 0 or kv % 8 != 0:
        pytest.skip("head-dim length must be an even value for CK-FlashAttention-1")

    # BottomRightMask requires generate {m0,m1,...}, {n0,n1,...} where mi <= ni
    if (
        bias_type is fmha.attn_bias.BlockDiagonalCausalFromBottomRightMask
        and q_len <= kv_len
    ):
        pytest.skip(
            "BlockDiagonalCausalFromBottomRightMask requires kv_len bigger than q_len"
        )

    if k != kv:
        pytest.skip("k same as kv is not well tested by CK-FlashAttention-1")

    # attn_bias_requires_grad = (
    #    random.Random(q_len + kv_len * batch_size).randint(0, 1) > 0
    # )
    attn_bias_requires_grad = False

    query, key, value, attn_bias = create_tensors(
        *opBW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv,
        attn_bias_requires_grad=attn_bias_requires_grad,
        fmt=fmt,
    )
    op_fw = (
        sample_random_supported_fw(
            fmha.Inputs(query=query, key=key, value=value, attn_bias=attn_bias),
            seed=q_len * kv + kv_len * k,
        )
        if op_bw != fmha.ck.BwOp
        else fmha.ck.FwOp
    )
    qkv = None

    if (
        fmt == "BMHK"
        and query.shape[3] == value.shape[3]
        and query.shape[1] == value.shape[1]
    ):
        qkv = torch.stack([query, key, value], 2)
        qkv.requires_grad_(True)
        # bm3hk -> 3 x bmhk
        query, key, value = xformers.ops.unbind(qkv, 2)
        assert not query.is_contiguous()

    query.requires_grad_(True)
    key.requires_grad_(True)
    value.requires_grad_(True)

    if not op_bw.supports(fmha.Inputs(query, key, value, attn_bias)):
        pytest.skip("inputs not supported")

    out = xformers.ops.memory_efficient_attention(
        query, key, value, attn_bias, op=(op_fw, op_bw)
    )

    grad_out = torch.ones_like(out)
    # if grad_out_contiguous is False:
    #    grad_out = torch.tensor([1.0], dtype=query.dtype, device=device)[
    #        None, None, :
    #    ].expand_as(out)

    out.backward(grad_out)

    if qkv is None and op_bw == fmha.ck.BwOp:
        assert query.stride() == query.grad.stride()

    grads = []
    if qkv is None:
        grads = [query.grad, key.grad, value.grad]
        query.grad = None
        key.grad = None
        value.grad = None
    else:
        grads = [qkv.grad]
        qkv.grad = None
    if attn_bias_requires_grad:
        attn_bias_grad = get_bias_grad(attn_bias, clear=True)
        if attn_bias_grad is not None:
            grads.append(attn_bias_grad)

    ref = ref_attention(query, key, value, attn_bias)
    ref.backward(grad_out)

    assert_allclose(
        out.float(),
        ref.float(),
        "fw pass",
        atol=op_fw.ERROR_ATOL[dtype],
        rtol=op_fw.ERROR_RTOL.get(dtype, 1e-5),
    )

    del out
    del grad_out
    del ref

    atol = op_bw.ERROR_ATOL[dtype]
    rtol = op_bw.ERROR_RTOL[dtype]

    grads_ref = []
    grads_name = []
    if qkv is None:
        assert isinstance(query.grad, torch.Tensor)
        assert isinstance(key.grad, torch.Tensor)
        assert isinstance(value.grad, torch.Tensor)
        grads_ref = [query.grad, key.grad, value.grad]
        grads_name = ["query", "key", "value"]
    else:
        assert isinstance(qkv.grad, torch.Tensor)
        grads_ref = [qkv.grad]
        grads_name = ["qkv"]

    if attn_bias_requires_grad:
        attn_bias_grad = get_bias_grad(attn_bias)
        if attn_bias_grad is not None:
            grads_ref.append(attn_bias.grad)
            grads_name.append("bias")

    del query
    del key
    del value
    del qkv

    assert len(grads_ref) == len(
        grads
    ), "Wrong number of gradients (maybe bias grad didn't backprop?)"
    for name, calc_grad, ref_grad in zip(grads_name, grads, grads_ref):
        assert_allclose(
            calc_grad,
            ref_grad,
            msg=f"{op_fw.NAME}+{op_bw.NAME}:{name}",
            atol=atol,
            rtol=rtol,
        )
