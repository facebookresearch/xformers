# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Sequence, Type, TypeVar

import pytest
import torch

import xformers.ops
from xformers.attn_bias_utils import create_attn_bias
from xformers.ops import fmha
from xformers.ops.common import get_xformers_operator

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

# ck_check_op is temporarily used to check ck-tiled availability
ck_check_op = get_xformers_operator("is_ck_tiled_used")
use_ck_tiled = ck_check_op()


def ref_attention(
    q, k, v, attn_bias=None, drop_mask=None, p=0.0, scale=None, dtype=None
):
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


@pytest.mark.parametrize("hdim_k,hdim_v", [(64, 64), (128, 128)])
@pytest.mark.parametrize("nhead_q,nhead_kv", [(8, 1), (8, 2), (12, 4), (4, 4)])
@pytest.mark.parametrize("seqlen_q,seqlen_kv", [(100, 128), (128, 100), (200, 1000)])
@pytest.mark.parametrize("batches", [100, 64, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "attn_bias_type", [type(None), torch.Tensor, fmha.attn_bias.LowerTriangularMask]
)
@pytest.mark.parametrize("op", ALL_FW_OPS)
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
    nhead_ratio_qk = Hq // Hkv

    device = torch.device("cuda")

    if not use_ck_tiled:
        pytest.skip("mqa/gqa is only supported with ck-tiled")

    torch.manual_seed(B * M + N * K + Hq * Hkv + Kv)

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
            num_heads_groups=nhead_ratio_qk,
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
        assert False, err_msg

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
