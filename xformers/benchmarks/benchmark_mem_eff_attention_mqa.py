# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import itertools
import random
from functools import partial

import torch
from torch.utils import benchmark

import xformers.ops
import xformers.ops.fmha as fmha
from xformers.attn_bias_utils import create_attn_bias
from xformers.benchmarks.utils import benchmark_main_helper

torch.backends.cuda.matmul.allow_tf32 = False


# this interface assumes the tensor is in BMHK, but q and k/v might has different number of heads
def ref_attention_mqa(
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


# ref_attention_bmhk is completely the same as used by test_forward_ck_tiled.py
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
    out = ref_attention_mqa(T(q), T(k), T(v), attn_bias, scale=scale, dtype=dtype)
    out = out.reshape([q.shape[0], q.shape[2], q.shape[1], v.shape[3]])
    return out.permute((0, 2, 1, 3))


min_run_time = 0.5
device = torch.device("cuda")

NUM_THREADS = [1] if device.type == "cuda" else [1, 40]
SHAPES = [
    (1, 512, 512, 64, 8, 128),
    (1, 1024, 1024, 64, 8, 128),
    (1, 2048, 2048, 64, 8, 128),
    (1, 4096, 4096, 64, 8, 128),
    (1, 8192, 8192, 64, 8, 128),
    (1, 16384, 16384, 64, 8, 128),
    (1, 1024, 1024, 64, 8, 64),
    (1, 1024, 1024, 8, 1, 64),
    (1, 1024, 1024, 4, 4, 64),
    # *sorted(itertools.product([1, 2], [2048, 4096], [2048, 4096], [4, 8], [1, 2], [128])),
    # *sorted(
    #    itertools.product([16], [128, 512], [512, 1024], [16], [2, 4], [64, 128])
    # ),
]

OPS = [
    xformers.ops.fmha.ck.FwOp,
    xformers.ops.fmha.flash.FwOp,
]


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


CASES = list(
    product_dict(
        shape=SHAPES,
        num_threads=NUM_THREADS,
        dropout_p=[0.0],
        attn_bias_type=[type(None)],
        dtype=[torch.half, torch.bfloat16],
    )
)

# Add more cases with some variations
for c in CASES.copy():
    c = c.copy()
    c.update(
        random.Random(str(c["shape"])).choice(
            [
                {"attn_bias_type": torch.Tensor},
                {"attn_bias_type": xformers.ops.LowerTriangularMask},
            ]
        )
    )
    CASES.append(c)


def create_tensors(shape, dtype, requires_grad=False):
    B, M, N, Hq, Hkv, K = shape
    q = torch.rand(
        [B, M, Hq, K], device=device, dtype=dtype, requires_grad=requires_grad
    )
    k = torch.rand(
        [B, N, Hkv, K], device=device, dtype=dtype, requires_grad=requires_grad
    )
    v = torch.rand(
        [B, N, Hkv, K], device=device, dtype=dtype, requires_grad=requires_grad
    )
    return q, k, v


def mem_eff_attention_fw(shape, num_threads: int, attn_bias_type, dropout_p, dtype):
    B, M, N, Hq, Hkv, K = shape
    nhead_ratio_qk = Hq // Hkv
    q, k, v = create_tensors(shape, dtype)
    bias = create_attn_bias(
        attn_bias_type,
        batch_size=B,
        num_heads=Hq,
        num_heads_groups=nhead_ratio_qk,
        q_len=M,
        kv_len=N,
        device=device,
        dtype=dtype,
        requires_grad=False,
        fmt="BMHK",
        op=fmha.ck.FwOp,  # only required as a refer op by create_attn_bias
    )
    inp = fmha.Inputs(query=q, key=k, value=v, attn_bias=bias, p=dropout_p)

    dtype_str = {
        torch.bfloat16: "b16",
        torch.half: "f16",
        torch.float: "f32",
    }[dtype]
    sub_label = (
        f"{dtype_str} {B}-{M}-{N}-{Hq}-{Hkv}-{K}, p={dropout_p}, "
        f"BiasT={attn_bias_type.__name__}"
    )

    has_run = False
    for fw_op in OPS:
        if not fw_op.supports(inp):
            continue

        yield benchmark.Timer(
            stmt="fn(q, k, v, attn_bias, p)",
            globals={
                "q": q,
                "k": k,
                "v": v,
                "attn_bias": inp.attn_bias,
                "p": dropout_p,
                "fn": partial(
                    xformers.ops.memory_efficient_attention_forward, op=fw_op
                ),
            },
            label=f"attention (attn_bias={attn_bias_type})",
            description=fw_op.NAME,
            sub_label=sub_label,
            num_threads=num_threads,
        )
        has_run = True

    if not has_run:
        return

    yield benchmark.Timer(
        stmt="fn(q, k, v, attn_bias, p)",
        globals={
            "q": q,
            "k": k,
            "v": v,
            "attn_bias": inp.attn_bias,
            "p": dropout_p,
            "fn": ref_attention_mqa,
        },
        label=f"attention (attn_bias={attn_bias_type})",
        description="eager",
        sub_label=sub_label,
        num_threads=num_threads,
    )


benchmark_main_helper(mem_eff_attention_fw, CASES, min_run_time=min_run_time)
