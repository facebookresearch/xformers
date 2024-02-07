# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import itertools
from functools import partial

import torch
from torch.utils import benchmark
from utils import benchmark_main_helper

import xformers.ops
import xformers.ops.fmha as fmha

torch.backends.cuda.matmul.allow_tf32 = False

# Run with
#  python xformers/benchmarks/benchmark_mem_eff_attn_decoder.py --omit-baselines --quiet
# The baselines for these benchmarks are really slow because there is
# so much padding in the inputs, so there is no point running them.


def ref_attention_bmk(q, k, v, attn_bias=None):
    if isinstance(attn_bias, xformers.ops.AttentionMask):
        attn_bias = (
            attn_bias.materialize((q.shape[0], 1, q.shape[1], k.shape[1]))
            .to(q)
            .squeeze()
        )
    q = q * (1.0 / q.shape[-1] ** 0.5)
    if attn_bias is None:
        attn = q @ k.transpose(-2, -1)
    else:
        # equivalent to (q @ k.transpose(-2, -1) + m).softmax(-1) @ v
        # but faster, and is what is used in PyTorch now
        attn = torch.baddbmm(attn_bias, q, k.transpose(-2, -1))
    attn = attn.softmax(-1)
    return attn @ v


def ref_attention(q, k, v, attn_bias):
    assert q.ndim == 4

    def T(t):
        return t.permute((0, 2, 1, 3)).reshape(
            [t.shape[0] * t.shape[2], t.shape[1], t.shape[3]]
        )

    out = ref_attention_bmk(T(q), T(k), T(v), attn_bias)
    out = out.reshape([q.shape[0], q.shape[2], q.shape[1], v.shape[3]])
    return out.permute((0, 2, 1, 3))


min_run_time = 0.5
device = torch.device("cuda")

NUM_THREADS = [1] if device.type == "cuda" else [1, 40]

OPS = [
    xformers.ops.fmha.cutlass.FwOp if torch.version.cuda else xformers.ops.fmha.ck.FwOp,
    (
        xformers.ops.fmha.decoder.FwOp
        if torch.version.cuda
        else xformers.ops.fmha.ck_decoder.FwOp
    ),
]

KV_SHAPES = [
    # list of n_keys, padding_length, batchsize
    (2, 64, 3),
    (32, 1024, 500),
    (1000, 1024, 2),
    (8000, 8192, 1),
    (240, 256, 32),
    (2048, 2 * 1024, 4),
    (4096 * 2, 8 * 1024, 1),
]

N_HEADS = [8, 16, 64]


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


CASES = list(
    product_dict(
        kv_shape=KV_SHAPES,
        n_heads=N_HEADS,
        num_threads=NUM_THREADS,
        multiquery=[True, False],
    )
)


def mem_eff_attention_decoder(
    kv_shape, n_heads: int, num_threads: int, multiquery: bool
):
    n_keys, padding, B = kv_shape
    torch.manual_seed(42)
    k_seqlen = torch.randint(1, n_keys + 1, (B,)).tolist()
    K = 128

    q = torch.rand(1, B, n_heads, K, device=device, dtype=torch.bfloat16)
    if multiquery:
        k = torch.rand(
            1, B * padding, 1, K, device=device, dtype=torch.bfloat16
        ).expand(1, B * padding, n_heads, K)
        v = torch.rand(
            1, B * padding, 1, K, device=device, dtype=torch.bfloat16
        ).expand(1, B * padding, n_heads, K)
    else:
        k = torch.rand(1, B * padding, n_heads, K, device=device, dtype=torch.bfloat16)
        v = torch.rand(1, B * padding, n_heads, K, device=device, dtype=torch.bfloat16)

    bias = fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
        q_seqlen=[1] * B,
        kv_seqlen=k_seqlen,
        kv_padding=padding,
    )

    sub_label = f"{B}batch-{k_seqlen[0]}keys-{n_heads}heads"
    if multiquery:
        sub_label += "-mq"

    has_run = False
    for fw_op in OPS:
        inp = fmha.Inputs(q, k, v, attn_bias=bias)
        if not fw_op.supports(inp):
            continue

        fn = partial(xformers.ops.memory_efficient_attention_forward, op=fw_op)

        yield benchmark.Timer(
            stmt="fn(q, k, v, attn_bias)",
            globals={
                "q": q,
                "k": k,
                "v": v,
                "attn_bias": bias,
                "fn": fn,
            },
            label="attention",
            description=fw_op.NAME,
            sub_label=sub_label,
            num_threads=num_threads,
        )

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn(q, k, v, bias)
        yield benchmark.Timer(
            stmt="graph.replay()",
            globals={
                "graph": graph,
            },
            label="cuda graphed attention",
            description=fw_op.NAME,
            sub_label=sub_label,
            num_threads=num_threads,
        )

        has_run = True

    if not has_run:
        return

    RUN_BASELINES = False
    if RUN_BASELINES:
        yield benchmark.Timer(
            stmt="fn(q, k, v, attn_bias)",
            globals={
                "q": q,
                "k": k,
                "v": v,
                "attn_bias": bias,
                "fn": ref_attention,
            },
            label="attention",
            description="eager",
            sub_label=sub_label,
            num_threads=num_threads,
        )


benchmark_main_helper(mem_eff_attention_decoder, CASES, min_run_time=min_run_time)
