# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import itertools

import torch
from torch.utils import benchmark
from triton.ops.matmul import matmul as triton_matmul

from xformers.benchmarks.utils import DTYPE2STR, benchmark_main_helper
from xformers.ops.tiled_matmul import tiled_matmul

min_run_time = 5


SHAPES = {
    "llama1_65b_mha_fwd": ([16384], [1024] * 3, [8192]),
    "llama1_65b_mha_bwd_input": ([16384], [8192], [1024] * 3),
    "llama1_65b_mha_bwd_weight": ([8192], [1024] * 3, [16384]),
    "llama1_65b_ffn_fwd": ([16384], [2752] * 2, [8192]),
    "llama1_65b_ffn_bwd_input": ([16384], [8192], [2752] * 2),
    "llama1_65b_ffn_bwd_weight": ([8192], [2752] * 2, [16384]),
    "llama2_150b_mha_fwd": ([16384], [1536, 128, 128], [12288]),
    "llama2_150b_mha_bwd_input": ([16384], [12288], [1536, 128, 128]),
    "llama2_150b_mha_bwd_weight": ([12288], [1536, 128, 128], [16384]),
    "llama2_150b_ffn_fwd": ([16384], [4096] * 2, [12288]),
    "llama2_150b_ffn_bwd_input": ([16384], [12288], [4096] * 2),
    "llama2_150b_ffn_bwd_weight": ([12288], [4096] * 2, [16384]),
}


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


CASES = list(
    product_dict(
        shape_name=SHAPES.keys(),
        dtype=[
            # torch.float32,
            torch.bfloat16,
            # torch.float16,
        ],
    )
)


def matmul_per_tile(a, b):
    c = []
    for n in range(len(a)):
        c.append([])
        for m in range(len(b[0])):
            c[-1].append(
                sum([torch.matmul(a[n][k], b[k][m]) for k in range(len(a[0]))])
            )
    return c


def benchmark_tiled_matmul(shape_name, dtype):
    ms, ns, ks = SHAPES[shape_name]
    m, n, k = sum(ms), sum(ns), sum(ks)

    a = torch.randn((m, k), device="cuda", dtype=dtype)
    b = torch.randn((k, n), device="cuda", dtype=dtype)

    a_tiles = [[y.clone() for y in x.split(ks, dim=1)] for x in a.split(ms, dim=0)]
    b_tiles = [[y.clone() for y in x.split(ns, dim=1)] for x in b.split(ks, dim=0)]

    dtype_str = DTYPE2STR.get(dtype, dtype)
    sub_label = (
        f"{dtype_str} {shape_name} "
        f"M={'+'.join(f'{m}' for m in ms)} "
        f"N={'+'.join(f'{n}' for n in ns)} "
        f"K={'+'.join(f'{k}' for k in ks)}"
    )

    # Warmup (maybe not needed?)
    torch.mm(a, b)
    matmul_per_tile(a_tiles, b_tiles)
    triton_matmul(a, b)
    tiled_matmul(a_tiles, b_tiles)

    yield benchmark.Timer(
        stmt="fn(a, b)",
        globals={
            "a": a,
            "b": b,
            "fn": torch.mm,
        },
        label="tiled_matmul",
        description="pytorch_fused",
        sub_label=sub_label,
    )
    yield benchmark.Timer(
        stmt="fn(a, b)",
        globals={
            "a": a_tiles,
            "b": b_tiles,
            "fn": matmul_per_tile,
        },
        label="tiled_matmul",
        description="pytorch_tiled",
        sub_label=sub_label,
    )
    yield benchmark.Timer(
        stmt="fn(a, b)",
        globals={
            "a": a,
            "b": b,
            "fn": triton_matmul,
        },
        label="tiled_matmul",
        description="triton_fused",
        sub_label=sub_label,
    )
    yield benchmark.Timer(
        stmt="fn(a, b)",
        globals={
            "a": a_tiles,
            "b": b_tiles,
            "fn": tiled_matmul,
        },
        label="tiled_matmul",
        description="xformers_tiled",
        sub_label=sub_label,
    )


benchmark_main_helper(benchmark_tiled_matmul, CASES, min_run_time=min_run_time)
