# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import itertools
import pprint
from typing import Dict

import torch
from torch.utils import benchmark

import xformers.ops


def ref_attention(q, k, v, attn_bias=None, p=0.0):
    q = q * (1.0 / q.shape[-1] ** 0.5)
    attn = q @ k.transpose(-2, -1)
    if attn_bias is None:
        attn = q @ k.transpose(-2, -1)
    else:
        # equivalent to (q @ k.transpose(-2, -1) + m).softmax(-1) @ v
        # but faster, and is what is used in PyTorch now
        attn = torch.baddbmm(attn_bias, q, k.transpose(-2, -1))
    attn = attn.softmax(-1)
    if p > 0:
        attn = torch.nn.functional.dropout(attn, p=p)
    return attn @ v


min_run_time = 2
device = torch.device("cuda")

NUM_THREADS = [1] if device.type == "cuda" else [1, 40]
SHAPES = list(
    itertools.product([1, 8, 32, 256], [127, 128, 512, 513, 1023, 1024], [16, 32])
)

p = 0.0

results = []
mem_use: Dict[str, Dict[str, float]] = dict(optimized={}, vanilla={})


def benchmark_forward():
    print(f"Processing {len(SHAPES)} cases")
    print("Forward")
    for num_threads in NUM_THREADS:
        for shape in SHAPES:
            print(f"===== {shape} =====")
            for use_attn_bias in [False, True]:
                print(f"Use attention bias: {use_attn_bias}")

                B, M, K = shape
                q = torch.rand(shape, device=device)
                attn_bias = None
                if use_attn_bias:
                    attn_bias = torch.rand(shape[0], 1, shape[1], device=device).expand(
                        shape[0], shape[1], shape[1]
                    )
                sub_label = f"B={B}, M={M}, K={K}"

                if True:
                    r = xformers.ops.memory_efficient_attention(q, q, q, attn_bias)

                    rr = ref_attention(q, q, q, attn_bias)
                    assert (r - rr).abs().max() < 1e-5
                    del r, rr

                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                results.append(
                    benchmark.Timer(
                        stmt="fn(q, q, q, attn_bias, p)",
                        globals={
                            "q": q,
                            "attn_bias": attn_bias,
                            "p": p,
                            "fn": xformers.ops.memory_efficient_attention,
                        },
                        label=f"attention (use_attn_bias={use_attn_bias})",
                        description="optimized",
                        sub_label=sub_label,
                        num_threads=num_threads,
                    ).blocked_autorange(min_run_time=min_run_time)
                )
                torch.cuda.synchronize()
                memory = torch.cuda.max_memory_allocated() / 2**20
                mem_use["optimized"][sub_label] = memory
                memory_str = f"Memory used: {memory} MB"

                print("Optimized", memory_str)

                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                results.append(
                    benchmark.Timer(
                        stmt="fn(q, q, q, attn_bias, p)",
                        globals={
                            "q": q,
                            "attn_bias": attn_bias,
                            "p": p,
                            "fn": ref_attention,
                        },
                        label=f"attention (use_attn_bias={use_attn_bias})",
                        description="vanilla",
                        sub_label=sub_label,
                        num_threads=num_threads,
                    ).blocked_autorange(min_run_time=min_run_time)
                )

                torch.cuda.synchronize()
                memory = torch.cuda.max_memory_allocated() / 2**20
                mem_use["vanilla"][sub_label] = memory
                memory_str = f"Memory used: {memory} MB"
                print("Vanilla", memory_str)

    pprint.pprint(mem_use)


def benchmark_backward():
    print(f"Processing {len(SHAPES)} cases")
    print("Backward")
    for num_threads in NUM_THREADS:
        for shape in SHAPES:
            print(f"===== {shape} =====")
            for use_attn_bias in [False, True]:
                print(f"Use attention bias: {use_attn_bias}")
                B, M, K = shape
                q = torch.rand(shape, device=device, requires_grad=True)
                attn_bias = None
                if use_attn_bias:
                    attn_bias = torch.rand(shape[0], 1, shape[1], device=device).expand(
                        shape[0], shape[1], shape[1]
                    )
                sub_label = f"B={B}, M={M}, K={K}"

                if True:
                    r = xformers.ops.memory_efficient_attention(q, q, q, attn_bias)
                    r.backward(torch.ones_like(q))

                    grad = q.grad
                    q.grad = None

                    rr = ref_attention(q, q, q, attn_bias)
                    rr.backward(torch.ones_like(q))
                    assert (
                        grad - q.grad
                    ).abs().max() < 1e-5, f"{(grad - q.grad).abs().max()}"
                    q.grad = None
                    del r, rr, grad

                out = xformers.ops.memory_efficient_attention(q, q, q, attn_bias, p)
                grad = torch.ones_like(q)

                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                results.append(
                    benchmark.Timer(
                        stmt="out.backward(grad, retain_graph=True)",
                        globals={
                            "out": out,
                            "grad": grad,
                        },
                        label=f"attention backward (use_attn_bias={use_attn_bias})",
                        description="optimized",
                        sub_label=sub_label,
                        num_threads=num_threads,
                    ).blocked_autorange(min_run_time=min_run_time)
                )
                torch.cuda.synchronize()
                memory = torch.cuda.max_memory_allocated() / 2**20
                mem_use["optimized"][sub_label] = memory
                memory_str = f"Memory used: {memory} MB"

                print("Optimized", memory_str)

                out = ref_attention(q, q, q, attn_bias, p)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                results.append(
                    benchmark.Timer(
                        stmt="out.backward(grad, retain_graph=True)",
                        globals={
                            "out": out,
                            "grad": grad,
                        },
                        label=f"attention backward (use_attn_bias={use_attn_bias})",
                        description="vanilla",
                        sub_label=sub_label,
                        num_threads=num_threads,
                    ).blocked_autorange(min_run_time=min_run_time)
                )

                torch.cuda.synchronize()
                memory = torch.cuda.max_memory_allocated() / 2**20
                mem_use["vanilla"][sub_label] = memory
                memory_str = f"Memory used: {memory} MB"
                print("Vanilla", memory_str)

    pprint.pprint(mem_use)


benchmark_forward()
benchmark_backward()

compare = benchmark.Compare(results)
compare.print()
