import itertools
import pprint
from typing import Dict

import torch
from torch.utils import benchmark

import xformers.ops


def ref_attention(q, k, v):
    q = q * (1.0 / q.shape[-1] ** 0.5)
    return (q @ k.transpose(-2, -1)).softmax(-1) @ v


min_run_time = 2
device = torch.device("cuda")

NUM_THREADS = [1] if device.type == "cuda" else [1, 40]
SHAPES = list(
    itertools.product([1, 8, 32, 256], [127, 128, 512, 513, 1023, 1024], [16, 32])
)

results = []
mem_use: Dict[str, Dict[str, float]] = dict(optimized={}, vanilla={})

print(f"Processing {len(SHAPES)} cases")
for num_threads in NUM_THREADS:
    for shape in SHAPES:
        print(f"===== {shape} =====")
        B, M, K = shape
        q = torch.rand(shape, device=device)
        sub_label = f"B={B}, M={M}, K={K}"

        if True:
            r = xformers.ops.memory_efficient_attention(q, q, q)

            rr = ref_attention(q, q, q)
            assert (r - rr).abs().max() < 1e-5

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        results.append(
            benchmark.Timer(
                stmt="fn(q, q, q)",
                globals={
                    "q": q,
                    "fn": torch.ops.xformers.efficient_attention,
                },
                label="attention",
                description="optimized",
                sub_label=sub_label,
                num_threads=num_threads,
            ).blocked_autorange(min_run_time=min_run_time)
        )
        torch.cuda.synchronize()
        memory = torch.cuda.max_memory_allocated() / 2 ** 20
        mem_use["optimized"][sub_label] = memory
        memory = f"Memory used: {memory} MB"

        print("Optimized", memory)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        results.append(
            benchmark.Timer(
                stmt="fn(q, q, q)",
                globals={
                    "q": q,
                    "fn": ref_attention,
                },
                label="attention",
                description="vanilla",
                sub_label=sub_label,
                num_threads=num_threads,
            ).blocked_autorange(min_run_time=min_run_time)
        )

        torch.cuda.synchronize()
        memory = torch.cuda.max_memory_allocated() / 2 ** 20
        mem_use["vanilla"][sub_label] = memory
        memory = f"Memory used: {memory} MB"
        print("Vanilla", memory)


compare = benchmark.Compare(results)
compare.print()

pprint.pprint(mem_use)
