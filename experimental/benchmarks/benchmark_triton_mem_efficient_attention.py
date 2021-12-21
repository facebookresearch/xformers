# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import Any, Dict

import torch
import triton
from mem_efficient_attention import mem_efficient_attention

from xformers.benchmarks.utils import TestCase, pretty_plot, pretty_print

SHAPES = [
    (8, 256, 512),
    (8, 512, 1024),
    (4, 1024, 1024),
    # (2, 2048, 2048),
    # (2, 4096, 4096),
    # (1, 2048, 12288),
]


def attention_pytorch(q, k, v):
    # attention matrix
    q = q / math.sqrt(q.size(-1))
    a = q @ k.transpose(-2, -1)

    # softmax
    a = torch.softmax(a, dim=-1)

    # retrieval
    return a @ v


def to_flops_fw(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ms,
):
    # q @ kt
    flop = q.shape[0] * q.shape[1] * k.shape[1] * (2 * q.shape[2] - 1)

    # normalization
    att_shape = q.shape[1] * k.shape[1]
    flop += 5 * att_shape  # max + substraction + exp + sum + /

    # exp(q @ kt) @ v
    flop += v.shape[0] * att_shape * v.shape[1] * 2

    return flop * 1e-12 / (ms * 1e-3)


def bench_mem_efficient_attention():
    device = torch.device("cuda")
    backward = False

    for dtype in [
        torch.float16,
        torch.float32,
    ]:
        results: Dict[str, Any] = {}

        for B, M, K in SHAPES:
            k = torch.rand(B, M, K, device=device, dtype=dtype, requires_grad=backward)
            q = torch.rand(B, M, K, device=device, dtype=dtype, requires_grad=backward)
            v = torch.rand(B, M, K, device=device, dtype=dtype, requires_grad=backward)

            def torch_step(x, y, z):
                a = attention_pytorch(x, y, z)
                if backward:
                    torch.norm(a).backward()
                return a

            def triton_step(x, y, z):
                a = mem_efficient_attention.apply(x, y, z)
                if backward:
                    torch.norm(a).backward()
                return a

            for testcase in [
                TestCase(
                    torch_step,
                    "pytorch - fw{}".format("+bw" if backward else ""),
                ),
                TestCase(
                    triton_step,
                    "triton - fw{}".format("+bw" if backward else ""),
                ),
            ]:
                time = triton.testing.do_bench(lambda: testcase.function(q, k, v))[0]
                key = f"B={B}, M={M}, K={K}"
                if key not in results:
                    results[key] = {}

                # Record BW
                bandwidth = to_flops_fw(q, k, v, time)
                results[key][testcase.name] = f"{bandwidth:.1f}"

        units = "TFlops/s"
        pretty_print(results, title="\n --- Type: {} --- ".format(dtype), units=units)
        pretty_plot(
            results,
            title="LayerNorm-FW-{} - {}".format("+BW" if backward else "", dtype),
            units=units,
            dash_key="pytorch",
        )


if __name__ == "__main__":
    bench_mem_efficient_attention()
