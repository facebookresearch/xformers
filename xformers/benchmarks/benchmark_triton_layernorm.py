# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Dict

import torch
import triton

from xformers.benchmarks.utils import TestCase, pretty_plot, pretty_print
from xformers.triton import FusedLayerNorm

SHAPES = [
    (8, 256, 512),
    (8, 512, 1024),
    (4, 1024, 1024),
    (2, 2048, 2048),
    (2, 4096, 4096),
    (1, 2048, 12288),
]


def to_gbs_fw(a, ms):
    # Read and write the full array
    return (2 * a.numel() * a.element_size() * 1e-9) / (ms * 1e-3)


def bench_layernorm(backward: bool):
    device = torch.device("cuda")

    for dtype in [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]:
        results: Dict[str, Any] = {}

        for B, M, K in SHAPES:
            a = torch.rand(B, M, K, device=device, dtype=dtype, requires_grad=backward)

            # Pytorch layer norn
            torch_layernorm = torch.nn.LayerNorm([K]).to(dtype=dtype, device=device)

            # pyre-ignore[16]: TODO(T101400990): Pyre did not recognize the
            # `FusedLinearNorm` import.
            # Fused layernorm equivalent
            fused_layernorm = FusedLayerNorm([K]).to(dtype=dtype, device=device)

            def torch_step(x):
                y = torch_layernorm(x)
                if backward:
                    torch.norm(y).backward()
                return y

            def triton_step(x):
                y = fused_layernorm(x)
                if backward:
                    torch.norm(y).backward()
                return y

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
                time = triton.testing.do_bench(lambda: testcase.function(a))[0]
                key = f"B={B}, M={M}, K={K}"
                if key not in results:
                    results[key] = {}

                # Record BW
                bandwidth = to_gbs_fw(a, time)
                results[key][testcase.name] = f"{bandwidth:.1f}"

        pretty_print(results, title="\n --- Type: {} --- ".format(dtype), units="GB/s")
        pretty_plot(
            results,
            title="LayerNorm-FW{}-{}".format("+BW" if backward else "", dtype),
            units="GB/s",
            dash_key="pytorch",
        )


for bw in [False, True]:
    bench_layernorm(bw)
