# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List

import torch
import triton

from xformers.benchmarks.utils import TestCase, pretty_plot, pretty_print
from xformers.triton.sum_strided import sum_2d_dim_0

SHAPES = [
    (128, 128),
    (384, 128),
    (784, 512),
    (1024, 768),
    (2048, 1024),
    (4096, 4096),
]


def to_gbs(a, ms):
    # Read the full array, write the non-reduced dimension
    return ((a.numel() + a.shape[1]) * a.element_size() * 1e-9) / (ms * 1e-3)


def bench_functions(
    test_cases: List[TestCase], shapes, metric_transform, unit, title=""
):
    device = torch.device("cuda")

    for dtype in [torch.float16, torch.float32]:
        results: Dict[str, Any] = {}

        for M, N in shapes:
            a = torch.rand(M, N, device=device, dtype=dtype, requires_grad=True)

            for testcase in test_cases:
                time = triton.testing.do_bench(lambda: testcase.function(a))[0]

                metric = metric_transform(a, time)

                key = f"M={M}, N={N}"
                if key not in results:
                    results[key] = {}

                results[key][testcase.name] = f"{metric:.1f}"

        _type = " fp16" if dtype == torch.float16 else " fp32"

        pretty_print(
            results,
            title=" ------------- Type: {} ------------- ".format(_type),
            units=unit,
        )

        pretty_plot(results, title + _type, unit, dash_key="pytorch")


bench_functions(
    [
        TestCase(lambda x: torch.sum(x, dim=0), "pytorch"),
        TestCase(sum_2d_dim_0, "triton"),
    ],
    SHAPES,
    to_gbs,
    "GB/s",
    "Strided_sum",
)
