# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from collections import namedtuple
from typing import Any, Dict, List

import torch
import triton

TestCase = namedtuple("TestCase", ["function", "name"])


def next_power_of_2(n):
    """Return the smallest power of 2 greater than or equal to n"""
    assert n < 2 ** 16, "Depths beyond 2^16 are not yet handled by this softmax kernel"

    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1
    return n


def bench_functions(test_cases: List[TestCase], shapes, metric_transform, unit):
    device = torch.device("cuda")

    for dtype in [torch.float16, torch.float32]:
        results: Dict[str, Any] = {}

        for B, M, K in shapes:
            a = torch.rand(B, M, K, device=device, dtype=dtype, requires_grad=True)

            for testcase in test_cases:
                time = triton.testing.do_bench(lambda: testcase.function(a))[0]

                metric = metric_transform(a, time)

                key = f"B={B}, M={M}, K={K}"
                if key not in results:
                    results[key] = {}

                results[key][testcase.name] = f"{metric:.1f} {unit}"

        pretty_print(
            results, title=" ------------- Type: {} ------------- ".format(dtype)
        )


def pretty_print(results, title):
    print(title)
    print("{0:<40}".format("") + "".join("{0:<20} ".format(k) for k in results.keys()))

    workloads: Dict[str, Any] = {k: [] for v in results.values() for k in v.keys()}
    for v in results.values():
        for k in v.keys():
            workloads[k].append(v[k])

    for k, w in workloads.items():
        print("{0:<40}".format(k) + "".join("{:<20} ".format(v) for v in w))

    print("")
