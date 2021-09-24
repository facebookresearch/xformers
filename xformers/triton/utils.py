# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections import namedtuple
from typing import Any, Dict, List, Optional

import torch
import triton

TestCase = namedtuple("TestCase", ["function", "name"])

_gpu_is_old: Optional[bool] = None


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
    print("{0:<45}".format("") + "".join("{0:<20} ".format(k) for k in results.keys()))

    workloads: Dict[str, Any] = {k: [] for v in results.values() for k in v.keys()}
    for v in results.values():
        for k in v.keys():
            workloads[k].append(v[k])

    for k, w in workloads.items():
        print("{0:<45}".format(k) + "".join("{:<20} ".format(v) for v in w))

    print("")


def gpu_capabilities_older_than_70() -> bool:
    """Return True if the GPU's compute capability is older than SM70."""
    global _gpu_is_old
    if _gpu_is_old is None:
        for i in range(torch.cuda.device_count()):
            major, _ = torch.cuda.get_device_capability(f"cuda:{i}")
            if major < 7:
                _gpu_is_old = True
        if _gpu_is_old is None:
            _gpu_is_old = False
    return _gpu_is_old


SUPPORTED_CUDA_DEVICES = ["V100", "A100"]


def get_current_cuda_device():
    current_device = str(torch.cuda.get_device_properties(torch.cuda.current_device()))
    for device_str in SUPPORTED_CUDA_DEVICES:
        if current_device.find(device_str) > 0:
            return device_str

    logging.warning("Unsupported device, Triton code generation may fail")
    return "P100"  # default to an old GPU
