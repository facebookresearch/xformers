# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections import namedtuple
from typing import Any, Dict, List, Optional

import torch
import triton

from xformers.benchmarks.utils import pretty_plot, pretty_print

TestCase = namedtuple("TestCase", ["function", "name"])

_gpu_is_old: Optional[bool] = None


def next_power_of_2(n):
    """Return the smallest power of 2 greater than or equal to n"""
    assert n < 2 ** 32, "Depths beyond 2^32 are not yet handled by this kernel"

    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


def bench_functions(
    test_cases: List[TestCase], shapes, metric_transform, unit, title=""
):
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

                results[key][testcase.name] = f"{metric:.1f}"

        pretty_print(
            results,
            title=" ------------- Type: {} ------------- ".format(dtype),
            units=unit,
        )
        _type = " fp16" if dtype == torch.float16 else " fp32"
        pretty_plot(results, title + _type, unit, dash_key="pytorch")


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


def assert_almost_equal(x, y, decimal=2, err_msg=""):
    import numpy.testing as npt

    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().detach().numpy()
    npt.assert_array_almost_equal(x, y, err_msg=err_msg, decimal=decimal)


@triton.jit
def minimum(x, y):
    """
    Computes the element-wise minimum of :code:`x` and :code:`y`.

    :param input: the first input block
    :type input: Block
    :param other: the second input block
    :type other: Block
    """
    return triton.language.where(x < y, x, y)


@triton.jit
def maximum(x, y):
    """
    Computes the element-wise maximum of :code:`x` and :code:`y`.

    :param input: the first input block
    :type input: Block
    :param other: the second input block
    :type other: Block
    """
    return triton.language.where(x > y, x, y)


@triton.jit
def swizzle2d(i, j, size_i, size_j, size_g):
    """
    transformes indices of a row-major size_i*size_j matrix into those
    of one where indices are row major for each group of size_j rows.
    For example, for size_i = size_j = 4 and size_g = 2, it will transform
    [[0 , 1 , 2 , 3 ],
     [4 , 5 , 6 , 7 ],
     [8 , 9 , 10, 11],
     [12, 13, 14, 15]]
    into
    [[0, 2,  4 , 6 ],
     [1, 3,  5 , 7 ],
     [8, 10, 12, 14],
     [9, 11, 13, 15]]
    """
    # "unrolled index in array"
    ij = i * size_j + j
    # number of elements in `size_g` groups
    # of `size_j` columns
    size_gj = size_g * size_j
    # index of the group in which (i,j) is
    group_id = ij // size_gj
    # row-index of the first element of this group
    off_i = group_id * size_g
    # last group may have fewer rows
    size_g = minimum(size_i - off_i, size_g)
    # new row and column indices
    new_i = off_i + (ij % size_g)
    new_j = (ij % size_gj) // size_g
    return new_i, new_j
