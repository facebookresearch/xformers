# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections import namedtuple
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import seaborn as sns
import torch

sns.set()

TestCase = namedtuple("TestCase", ["function", "name"])


_triton_is_available = torch.cuda.is_available()
if _triton_is_available:
    try:
        import triton
    except ImportError as e:
        logging.warning(f"Triton is not available: {e}.\nbench_functions")
        _triton_is_available = False


def pretty_print(results, title, units):
    """ Printout the contents of a dict as a human-readable and Markdown compatible array"""
    print(title)
    header = " Units: {:<45}".format(units)
    print("| " + header + "|" + "".join("{0:<20}|".format(k) for k in results.keys()))

    offset = len(header)
    print(
        "|-{}|".format("-" * offset)
        + "".join("{}|".format("-" * 20) for _ in results.keys())
    )

    workloads: Dict[str, Any] = {k: [] for v in results.values() for k in v.keys()}
    for v in results.values():
        for k in v.keys():
            workloads[k].append(v[k])

    for k, w in workloads.items():
        print(
            "| {0:<{offset}}|".format(k, offset=offset)
            + "".join("{:<20}|".format(v) for v in w)
        )

    print("")


def pretty_plot(results, title, units: str, filename=None, dash_key=""):
    """Graph out the contents of a dict.
    Dash key means that if the result label has this key, then it will be displayed with a dash"""

    if not filename:
        filename = title + ".png"

    # Sanitize the filename
    filename = (
        filename.replace(" ", "_").replace("/", "_").replace("-", "_").replace(":", "")
    )

    # Gather all the results in "collumns"
    workloads: Dict[str, Any] = {k: [] for v in results.values() for k in v.keys()}
    for v in results.values():
        for k in v.keys():
            workloads[k].append(float(v[k]))

    # Make sure that the plot is big enough
    f = plt.figure()
    f.set_figwidth(6)
    f.set_figheight(6)

    # Display the collections
    for k, v in workloads.items():
        if dash_key and dash_key in k:
            plt.plot(list(results.keys()), v, "--")
        else:
            plt.plot(list(results.keys()), v)

    plt.title(title)
    plt.legend(list(workloads.keys()), loc="lower right")
    plt.ylabel(units)
    plt.xticks(rotation=45)

    plt.savefig(filename, bbox_inches="tight")
    plt.close(f)


if _triton_is_available:

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
