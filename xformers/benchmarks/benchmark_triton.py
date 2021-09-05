from collections import namedtuple
from typing import Any, Dict, List

import torch
import triton

from tests.test_triton import SHAPES
from xformers.triton.softmax import softmax as triton_softmax

MIN_RUN_TIME = 1

TestCase = namedtuple("TestCase", ["function", "name"])


def bench_functions(test_cases: List[TestCase]):
    device = torch.device("cuda")

    for dtype in [torch.float16, torch.float32]:
        results: Dict[str, Any] = {}

        for B, M, K in SHAPES:
            a = torch.rand(B, M, K, device=device, dtype=dtype, requires_grad=True)

            for testcase in test_cases:
                time = triton.testing.do_bench(lambda: testcase.function(a))[0]
                key = f"B={B}, M={M}, K={K}"
                if key not in results:
                    results[key] = {}

                results[key][testcase.name] = f"{time:.2f}"

        # Pretty print
        print("Type: {}".format(dtype))
        print(
            "{0:<20}".format("") + "".join("{0:<20} ".format(k) for k in results.keys())
        )

        workloads: Dict[str, Any] = {k: [] for v in results.values() for k in v.keys()}
        for v in results.values():
            for k in v.keys():
                workloads[k].append(v[k])

        for k, w in workloads.items():
            print("{0:<20}".format(k) + "".join("{:<20} ".format(v) for v in w))

        print("")


def pytorch_fw_bw(x):
    y = torch.norm(torch.softmax(x, dim=-1))
    y.backward()


def triton_fw_bw(x):
    y = torch.norm(triton_softmax(x))
    y.backward()


# Test FW
bench_functions(
    [
        TestCase(lambda x: torch.softmax(x, dim=-1), "pytorch - fw"),
        TestCase(triton_softmax, "triton - fw"),
    ]
)

# Test FW+BW
bench_functions(
    [
        TestCase(pytorch_fw_bw, "pytorch - fw+bw"),
        TestCase(triton_fw_bw, "triton - fw+bw"),
    ]
)
