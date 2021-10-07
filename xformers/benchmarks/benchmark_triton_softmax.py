# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch

from xformers.triton.softmax import log_softmax as triton_log_softmax
from xformers.triton.softmax import softmax as triton_softmax
from xformers.triton.utils import TestCase, bench_functions

SHAPES = [
    (8, 384, 128),
    (8, 784, 512),
    (4, 2048, 384),
    (4, 3136, 1024),
    (2, 1024, 2048),
    (2, 2048, 4096),
    (2, 4096, 4096),
]


def pytorch_fw_bw(x):
    y = torch.norm(torch.softmax(x, dim=-1))
    y.backward()


def triton_fw_bw(x):
    y = torch.norm(triton_softmax(x))
    y.backward()


def pytorch_log_fw_bw(x):
    y = torch.norm(torch.log_softmax(x, dim=-1))
    y.backward()


def triton_log_fw_bw(x):
    y = torch.norm(triton_log_softmax(x))
    y.backward()


# Test FW
def to_gbs_fw(a, ms):
    # Read and write the full array
    return (2 * a.numel() * a.element_size() * 1e-9) / (ms * 1e-3)


def to_gbs_fwbw(a, ms):
    # same as above, but we do it twice (FW and then gradient)
    return 2 * to_gbs_fw(a, ms)


bench_functions(
    [
        TestCase(lambda x: torch.softmax(x, dim=-1), "pytorch - fw"),
        TestCase(triton_softmax, "triton  - fw"),
        TestCase(lambda x: torch.log_softmax(x, dim=-1), "pytorch - log - fw"),
        TestCase(triton_log_softmax, "triton  - log - fw"),
    ],
    SHAPES,
    to_gbs_fw,
    "GB/s",
    "Softmax_Bandwidth_FW",
)

# Test FW+BW
bench_functions(
    [
        TestCase(pytorch_fw_bw, "pytorch - fw+bw"),
        TestCase(triton_fw_bw, "triton  - fw+bw"),
        TestCase(pytorch_log_fw_bw, "pytorch - log - fw+bw"),
        TestCase(triton_log_fw_bw, "triton  - log - fw+bw"),
    ],
    SHAPES,
    to_gbs_fwbw,
    "GB/s",
    "Softmax_Bandwidth_FW_BW",
)
