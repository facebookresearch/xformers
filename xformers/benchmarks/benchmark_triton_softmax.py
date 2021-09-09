import torch

from tests.test_triton_softmax import SHAPES
from xformers.triton.softmax import log_softmax as triton_log_softmax
from xformers.triton.softmax import softmax as triton_softmax
from xformers.triton.utils import TestCase, bench_functions

MIN_RUN_TIME = 1


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
bench_functions(
    [
        TestCase(lambda x: torch.softmax(x, dim=-1), "pytorch - fw"),
        TestCase(triton_softmax, "triton  - fw"),
        TestCase(lambda x: torch.log_softmax(x, dim=-1), "pytorch - log - fw"),
        TestCase(triton_log_softmax, "triton  - log - fw"),
    ],
    SHAPES,
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
)
