import torch

from tests.test_triton_softmax import SHAPES
from xformers.triton.softmax import softmax as triton_softmax
from xformers.triton.utils import TestCase, bench_functions

MIN_RUN_TIME = 1


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
    ],
    SHAPES,
)

# Test FW+BW
bench_functions(
    [
        TestCase(pytorch_fw_bw, "pytorch - fw+bw"),
        TestCase(triton_fw_bw, "triton - fw+bw"),
    ],
    SHAPES,
)
