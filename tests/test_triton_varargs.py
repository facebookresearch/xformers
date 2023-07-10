# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys

import pytest
import torch

import xformers

try:
    import triton
    import triton.language as tl

    from xformers.triton.vararg_kernel import unroll_varargs

    _triton_available = xformers._is_triton_available()
except ImportError as e:
    logging.warning(
        f"Triton is not available, some optimizations will not be tested.\n{e}"
    )
    _triton_available = False

enable_tests = (
    (sys.version_info.major, sys.version_info.minor) >= (3, 9)
    and _triton_available
    and torch.cuda.is_available()
)


@pytest.mark.skipif(not enable_tests, reason="moe not supported")
def test_triton_varargs_kernel():
    @triton.jit
    def sumN(output_ptr, scaling_ptr, *inputs, BLOCK_SIZE: tl.constexpr):
        offset = tl.arange(0, BLOCK_SIZE)
        output = tl.zeros([BLOCK_SIZE], tl.float32)
        scaling: "VAR_ARGS_ARRAY"  # type: ignore # noqa: F821
        for i in range(len(scaling)):
            scaling[i] = tl.load(scaling_ptr + i)

        for i in range(2):
            for j in range(len(inputs)):
                output = output + tl.load(inputs[j] + offset) * scaling[j]
        tl.store(output_ptr + offset, output)

    BLOCK_SIZE = 32
    NUM_INPUTS = 2
    torch.manual_seed(0)
    inputs = [
        torch.randn([BLOCK_SIZE], dtype=torch.float32, device="cuda")
        for _ in range(NUM_INPUTS)
    ]
    output = torch.randn([BLOCK_SIZE], dtype=torch.float32, device="cuda")
    scaling = torch.randn([NUM_INPUTS, 1], dtype=torch.float32, device="cuda")
    sumN_unrolled = unroll_varargs(sumN, N=NUM_INPUTS)
    sumN_unrolled[(1,)](output, scaling, *inputs, BLOCK_SIZE=32)
    assert torch.allclose((2 * torch.stack(inputs) * scaling).sum(0), output)


@pytest.mark.skipif(not enable_tests, reason="moe not supported")
def test_triton_multiple_varargs_kernel():
    @triton.jit
    def weighted_sumN(
        output_ptr,
        a_ptr: "VAR_ARGS_ARRAY",  # type: ignore # noqa: F821
        b: "VAR_ARGS_ARRAY",  # type: ignore # noqa: F821
        BLOCK_SIZE: tl.constexpr,
    ):
        # Weighted sum, where the weights are on CPU
        offset = tl.arange(0, BLOCK_SIZE)
        output = tl.zeros([BLOCK_SIZE], tl.float32)

        for i in range(len(a_ptr)):
            output = output + tl.load(a_ptr[i] + offset) * b[i]
        tl.store(output_ptr + offset, output)

    BLOCK_SIZE = 32
    NUM_INPUTS = 2
    torch.manual_seed(0)
    a = [
        torch.randn([BLOCK_SIZE], dtype=torch.float32, device="cuda")
        for _ in range(NUM_INPUTS)
    ]
    b = [torch.randn([], dtype=torch.float32, device="cuda") for _ in range(NUM_INPUTS)]
    b_list = [x.item() for x in b]
    output = torch.randn([BLOCK_SIZE], dtype=torch.float32, device="cuda")
    kernel = unroll_varargs(weighted_sumN, N=NUM_INPUTS)
    kernel[(1,)](output, *a, *b_list, BLOCK_SIZE=32)
    expected_output = (torch.stack(a) * torch.stack(b).unsqueeze(1)).sum(0)
    assert torch.allclose(expected_output, output)
