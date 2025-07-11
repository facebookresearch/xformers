# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import ast
import logging
import sys
from typing import Any, List

import pytest
import torch

import xformers

try:
    import triton
    import triton.language as tl

    from xformers.triton.vararg_kernel import (
        _VisitorConditionalKernel,
        unroll_varargs,
        VarargMode,
    )

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
@pytest.mark.parametrize("conditional", [True, False])
def test_triton_multiple_varargs_kernel(conditional: bool):
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
    if conditional:
        kernel = unroll_varargs(
            weighted_sumN, N=NUM_INPUTS, mode=VarargMode.CONDITIONAL
        )
    else:
        kernel = unroll_varargs(weighted_sumN, N=NUM_INPUTS)
    kernel[(1,)](output, *a, *b_list, BLOCK_SIZE=32)
    expected_output = (torch.stack(a) * torch.stack(b).unsqueeze(1)).sum(0)
    assert torch.allclose(expected_output, output)


@pytest.mark.skipif(not enable_tests, reason="moe not supported")
def test_triton_varargs_conditional():
    # to make linter happy
    VAR_ARGS_ARRAY = List[Any]

    @triton.jit
    def kernel(
        x_ptrs: "VAR_ARGS_ARRAY",  # noqa: F821
        y_ptrs: "VAR_ARGS_ARRAY",  # noqa: F821
        numel,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = BLOCK_SIZE * pid + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        for i in range(len(x_ptrs)):
            x_ptr = x_ptrs[i]
            y_ptr = y_ptrs[i]

            data = tl.load(x_ptr + offsets, mask)
            result = data * data
            tl.store(y_ptr + offsets, result, mask)

    k = triton.JITFunction(kernel.fn)
    parsed = ast.parse(k.src)
    visitor = _VisitorConditionalKernel(N=3)
    parsed = visitor.visit(parsed)
    parsed = ast.fix_missing_locations(parsed)
    new_src = ast.unparse(parsed)  # type: ignore

    assert "x_ptrs0, x_ptrs1, x_ptrs2" in new_src
    assert "y_ptrs0, y_ptrs1, y_ptrs2", new_src
    assert "for i in range(3):" in new_src
    assert "x_ptr = x_ptrs0 if i == 0 else x_ptrs1 if i == 1 else x_ptrs2" in new_src
    assert "y_ptr = y_ptrs0 if i == 0 else y_ptrs1 if i == 1 else y_ptrs2" in new_src


@pytest.mark.skipif(not enable_tests, reason="moe not supported")
def test_subscripting_call():
    @triton.jit
    def fused_group_contiguous_nan_clamp_copied_from_inductor(
        _group_a_ptrs: "VAR_ARGS_ARRAY",  # type: ignore # noqa: F821
        XBLOCK: tl.constexpr,
    ):
        xoffset = tl.program_id(0) * XBLOCK
        xoffset + tl.arange(0, XBLOCK)[:]

    unroll_varargs(
        fused_group_contiguous_nan_clamp_copied_from_inductor,
        N=2,
        mode=VarargMode.CONDITIONAL,
    )
