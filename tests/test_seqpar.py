# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
from typing import Tuple

import pytest
import torch

from xformers.ops import (
    sequence_parallel_leading_matmul,
    sequence_parallel_trailing_matmul,
)

from .multiprocessing_utils import launch_subprocesses

compute_capability = (0, 0)
if torch.cuda.is_available():
    compute_capability = torch.cuda.get_device_capability("cuda")
cuda_sm80_only = pytest.mark.skipif(
    compute_capability < (8, 0), reason="requires sm70+"
)
at_least_2_gpus = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs at least 2 GPUs"
)


def reference_leading(input_, w1, w2):
    hidden1 = torch.matmul(input_, w1.t())
    hidden2 = torch.matmul(input_, w2.t())
    return [hidden1, hidden2]


def reference_trailing(hidden, w):
    output = torch.matmul(hidden, w.t())
    return output


def xformers_leading(input_, w1, w2, *, fuse, group):
    return sequence_parallel_leading_matmul(
        input_, [w1.t(), w2.t()], fuse=fuse, process_group=group
    )


def xformers_trailing(hidden, w, *, fuse, group):
    return sequence_parallel_trailing_matmul(
        hidden, w.t(), fuse=fuse, process_group=group
    )


def inner_seqpar(
    kind: str,
    step: str,
    dims: Tuple[int, ...],
    dtype: torch.dtype,
    compile: bool,
    seed: int,
):
    os.environ["TORCH_SYMM_MEM_ALLOW_OVERLAPPING_DEVICES"] = "1"

    my_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    subgroup = torch.distributed.new_group()

    fused = True
    if kind == "unfused":
        fused = False
    elif kind == "fallback":
        os.environ["DISABLE_FUSED_SEQUENCE_PARALLEL"] = "1"

    torch.random.manual_seed(seed)
    torch._dynamo.reset_code_caches()  # avoids hitting recompilation limit

    batch_dims = dims[:-2]
    outer_dim = dims[-2]
    inner_dim = dims[-1]

    # To check for correctness we want to compare the outputs but the accuracy
    # of matmuls, apparently, is not that great. We thus try to produce inputs
    # for which no rounding at all will occur. We do this by using zero or one
    # inputs, so their product will also be zero or one, and keep the reduction
    # dimension small enough so that they fit in the mantissa without overflow.
    max_exact_value = 2 * (1 / torch.finfo(dtype).eps)
    # 0.25 is the ratio of expected ones and we aim at 2/3 of the safe range
    assert outer_dim * 0.25 <= max_exact_value * 0.66
    assert inner_dim * world_size * 0.25 <= max_exact_value * 0.66

    def my_chunk(t, *, dim):
        return t.tensor_split(world_size, dim=dim)[my_rank]

    if step == "leading":
        input_ = torch.testing.make_tensor(
            batch_dims + (outer_dim,),
            dtype=dtype,
            device="cuda",
            low=0,
            high=1,
        ).round()
        weight1, weight2 = [
            torch.testing.make_tensor(
                (inner_dim * (idx + 1), outer_dim),
                dtype=dtype,
                device="cuda",
                low=0,
                high=1,
            ).round()
            for idx in range(2)
        ]
        gradient1, gradient2 = [
            torch.testing.make_tensor(
                batch_dims + (inner_dim * (idx + 1),),
                dtype=dtype,
                device="cuda",
                low=0,
                high=1,
            ).round()
            for idx in range(2)
        ]

        # Non-fused reference code
        input_ref = input_.detach().requires_grad_()
        weight1_ref = weight1.detach().requires_grad_()
        weight2_ref = weight2.detach().requires_grad_()

        output1_ref, output2_ref = reference_leading(
            input_ref, weight1_ref, weight2_ref
        )
        torch.autograd.backward([output1_ref, output2_ref], [gradient1, gradient2])

        my_output1_ref = my_chunk(output1_ref, dim=-1)
        my_output2_ref = my_chunk(output2_ref, dim=-1)
        my_weight1_grad_ref = my_chunk(weight1_ref.grad, dim=0)
        my_weight2_grad_ref = my_chunk(weight2_ref.grad, dim=0)
        my_input_grad_ref = my_chunk(input_ref.grad, dim=0)

        # Faster fused mode
        my_input_xf = my_chunk(input_, dim=0).detach().requires_grad_()
        my_weight1_xf = my_chunk(weight1, dim=0).detach().requires_grad_()
        my_weight2_xf = my_chunk(weight2, dim=0).detach().requires_grad_()
        my_gradient1 = my_chunk(gradient1, dim=-1)
        my_gradient2 = my_chunk(gradient2, dim=-1)

        my_output1_xf, my_output2_xf = torch.compile(
            xformers_leading, fullgraph=True, disable=not compile
        )(my_input_xf, my_weight1_xf, my_weight2_xf, fuse=fused, group=subgroup)
        torch.autograd.backward(
            [my_output1_xf, my_output2_xf], [my_gradient1, my_gradient2]
        )

        my_weight1_grad_xf = my_weight1_xf.grad
        my_weight2_grad_xf = my_weight2_xf.grad
        my_input_grad_xf = my_input_xf.grad

        # Checks
        torch.testing.assert_close(my_output1_ref, my_output1_xf)
        torch.testing.assert_close(my_output2_ref, my_output2_xf)
        torch.testing.assert_close(my_input_grad_ref, my_input_grad_xf)
        torch.testing.assert_close(my_weight1_grad_ref, my_weight1_grad_xf)
        torch.testing.assert_close(my_weight2_grad_ref, my_weight2_grad_xf)

        torch.library.opcheck(
            torch.ops.xformers_python.sequence_parallel_leading_matmul_fwd,
            (my_input_xf.flatten(0, -2), [my_weight1_xf.t(), my_weight2_xf.t()]),
            {"fuse": fused, "process_group_name": subgroup.group_name},
        )

    elif step == "trailing":
        input_ = torch.testing.make_tensor(
            batch_dims + (inner_dim,),
            dtype=dtype,
            device="cuda",
            low=0,
            high=1,
        ).round()
        weight = torch.testing.make_tensor(
            (outer_dim, inner_dim),
            dtype=dtype,
            device="cuda",
            low=0,
            high=1,
        ).round()
        gradient = torch.testing.make_tensor(
            batch_dims + (outer_dim,),
            dtype=dtype,
            device="cuda",
            low=0,
            high=1,
        ).round()

        # Non-fused reference code
        input_ref = input_.detach().requires_grad_()
        weight_ref = weight.detach().requires_grad_()

        output_ref = reference_trailing(input_ref, weight_ref)
        torch.autograd.backward([output_ref], [gradient])

        my_output_ref = my_chunk(output_ref, dim=0)
        my_weight_grad_ref = my_chunk(weight_ref.grad, dim=1)
        my_input_grad_ref = my_chunk(input_ref.grad, dim=-1)

        # Faster fused mode
        my_input_xf = my_chunk(input_, dim=-1).detach().clone().requires_grad_()
        my_weight_xf = my_chunk(weight, dim=1).detach().requires_grad_()
        my_gradient = my_chunk(gradient, dim=0)

        my_output_xf = torch.compile(
            xformers_trailing, fullgraph=True, disable=not compile
        )(my_input_xf, my_weight_xf, fuse=fused, group=subgroup)
        torch.autograd.backward([my_output_xf], [my_gradient])

        my_weight_grad_xf = my_weight_xf.grad
        my_input_grad_xf = my_input_xf.grad

        # Checks
        torch.testing.assert_close(my_output_ref, my_output_xf)
        torch.testing.assert_close(my_input_grad_ref, my_input_grad_xf)
        torch.testing.assert_close(my_weight_grad_ref, my_weight_grad_xf)

        torch.library.opcheck(
            torch.ops.xformers_python.sequence_parallel_trailing_matmul_fwd,
            (my_input_xf.flatten(0, -2), my_weight_xf.t()),
            {"fuse": fused, "process_group_name": subgroup.group_name},
        )


# PyTorch doesn't support pre-sm80 for its signaling kernels
# https://github.com/pytorch/pytorch/pull/146308
@cuda_sm80_only
@pytest.mark.parametrize(
    "kind",
    [
        "singleton",
        pytest.param("unfused", marks=at_least_2_gpus),
        pytest.param("fallback", marks=at_least_2_gpus),
        "fused",
    ],
)
@pytest.mark.parametrize(
    "step",
    [
        "leading",
        "trailing",
    ],
)
@pytest.mark.parametrize(
    "dims",
    [
        pytest.param((2, 2, 512, 512, 256), id="nice-shapes"),
        pytest.param((2, 1023, 511, 257), id="ugly-shapes"),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.bfloat16, id="bf16"),
        pytest.param(torch.float16, id="fp16"),
        pytest.param(torch.float32, id="fp32"),
    ],
)
@pytest.mark.parametrize(
    "compile", [pytest.param(False, id="eager"), pytest.param(True, id="compile")]
)
def test_seqpar(
    kind: str,
    step: str,
    dims: Tuple[int, ...],
    dtype: torch.dtype,
    compile: bool,
):
    if compile and dtype is torch.bfloat16 and compute_capability < (8, 0):
        # https://fb.workplace.com/groups/1075192433118967/posts/1480158559289017
        pytest.skip("Dynamo misbehaves on V100 or earlier when handling bf16")

    world_size = 1 if kind == "singleton" else 2
    seed = random.getrandbits(32)
    launch_subprocesses(
        world_size=world_size,
        fn=inner_seqpar,
        kind=kind,
        step=step,
        dims=dims,
        dtype=dtype,
        compile=compile,
        seed=seed,
    )
