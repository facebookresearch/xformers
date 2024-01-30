# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import os
import random
from typing import Tuple

import pytest
import torch

from xformers import _is_triton_available
from xformers.ops import fused_allgather_and_linear, fused_linear_and_reducescatter

from .multiprocessing_utils import launch_subprocesses

compute_capability = (0, 0)
if torch.cuda.is_available():
    compute_capability = torch.cuda.get_device_capability("cuda")
cuda_sm70_only = pytest.mark.skipif(
    compute_capability < (7, 0), reason="requires sm70+"
)
at_least_2_gpus = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs at least 2 GPUs"
)


# We care about correctness, not performance, hence let's "disable" the
# expensive autotuning by removing all configs except one (the first one).
if _is_triton_available():
    from xformers.ops._triton.sequence_parallel_fused_kernels import (
        _xformers_seqpar_matmul_kernel,
    )

    while len(_xformers_seqpar_matmul_kernel.configs) > 1:
        _xformers_seqpar_matmul_kernel.configs.pop()


def inner_sequence_parallel_fused(
    seed: int,
    kind: str,
    step: str,
    dims: Tuple[int, ...],
    dtype: torch.dtype,
):
    my_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    subgroup = torch.distributed.new_group()

    triton = True
    if kind == "fallback":
        os.environ["DISABLE_FUSED_SEQUENCE_PARALLEL"] = "1"
    elif kind == "pytorch":
        triton = False

    torch.random.manual_seed(seed)

    batch_dims = dims[:-2]
    subbatch_dims = (batch_dims[0] // world_size,) + batch_dims[1:]
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

    if step == "all-gather":
        inputs = torch.testing.make_tensor(
            (world_size,) + subbatch_dims + (outer_dim,),
            dtype=dtype,
            device="cuda",
            low=0,
            high=1,
        ).round()
        weight = torch.testing.make_tensor(
            (inner_dim, outer_dim), dtype=dtype, device="cuda", low=0, high=1
        ).round()

        # Non-fused reference code
        output_reference = torch.matmul(inputs, weight.t()).flatten(0, 1)

        # Faster fused mode
        output_fused = fused_allgather_and_linear(
            inputs[my_rank], weight, group=subgroup, _triton=triton
        )

    elif step == "reduce-scatter":
        inputs = torch.testing.make_tensor(
            (world_size,) + batch_dims + (inner_dim,),
            dtype=dtype,
            device="cuda",
            low=0,
            high=1,
        ).round()
        weights = torch.testing.make_tensor(
            (world_size, outer_dim, inner_dim),
            dtype=dtype,
            device="cuda",
            low=0,
            high=1,
        ).round()

        # Non-fused reference code
        staging = torch.empty(
            (world_size,) + subbatch_dims + (outer_dim,), dtype=dtype, device="cuda"
        )
        for rank in range(world_size):
            torch.matmul(
                inputs[rank].tensor_split(world_size, dim=0)[my_rank],
                weights[rank].t(),
                out=staging[rank],
            )
        output_reference = torch.sum(staging, dim=0, dtype=dtype)

        # Faster fused mode
        output_fused = fused_linear_and_reducescatter(
            inputs[my_rank], weights[my_rank], group=subgroup, _triton=triton
        )

    torch.testing.assert_close(output_reference, output_fused, atol=0, rtol=0)


@cuda_sm70_only
@pytest.mark.parametrize(
    "kind",
    ["singleton", pytest.param("fallback", marks=at_least_2_gpus), "pytorch", "triton"],
)
@pytest.mark.parametrize("step", ["all-gather", "reduce-scatter"])
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
def test_sequence_parallel_fused(
    kind: str,
    step: str,
    dims: Tuple[int, ...],
    dtype: torch.dtype,
):
    world_size = 1 if kind == "singleton" else 2
    seed = random.getrandbits(32)
    launch_subprocesses(
        world_size,
        inner_sequence_parallel_fused,
        seed=seed,
        kind=kind,
        step=step,
        dims=dims,
        dtype=dtype,
    )
