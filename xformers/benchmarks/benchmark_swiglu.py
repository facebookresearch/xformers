# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import itertools
from contextlib import nullcontext
from functools import partial
from typing import Any

import torch
from torch.utils import benchmark

import xformers.ops.swiglu_op as xsw
from xformers.benchmarks.utils import benchmark_main_helper

min_run_time = 0.5
device = torch.device("cuda")

SHAPES = [
    # Format: [inp.shape[0], inp.shape[1], hidden.shape[1]]
    # ViT-Giant
    (9456, 1536, 2736),
    (4440, 1536, 2736),
    (4728, 1536, 2736),
    # Some smaller shapes as well
    (4728, 1536, 1024),
    # GPT-3 (small)
    (32768, 2048, 5632),
    # Chinchilla
    (32768, 8192, 22016),
]


# OP = xsw._SwiGLUDecomposedOp
# OP = xsw.SwiGLUFusedOp
OP = xsw.SwiGLUPackedFusedOp


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


CASES = list(
    product_dict(
        shape=SHAPES,
        dtype=[torch.bfloat16, torch.half, "autocast_half"],
        bias=[True, False],
    )
)

DTYPE2STR = {
    torch.bfloat16: "b16   ",
    torch.half: "f16   ",
    "autocast_half": "f16.ac",
}


def benchmark_swiglu(shape, dtype, bias: bool):
    if dtype == "autocast_half":
        inp_dtype, model_dtype, autocast = torch.float, torch.float, True
    else:
        inp_dtype, model_dtype, autocast = dtype, dtype, False

    x = torch.randn(shape[:2], device=device, dtype=inp_dtype)
    module = (
        xsw.SwiGLU(in_features=shape[1], hidden_features=shape[2], bias=bias)
        .to(device)
        .to(model_dtype)
    )

    dtype_str = DTYPE2STR.get(dtype, dtype)
    bstr = "bias" if bias else "nobi"
    sub_label = f"{dtype_str} B={shape[0]}, I={shape[1]}, H={shape[2]} {bstr}"

    params = module._ordered_params()

    PREFIX = 'with torch.autocast("cuda", dtype=torch.half):\n    ' if autocast else ""
    yield benchmark.Timer(
        stmt=f"{PREFIX}fn(x, *args)",
        globals={
            "x": x,
            "args": params,
            "fn": partial(xsw.swiglu, op=OP),
        },
        label="swiglu_fw",
        description=OP.NAME,
        sub_label=sub_label,
    )
    yield benchmark.Timer(
        stmt=f"{PREFIX}fn(x, *args)",
        globals={
            "x": x,
            "args": params,
            "fn": partial(xsw.swiglu, op=xsw.SwiGLUEagerOp),
        },
        label="swiglu_fw",
        description="eager",
        sub_label=sub_label,
    )


def benchmark_swiglu_bw(shape, dtype, bias: bool):
    if dtype == "autocast_half":
        inp_dtype, model_dtype = torch.float, torch.float
        cm: Any = partial(torch.cuda.amp.autocast, enabled=True, dtype=torch.float16)
    else:
        inp_dtype, model_dtype = dtype, dtype
        cm = nullcontext

    x = torch.randn(shape[:2], device=device, dtype=inp_dtype)
    x.requires_grad_()
    module = (
        xsw.SwiGLU(in_features=shape[1], hidden_features=shape[2], bias=bias)
        .to(device)
        .to(model_dtype)
    )

    dtype_str = DTYPE2STR.get(dtype, dtype)
    bstr = "bias" if bias else "nobi"
    sub_label = f"{dtype_str} B={shape[0]}, I={shape[1]}, H={shape[2]} {bstr}"

    params = module._ordered_params()
    with cm():
        out = xsw.swiglu(x, *params, op=OP)
    grad = torch.zeros_like(out)

    yield benchmark.Timer(
        stmt="out.backward(grad, retain_graph=True)",
        globals={
            "out": out,
            "grad": grad,
        },
        label="swiglu_bw",
        description=OP.NAME,
        sub_label=sub_label,
    )
    del out

    with cm():
        out = xsw.swiglu(x, *params, op=xsw.SwiGLUEagerOp)

    yield benchmark.Timer(
        stmt="out.backward(grad, retain_graph=True)",
        globals={
            "out": out,
            "grad": grad,
        },
        label="swiglu_bw",
        description="eager",
        sub_label=sub_label,
    )


if torch.version.hip:
    print("This benchmark could not be done on ROCM!")
else:
    benchmark_main_helper(benchmark_swiglu, CASES, min_run_time=min_run_time)
    benchmark_main_helper(benchmark_swiglu_bw, CASES, min_run_time=min_run_time)
