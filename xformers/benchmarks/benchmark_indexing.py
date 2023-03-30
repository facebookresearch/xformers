# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import itertools
import random

import torch
from torch.utils import benchmark
from utils import benchmark_main_helper

import xformers.ops as xops

min_run_time = 0.5
device = torch.device("cuda")


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


CASES_IADD = list(
    product_dict(
        shape=[
            (int(48 * 0.6), 48, 1, 257 * 1536),
            (int(48 * 0.6), 48, 257, 1536),
        ],
        scaling=[False, True],
        dtype=[torch.half],
    )
) + list(
    product_dict(
        shape=[
            # Format: [B_src, B_inp, M, D]
            (int(192 * 0.6), 192, 50, 1536),
            (int(48 * 257 * 0.6), 257 * 48, 1, 1536),
            (int(192 * 50 * 0.6), 192 * 50, 1, 1536),
            (int(16 * 257 * 0.6), 48 * 257, 1, 1536),
        ],
        scaling=[False],
        dtype=[torch.half],
    )
)

CASES_ISELECT = list(
    product_dict(
        batches=[((48, 257), (50, 192))],
        D=[1536],
        keep_ratio=[0.6],
        dtype=[torch.half],
    )
)

DTYPE2STR = {
    torch.bfloat16: "b16",
    torch.half: "f16",
    torch.float32: "f32",
}


def _setup_test(functions, fw: bool = False, bw: bool = False, **kwargs):
    for k, benchmark_cls in functions.items():
        benchmark_object = benchmark_cls(**kwargs, bw=bw)
        label = benchmark_object.label
        label += "fw" if fw else ""
        label += "bw" if bw else ""

        def run_one():
            if fw:
                benchmark_object.fw()
            if bw:
                benchmark_object.bw()

        yield benchmark.Timer(
            stmt="fn()",
            globals={
                "fn": run_one,
            },
            label=label,
            description=k,
            sub_label=benchmark_object.sub_label,
        )


class ScaledIndexAddBenchmark:
    def __init__(self, dtype, scaling: bool, shape, bw: bool) -> None:
        B_src, B_out, M, D = shape
        torch.manual_seed(B_out + B_src)
        dtype_str = DTYPE2STR.get(dtype, dtype)
        self.sub_label = f"{dtype_str} B_src={B_src}, B_out={B_out}, M={M}, D={D} s={'Y' if scaling else 'N'}"
        self.label = "scaled_index_add"
        self.alpha = 0.73

        self.inp = torch.randn(
            [B_out, M, D], device="cuda", dtype=dtype, requires_grad=bw
        )
        self.src = torch.randn(
            [B_src, M, D], device="cuda", dtype=dtype, requires_grad=bw
        )
        self.scaling = (
            torch.randn([D], device="cuda", dtype=dtype, requires_grad=bw)
            if scaling
            else None
        )
        self.index = torch.tensor(
            [i for i in range(self.src.shape[0])], dtype=torch.int64, device="cuda"
        )
        self.grad = torch.randn([B_out, M, D], device="cuda", dtype=dtype)
        self.out = torch.Tensor()

    def fw(self) -> None:
        self.out = xops.scaled_index_add(
            input=self.inp.clone(),
            index=self.index,
            source=self.src,
            scaling=self.scaling,
            alpha=self.alpha,
        )

    def bw(self):
        self.inp.grad = None
        self.src.grad = None
        if self.scaling is not None:
            self.scaling.grad = None
        self.out.backward(self.grad, retain_graph=True)


class ScaledIndexAddBenchmarkBaseline(ScaledIndexAddBenchmark):
    def fw(self) -> None:
        src_scaled = self.src
        if self.scaling is not None:
            src_scaled * self.scaling.unsqueeze(0).unsqueeze(0)
        self.out = self.inp.index_add(
            dim=0,
            source=src_scaled,
            index=self.index,
            alpha=self.alpha,
        )


def scaled_index_add_fw(**kwargs):
    yield from _setup_test(
        **kwargs,
        fw=True,
        functions={
            "xformers": ScaledIndexAddBenchmark,
            "pytorch": ScaledIndexAddBenchmarkBaseline,
        },
    )


def scaled_index_add_fwbw(**kwargs):
    yield from _setup_test(
        **kwargs,
        fw=True,
        bw=True,
        functions={
            "xformers": ScaledIndexAddBenchmark,
            "pytorch": ScaledIndexAddBenchmarkBaseline,
        },
    )


class IndexSelectBenchmark:
    def __init__(self, dtype, batches, D, keep_ratio, bw: bool) -> None:
        dtype_str = DTYPE2STR.get(dtype, dtype)
        self.sub_label = f"{dtype_str} D={D} batches={batches} keep={keep_ratio}"
        self.label = "index_select"
        srcs = [torch.randn([B, seqlen * D]) for (B, seqlen) in batches]
        src = torch.cat([s.view([-1, D]) for s in srcs], dim=0).cuda().to(dtype)
        src.requires_grad_(True)

        indices = []
        sources = []
        elements_i = 0
        for source_i in srcs:
            index = [i for i in range(source_i.shape[0])]
            random.Random(source_i.shape[0]).shuffle(index)
            indices.append(
                torch.tensor(
                    index[: int(keep_ratio * source_i.shape[0])],
                    dtype=torch.int64,
                    device="cuda",
                )
            )
            sources.append(
                src[
                    elements_i : elements_i + source_i.shape[0] * source_i.shape[1] // D
                ].reshape(source_i.shape)
            )
            elements_i += source_i.shape[0] * source_i.shape[1] // D
        self.indices, self.sources, self.src = indices, sources, src
        self.out = torch.Tensor()

    def fw(self) -> None:
        self.out = xops.index_select_cat(self.sources, self.indices)

    def bw(self):
        self.src.grad = None
        self.out.backward(self.out, retain_graph=True)


class IndexSelectBenchmarkBaseline(IndexSelectBenchmark):
    def fw(self) -> None:
        self.out = torch.cat(
            [s[i].flatten() for s, i in zip(self.sources, self.indices)], dim=0
        )


def index_select_fw(**kwargs):
    yield from _setup_test(
        **kwargs,
        fw=True,
        functions={
            "xformers": IndexSelectBenchmark,
            "pytorch": IndexSelectBenchmarkBaseline,
        },
    )


def index_select_fwbw(**kwargs):
    yield from _setup_test(
        **kwargs,
        fw=True,
        bw=True,
        functions={
            "xformers": IndexSelectBenchmark,
            "pytorch": IndexSelectBenchmarkBaseline,
        },
    )


benchmark_main_helper(scaled_index_add_fw, CASES_IADD, min_run_time=min_run_time)
benchmark_main_helper(scaled_index_add_fwbw, CASES_IADD, min_run_time=min_run_time)
benchmark_main_helper(index_select_fw, CASES_ISELECT, min_run_time=min_run_time)
benchmark_main_helper(index_select_fwbw, CASES_ISELECT, min_run_time=min_run_time)
