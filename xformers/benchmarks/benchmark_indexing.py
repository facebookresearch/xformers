# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import random

import torch
from utils import DTYPE2STR, benchmark_main_helper2, product_dict

import xformers.ops as xops

min_run_time = 0.2
device = torch.device("cuda")

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


class IndexSelectBenchmark:
    def __init__(self, dtype, batches, D, keep_ratio, bw: bool) -> None:
        dtype_str = DTYPE2STR.get(dtype, dtype)
        self.sub_label = f"{dtype_str} D={D} batches={batches} keep={keep_ratio}"
        self.label = "index_select"

        indices = []
        sources = []
        for B, seqlen in batches:
            index = [i for i in range(B)]
            random.Random(B).shuffle(index)
            indices.append(
                torch.zeros(
                    index[int(keep_ratio * B)],
                    dtype=torch.int64,
                    device="cuda",
                )
            )
            source_i = torch.randn(
                [B, seqlen * D], dtype=dtype, device="cuda", requires_grad=bw
            )
            sources.append(source_i)
        self.indices, self.sources = indices, sources
        self.out = torch.Tensor()

    def fw(self) -> None:
        self.out = xops.index_select_cat(self.sources, self.indices)

    def bw(self):
        for src in self.sources:
            src.grad = None
        self.out.backward(self.out, retain_graph=True)


class IndexSelectBenchmarkBaseline(IndexSelectBenchmark):
    def fw(self) -> None:
        self.out = torch.cat(
            [s[i].flatten() for s, i in zip(self.sources, self.indices)], dim=0
        )


benchmark_main_helper2(
    "scaled_index_add_fw",
    fw=True,
    functions={
        "xformers": ScaledIndexAddBenchmark,
        "pytorch": ScaledIndexAddBenchmarkBaseline,
    },
    cases=CASES_IADD,
    min_run_time=min_run_time,
)

benchmark_main_helper2(
    "scaled_index_add_fwbw",
    fw=True,
    bw=True,
    functions={
        "xformers": ScaledIndexAddBenchmark,
        "pytorch": ScaledIndexAddBenchmarkBaseline,
    },
    cases=CASES_IADD,
    min_run_time=min_run_time,
)

benchmark_main_helper2(
    "index_select_fw",
    fw=True,
    functions={
        "xformers": IndexSelectBenchmark,
        "pytorch": IndexSelectBenchmarkBaseline,
    },
    cases=CASES_ISELECT,
    min_run_time=min_run_time,
)

benchmark_main_helper2(
    "index_select_fwbw",
    fw=True,
    bw=True,
    functions={
        "xformers": IndexSelectBenchmark,
        "pytorch": IndexSelectBenchmarkBaseline,
    },
    cases=CASES_ISELECT,
    min_run_time=min_run_time,
)
