# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import torch
import torch.nn.functional as F

import xformers.ops as xops
from torch import nn
from utils import benchmark_main_helper2, DTYPE2STR, product_dict

min_run_time = 0.5
device = torch.device("cuda")

CASES = list(
    product_dict(
        B_in_hidden_out_ft=[
            (2048 * 8, 2048, 2048 * 3, 2048),
            (2048, 5120, 5120 * 3, 5120),  # 13b
            (1024, 8192, 8192 * 3, 8192),  # 30b
            (2048, 8192, 8192 * 3, 8192),  # 30b
            (2048 * 2, 8192, 8192 * 3, 8192),  # 30b
            # DINO ViT-L: lg + sm crops (patch16)
            (64 * 2 * (14 * 14 + 1) + 64 * 8 * (6 * 6 + 1), 1024, 1024 * 4, 1024),
            # DINO ViT-g: lg + sm crops (patch16)
            (
                12 * 2 * (16 * 16 + 1 + 11) + 12 * 8 * (7 * 7 + 1 + 11),
                1536,
                1536 * 4,
                1536,
            ),
        ],
        dtype=[torch.half],
        bias=[False],
    )
)


class Mlp(nn.Module):
    LINEAR_CLS = nn.Linear

    def __init__(
        self, B_in_hidden_out_ft: Tuple[int, int, int, int], dtype, bias: bool, bw: bool
    ) -> None:
        B, in_ft, hid_ft, out_ft = B_in_hidden_out_ft
        super().__init__()
        self.label = "mlp"
        self.sub_label = (
            f"{DTYPE2STR[dtype]} ({B},{in_ft},{hid_ft},{out_ft}){' b' if bias else ''}"
        )
        self.fc1 = self.LINEAR_CLS(in_ft, hid_ft, bias=bias)
        self.act = nn.GELU()
        self.fc2 = self.LINEAR_CLS(hid_ft, out_ft, bias=bias)
        self.grad = torch.randn([B, out_ft], device="cuda", dtype=dtype)
        self.input = torch.randn(
            [B, in_ft], device="cuda", dtype=dtype, requires_grad=True
        )
        self.out = self.input
        self.to("cuda").to(dtype)

    def fw(self):
        x = self.input
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        self.out = x

    def bw(self):
        self.out.backward(self.grad, retain_graph=True)


class MlpDenseMask(Mlp):
    def fw(self):
        x = self.input
        x = self.fc1(x)

        mask = torch.ops.xformers.sparse24_largest_mask_2d(x)
        x = mask * x

        x = self.act(x)
        x = self.fc2(x)
        self.out = x


class MlpAct24(Mlp):
    def fw(self):
        x = self.input
        x = self.fc1(x)

        x = xops.sparsify24(x)

        x = self.act(x)
        x = self.fc2(x)
        self.out = x


class LinearW24(torch.nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w_sparse = xops.sparsify24(
            self.weight,
            gradient="24dense",
            backend="cusparselt",
        )
        return F.linear(input, w_sparse, self.bias)


class MlpW24(Mlp):
    LINEAR_CLS = LinearW24


class MicrobenchmarkBase:
    def __init__(
        self, B_in_hidden_out_ft: Tuple[int, int, int, int], dtype, bias: bool, bw: bool
    ) -> None:
        B, in_ft, hid_ft, out_ft = B_in_hidden_out_ft
        super().__init__()
        self.label = "mlp"
        self.sub_label = (
            f"{DTYPE2STR[dtype]} ({B},{in_ft},{hid_ft},{out_ft}){' b' if bias else ''}"
        )
        self.input = torch.randn(
            [B, in_ft], device="cuda", dtype=dtype, requires_grad=True
        )
        self.input_colMajor = self.input.t().contiguous().t()
        self.input_sp = xops.sparsify24(self.input)

    def bw(self) -> None:
        return None


class MicrobenchmarkSparsify24(MicrobenchmarkBase):
    def fw(self) -> torch.Tensor:
        xops.sparsify24(self.input)
        return self.input


class MicrobenchmarkSp24ApplyDense(MicrobenchmarkBase):
    def fw(self) -> torch.Tensor:
        xops.sparsify24_like(self.input, pattern=self.input_sp, out_dense=True)
        return self.input


class MicrobenchmarkSp24ApplyDenseT(MicrobenchmarkBase):
    def fw(self) -> torch.Tensor:
        xops.sparsify24_like(self.input_colMajor, pattern=self.input_sp, out_dense=True)
        return self.input


class MicrobenchmarkInputClone(MicrobenchmarkBase):
    def fw(self) -> torch.Tensor:
        self.input.clone()
        return self.input


functions = {
    "act24": MlpAct24,
    "dense": Mlp,
    "w24": MlpW24,
    "s24_inp_sparsify24": MicrobenchmarkSparsify24,
    "s24_inp_apply_dense": MicrobenchmarkSp24ApplyDense,
    "s24_inp_apply_dense_t": MicrobenchmarkSp24ApplyDenseT,
    "s24_inp_clone": MicrobenchmarkInputClone,
}
benchmark_main_helper2(
    "sp24_fw", fw=True, cases=CASES, functions=functions, min_run_time=min_run_time
)
benchmark_main_helper2(
    "sp24_fwbw",
    fw=True,
    bw=True,
    cases=CASES,
    functions=functions,
    min_run_time=min_run_time,
)
