# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import argparse
from typing import Any, Dict

import torch
import torchdynamo
import triton

import xformers
from xformers.benchmarks.utils import TestCase, pretty_plot, pretty_print
from xformers.components import Activation
from xformers.components.feedforward import MLP, FusedMLP

ORIG_FLAG = xformers._is_functorch_available
SHAPES = [
    (8, 256, 512),
    (8, 512, 1024),
    (4, 1024, 1024),
    (2, 2048, 2048),
    # (1, 2048, 4096),
    # (1, 1024, 12288),
]

HIDDEN_LAYER_MULTIPLIER = [4]


def bench_MLP(backward: bool, bias: bool, dropout: float, activation: Activation):
    device = torch.device("cuda")
    bw = "+bw" if backward else ""

    for dtype in [torch.float16, torch.float32]:
        results: Dict[str, Any] = {}
        results_mem: Dict[str, Any] = {}

        print(backward, bias, dropout, activation)

        for B, M, K in SHAPES:
            for hlm in HIDDEN_LAYER_MULTIPLIER:
                fused_mlp = FusedMLP(
                    dim_model=K,
                    dropout=dropout,
                    activation=activation,
                    hidden_layer_multiplier=hlm,
                    bias=bias,
                ).to(device=device, dtype=dtype)

                # Vanilla MLP
                xformers._is_functorch_available = False
                standard_mlp = MLP(
                    dim_model=K,
                    dropout=dropout,
                    activation=activation,
                    hidden_layer_multiplier=hlm,
                    bias=bias,
                ).to(device=device, dtype=dtype)

                # Enable functorch
                xformers._is_functorch_available = True
                functorch_mlp = MLP(
                    dim_model=K,
                    dropout=dropout,
                    activation=activation,
                    hidden_layer_multiplier=hlm,
                    bias=bias,
                ).to(device=device, dtype=dtype)

                xformers._is_functorch_available = ORIG_FLAG

                a = torch.randn(
                    (B, M, K), requires_grad=backward, device=device, dtype=dtype
                )

                def mlp_standard():
                    y = standard_mlp(a)
                    if backward:
                        torch.norm(y).backward()
                    return y

                def mlp_dynamo():
                    with torchdynamo.optimize("nvfuser"):
                        y = standard_mlp(a)
                        if backward:
                            torch.norm(y).backward()
                        return y

                # def mlp_triton_fused():
                #     with torchdynamo.optimize("nvfuser"):
                #         y = triton_fused_mlp(a)
                #         if backward:
                #             torch.norm(y).backward()
                #         return y

                # def mlp_nvfused():
                #     y = nvfused_mlp(a)
                #     if backward:
                #         torch.norm(y).backward()
                #     return y

                def mlp_functorch():
                    y = functorch_mlp(a)
                    if backward:
                        torch.norm(y).backward()
                    return y

                for testcase in [
                    TestCase(
                        mlp_standard,
                        "standard - {} - {} bias - {} drop - fw{}".format(
                            activation,
                            "no" if not bias else "",
                            dropout,
                            "+bw" if backward else "",
                        ),
                    ),
                    TestCase(
                        mlp_dynamo,
                        "dynamo/nvfuser - {} - {} bias - {} drop - fw{}".format(
                            activation,
                            "no" if not bias else "",
                            dropout,
                            "+bw" if backward else "",
                        ),
                    ),
                    TestCase(
                        mlp_functorch,
                        "aotautograd/nvfuser - {} - {} bias - {} drop - fw{}".format(
                            activation,
                            "no" if not bias else "",
                            dropout,
                            "+bw" if backward else "",
                        ),
                    ),
                ]:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()

                    time = triton.testing.do_bench(testcase.function)[0]

                    torch.cuda.synchronize()
                    max_memory = torch.cuda.max_memory_allocated() // 2**20

                    key = f"{B} x {M} x {K} - {hlm}"
                    if key not in results:
                        results[key] = {}
                    results[key][testcase.name] = f"{time:.2f}"

                    if key not in results_mem:
                        results_mem[key] = {}

                    results_mem[key][testcase.name] = f"{max_memory:.1f}"

        pretty_print(
            results,
            title=f"\n --- Type: {dtype} --- ",
            units="runtime in ms ",
        )
        pretty_plot(
            results,
            title=f"RUNTIME MLP-{activation}-FW{bw}-{dtype}-{dropout}drop",
            units="runtime in ms",
            dash_key="torch",
        )

        pretty_print(
            results_mem,
            title=f"\n --- Type: {dtype} --- ",
            units="MB ",
        )
        pretty_plot(
            results_mem,
            title=f"MAXMEM MLP-{activation}-FW{bw}-{dtype}-{dropout}drop",
            units="MB",
            dash_key="torch",
        )


if __name__ == "__main__":
    # Get the user requests
    parser = argparse.ArgumentParser("Benchmark MLP")
    parser.add_argument("-act", "--activations", nargs="+", default=[Activation.GeLU])
    parser.add_argument("-bias", "--bias", nargs="+", default=[False, True])
    parser.add_argument("-dropout", "--dropout", nargs="+", default=[0.0, 0.1])
    args = parser.parse_args()

    for bw in [True]:
        # for bw in [False, True]:
        for bias in args.bias:
            for dropout in args.dropout:
                for activation in args.activations:
                    bench_MLP(
                        backward=bw,
                        bias=bias,
                        dropout=float(dropout),
                        activation=activation,
                    )
