# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import triton

from xformers.benchmarks.utils import TestCase, pretty_plot, pretty_print
from xformers.components import Activation, ResidualNormStyle, build_activation
from xformers.components.nvfuser import (
    NVFusedBiasActivationDropout,
    NVFusedBiasDropoutRes,
    NVFusedBiasDropoutResLayerNorm,
)
from xformers.components.nvfuser.bias_act_dropout import _fn as bias_act_dropout
from xformers.components.nvfuser.bias_dropout_res import _fn as bias_dropout_res
from xformers.components.nvfuser.bias_dropout_res_layernorm import (
    _fn as bias_dropout_res_layernorm,
)
from xformers.components.nvfuser.utils import build_nvfused
from xformers.triton import FusedDropoutBias

SHAPES = [
    (8, 256, 512),
    (8, 512, 1024),
    (4, 1024, 1024),
    (2, 2048, 2048),
    (1, 2048, 12288),
    (2, 4096, 4096),
]

P = 0.1


def build_torch_fn(
    pattern: nn.Module,
    shape: tuple,
    bias: Optional[torch.Tensor],
    activation: Optional[Activation],
    p: float,
    layer_norm_style: Optional[ResidualNormStyle],
    dtype: torch.dtype,
):
    torch_act = build_activation(activation)
    if pattern == NVFusedBiasActivationDropout:
        return partial(bias_act_dropout, bias=bias, activation=torch_act, prob=p)
    elif pattern == NVFusedBiasDropoutRes:
        return partial(bias_dropout_res, bias=bias, prob=p)
    elif pattern == NVFusedBiasDropoutResLayerNorm:
        norm = nn.LayerNorm(shape[-1]).to(device=torch.device("cuda"), dtype=dtype)
        return partial(
            bias_dropout_res_layernorm,
            bias=bias,
            prob=p,
            layer_norm_style=layer_norm_style,
            norm=norm,
        )
    else:
        raise ValueError


def bench_nvfused(
    fused_pattern: nn.Module,
    bias: bool,
    backward: bool,
    activation: Optional[Activation],
    layer_norm_style: Optional[ResidualNormStyle],
):
    device = torch.device("cuda")

    pattern_str = {
        NVFusedBiasActivationDropout: "Bias_Act_Dropout",
        NVFusedBiasDropoutRes: "Bias_Dropout_Res",
        NVFusedBiasDropoutResLayerNorm: "Bias_Dropout_Res_LayerNorm",
    }[
        fused_pattern  # type: ignore
    ]

    for dtype in [
        torch.float16,
        torch.float32,
    ]:
        results: Dict[str, Any] = {}
        results_mem: Dict[str, Any] = {}

        for B, M, K in SHAPES:
            a = torch.rand(
                (B, M, K), device=device, dtype=dtype, requires_grad=backward
            )
            b = torch.rand(K, device=device, dtype=dtype, requires_grad=backward)

            torch_fn = build_torch_fn(
                fused_pattern,
                (B, M, K),
                b if bias else None,
                activation,
                P,
                layer_norm_style,
                dtype,
            )

            nvfuser_fn = build_nvfused(
                fused_pattern, (B, M, K), bias, activation, P, layer_norm_style
            )
            nvfuser_fn.cuda()
            nvfuser_fn.to(device=device, dtype=dtype)
            residual = nvfuser_fn.requires_residual

            triton_fn = (
                FusedDropoutBias(
                    P, bias_shape=K if bias else None, activation=activation
                )
                if fused_pattern == NVFusedBiasActivationDropout
                else None
            )

            def step(fn, residual, x):
                y = fn(x=x, residual=x) if residual else fn(x)
                if backward:
                    y.grad = None
                    torch.norm(y).backward()
                return y

            testcases = [
                TestCase(
                    partial(step, fn=torch_fn, residual=residual),
                    "pytorch- bias: {} - fw{}{}{}".format(
                        bias,
                        "+bw" if backward else "",
                        f" - Act: {activation}" if activation is not None else "",
                        f" - Style: {layer_norm_style}"
                        if layer_norm_style is not None
                        else "",
                    ),
                ),
                TestCase(
                    partial(step, fn=nvfuser_fn, residual=residual),
                    "nvFuser- bias: {} - fw{}{}{}".format(
                        bias,
                        "+bw" if backward else "",
                        f" - Act: {activation}" if activation is not None else "",
                        f" - Style: {layer_norm_style}"
                        if layer_norm_style is not None
                        else "",
                    ),
                ),
            ]
            if triton_fn is not None:
                triton_test = TestCase(
                    partial(step, fn=triton_fn, residual=residual),
                    "triton- bias: {} - fw{}{}{}".format(
                        bias,
                        "+bw" if backward else "",
                        f" - Act: {activation}" if activation is not None else "",
                        f" - Style: {layer_norm_style}"
                        if layer_norm_style is not None
                        else "",
                    ),
                )
                testcases.append(triton_test)

            for testcase in testcases:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

                time = triton.testing.do_bench(
                    lambda: testcase.function(x=a), grad_to_none=[a, b]
                )[0]

                torch.cuda.synchronize()
                max_memory = torch.cuda.max_memory_allocated() // 2**20

                key = f"B={B}, M={M}, K={K}"
                if key not in results:
                    results[key] = {}

                results[key][testcase.name] = f"{time:.3f}"

                # Record peak mem usage
                if key not in results_mem:
                    results_mem[key] = {}
                results_mem[key][testcase.name] = f"{max_memory:.1f}"

        pretty_print(
            results,
            title="\n --- RUNTIME Type: {} {} --- ".format(pattern_str, dtype),
            units="ms",
        )
        pretty_print(
            results_mem,
            title="\n --- PEAK MEMORY Type: {} {} --- ".format(pattern_str, dtype),
            units="MB",
        )
        pretty_plot(
            results,
            title="RUNTIME-{}-FW{}-{}{}-{}{}".format(
                pattern_str,
                "+BW" if backward else "",
                bias,
                f"-{activation}" if activation is not None else "",
                dtype,
                f"-{layer_norm_style}" if layer_norm_style is not None else "",
            ),
            units="ms",
            dash_key="pytorch",
            legend_loc="upper left",
        )
        pretty_plot(
            results_mem,
            title="MAXMEM-{}-FW{}-{}{}-{}{}".format(
                pattern_str,
                "+BW" if backward else "",
                bias,
                f"-{activation}" if activation is not None else "",
                dtype,
                f"-{layer_norm_style}" if layer_norm_style is not None else "",
            ),
            units="MB",
            dash_key="pytorch",
            legend_loc="upper left",
        )


PATTERNS = [
    NVFusedBiasActivationDropout,
    NVFusedBiasDropoutRes,
    NVFusedBiasDropoutResLayerNorm,
]

for pattern in PATTERNS:
    activations: List[Optional[Activation]] = (
        [Activation.ReLU, Activation.GeLU, Activation.SquaredReLU]
        if pattern == NVFusedBiasActivationDropout
        else [None]
    )
    for activation in activations:
        for bw in [True, False]:
            for bias in [True, False]:
                styles: List[Optional[ResidualNormStyle]] = (
                    [ResidualNormStyle.Pre, ResidualNormStyle.Post]
                    if pattern == NVFusedBiasDropoutResLayerNorm
                    else [None]
                )
                for style in styles:
                    bench_nvfused(pattern, bias, bw, activation, style)  # type: ignore
