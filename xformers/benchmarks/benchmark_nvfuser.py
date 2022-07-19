# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Any, Dict

import torch
import torch.nn as nn
import triton

from xformers.benchmarks.utils import TestCase, pretty_plot, pretty_print
from xformers.components import Activation, LayerNormStyle, build_activation
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


def to_gbs_fw(a, ms, bias):
    # Read and write the full array
    total = 2 * a.numel() * a.element_size()

    if bias:
        # Read the bias, ideally only once
        total += a.shape[-1] * a.element_size()

    return total * 1e-9 / (ms * 1e-3)


def build_torch_fn(
    pattern: nn.Module,
    shape: tuple,
    bias: torch.Tensor,
    activation: Activation,
    p: float,
    layer_norm_style: LayerNormStyle,
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
    activation: Activation,
    layer_norm_style: LayerNormStyle,
):
    device = torch.device("cuda")

    pattern_str = {
        NVFusedBiasActivationDropout: "Bias_Act_Dropout",
        NVFusedBiasDropoutRes: "Bias_Dropout_Res",
        NVFusedBiasDropoutResLayerNorm: "Bias_Dropout_Res_LayerNorm",
    }[fused_pattern]

    for dtype in [
        torch.float16,
        torch.float32,
    ]:
        results: Dict[str, Any] = {}

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
            residual = nvfuser_fn.requires_residual

            triton_fn = (
                FusedDropoutBias(
                    P, bias_shape=K if bias else None, activation=activation
                )
                if fused_pattern == NVFusedBiasActivationDropout
                else None
            )

            def step(fn, res, x):
                y = fn(x=x, res=x) if res else fn(x)
                if backward:
                    y.grad = None
                    torch.norm(y).backward()
                return y

            testcases = [
                TestCase(
                    partial(step, fn=torch_fn, res=residual),
                    "pytorch- bias: {} - fw{} - act: {}{}".format(
                        bias,
                        "+bw" if backward else "",
                        activation,
                        f" - Style: {layer_norm_style}"
                        if layer_norm_style is not None
                        else "",
                    ),
                ),
                TestCase(
                    partial(step, fn=nvfuser_fn, res=residual),
                    "nvFuser- bias: {} - fw{} - act: {}{}".format(
                        bias,
                        "+bw" if backward else "",
                        activation,
                        f" - Style: {layer_norm_style}"
                        if layer_norm_style is not None
                        else "",
                    ),
                ),
            ]
            if triton_fn is not None:
                triton_test = TestCase(
                    partial(step, fn=triton_fn, res=residual),
                    "triton- bias: {} - fw{} - act: {}{}".format(
                        bias,
                        "+bw" if backward else "",
                        activation,
                        f" - Style: {layer_norm_style}"
                        if layer_norm_style is not None
                        else "",
                    ),
                )
                testcases.append(triton_test)

            for testcase in testcases:
                time = triton.testing.do_bench(
                    lambda: testcase.function(x=a), grad_to_none=[a, b]
                )[0]
                key = f"B={B}, M={M}, K={K}"
                if key not in results:
                    results[key] = {}

                # Record BW
                bandwidth = to_gbs_fw(a, time, bias)
                results[key][testcase.name] = f"{bandwidth:.1f}"

        pretty_print(
            results,
            title="\n --- Type: {}{} --- ".format(pattern_str, dtype),
            units="GB/s",
        )
        pretty_plot(
            results,
            title="{}-{}-FW{}-{}-{}{}".format(
                pattern_str,
                bias,
                "+BW" if backward else "",
                dtype,
                activation,
                f"-{layer_norm_style}" if layer_norm_style is not None else "",
            ),
            units="GB/s",
            dash_key="pytorch",
            legend_loc="upper left",
        )


# for activation in [Activation.GeLU, None, Activation.SquaredReLU]:
for pattern in [NVFusedBiasDropoutResLayerNorm]:
    for activation in [Activation.ReLU, Activation.GeLU]:
        for bw in [True, False]:
            for bias in [True, False]:
                styles = (
                    [LayerNormStyle.Pre, LayerNormStyle.Post]
                    if pattern == NVFusedBiasDropoutResLayerNorm
                    else [None]
                )
                for style in styles:
                    bench_nvfused(pattern, bias, bw, activation, style)
