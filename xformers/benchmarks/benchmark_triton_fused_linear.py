# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Dict, List, Optional

import torch
import triton

from xformers.benchmarks.utils import TestCase, pretty_plot, pretty_print
from xformers.components import Activation, build_activation
from xformers.triton.fused_linear_layer import FusedLinear

SHAPES = [
    (8, 512, 256),  # Batch x Seq x Embedding
    (8, 512, 512),
    (4, 512, 1024),
    (2, 512, 2048),
    (2, 512, 4096),
    (2, 512, 8192),
]

# Switch PyTorch to TF32 accumulations, Triton does that also
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def get_metrics_transform(
    activation: Optional[Activation],
    a: torch.Tensor,
    w: torch.Tensor,
    b: Optional[torch.Tensor],
    backward: bool,
):
    # all operations will involve a * weight.
    flop = a.shape[0] * a.shape[1] * w.shape[1] * (2 * a.shape[2] - 1)

    # optional activation on top
    if activation is not None:
        flop += a.numel()

    # optionally * 2 (before the bias) if backward
    if backward:
        flop *= 2

        # backward will also output a gradient with respect to the bias
        # which consolidates on all the activation gradient
        flop += a.shape[0] * a.shape[1] * w.shape[1]

        # backward will also ouput another gradient with respect to the weight,
        # which is another matmul, in between the grad_out and the inputs this time
        flop += a.shape[0] * a.shape[1] * w.shape[1] * (2 * a.shape[2] - 1)

    # optional bias on top
    if b is not None:
        flop += b.numel()

    def metric_conversion(ms):
        # Returns TFlops/second
        return flop * 1e-12 / (ms * 1e-3)

    return metric_conversion


def bench_linear(activations: List[Optional[Activation]]):
    device = torch.device("cuda")

    for dtype in [
        torch.float32,
        torch.float16,
    ]:
        for backward in [True, False]:

            for activation in activations:
                results: Dict[str, Any] = {}

                for bias in [False, True]:
                    for B, M, K in SHAPES:
                        a = torch.rand(
                            B, M, K, device=device, dtype=dtype, requires_grad=backward
                        )

                        # Pytorch linear layer + activation
                        torch_linear = torch.nn.Linear(K, 4 * K, bias=bias).to(
                            dtype=dtype, device=device
                        )
                        torch_activation = build_activation(activation)

                        # Fused layer equivalent
                        fused_linear = FusedLinear(
                            K, 4 * K, bias=bias, activation=activation
                        ).to(dtype=dtype, device=device)

                        def torch_step(x):
                            y = torch_activation(torch_linear(x))
                            if backward:
                                torch.norm(y).backward()
                            return y

                        def triton_step(x):
                            y = fused_linear(x)

                            if backward:
                                torch.norm(y).backward()
                            return y

                        metrics_transform = get_metrics_transform(
                            activation,
                            a,
                            torch_linear.weight,
                            torch_linear.bias,
                            backward,
                        )

                        for testcase in [
                            TestCase(
                                torch_step,
                                "pytorch - {} - {} bias - fw{}".format(
                                    activation,
                                    "no" if not bias else "",
                                    "+bw" if backward else "",
                                ),
                            ),
                            TestCase(
                                triton_step,
                                "triton  - {} - {} bias - fw{}".format(
                                    activation,
                                    "no" if not bias else "",
                                    "+bw" if backward else "",
                                ),
                            ),
                        ]:
                            time = triton.testing.do_bench(
                                lambda: testcase.function(a)
                            )[0]
                            key = f"B={B}, M={M}, K={K}"
                            if key not in results:
                                results[key] = {}

                            metric = metrics_transform(time)
                            results[key][testcase.name] = f"{metric:.1f}"

                pretty_print(
                    results,
                    title="\n --- Type: {} ---".format(dtype),
                    units="TFlops/s",
                )

                _type = "_fp16" if dtype == torch.float16 else "_fp32"
                title = "FusedLinear" + _type + "_FW"
                if backward:
                    title += "_BW"
                title += "_" + activation.value if activation else "_none"
                pretty_plot(results, title, "TFlops/s", dash_key="pytorch")


activations = [ac for ac in Activation] + [None]  # type: ignore
bench_linear(activations)  # type: ignore
