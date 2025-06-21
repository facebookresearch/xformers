# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import math
from contextlib import contextmanager
from typing import cast, Union

import pytest
import torch
import torch.nn as nn

import xformers.ops as xops
import xformers.ops.fmha as fmha
import xformers.profiler
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils._python_dispatch import _get_current_dispatch_mode
from xformers.profiler import profile_analyzer

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


# Not using the PyTorch profiler, as it causes segfaults
# in the CI ~30% of the time
TEST_SCHEDULE = tuple(
    x
    for x in xformers.profiler.api.DEFAULT_SCHEDULE
    if x[0] is not xformers.profiler.PyTorchProfiler
)


@cuda_only
def test_profiler_dispatcher_stream_workaround() -> None:
    x = torch.zeros([10, 10], device="cuda")
    with xformers.profiler.profile(
        "test_profiler_dispatcher_stream_workaround", schedule=TEST_SCHEDULE
    ):
        for _ in range(20):
            x.record_stream(torch.cuda.Stream())  # type: ignore
            xformers.profiler.step()


@cuda_only
@pytest.mark.parametrize(
    "device_bs_mm",
    [("cpu", 512, 1)]
    + (
        [
            # GPU bound
            ("cuda", 4096, 8),
            # CPU bound on GPU
            ("cuda", 1, 1),
        ]
        if torch.cuda.is_available()
        else []
    ),
)
def test_profiler_overhead(device_bs_mm) -> None:
    PROFILER_MAX_STEPS_OVERHEAD = 30

    device, bs, model_mult = device_bs_mm

    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 512 * model_mult),
        torch.nn.Linear(512 * model_mult, 1024),
    )
    model.to(device)
    inp = torch.randn([bs, 1024], device=device)
    optim = torch.optim.Adam(model.parameters())

    def one_step(model) -> None:
        model(inp).sum().backward()
        optim.step()
        optim.zero_grad()

    # Warmup
    for _ in range(2):
        one_step(model)

    # Run with profiler
    with xformers.profiler.profile(
        "test_profiler_overhead", module=model, schedule=TEST_SCHEDULE
    ):
        for _ in range(PROFILER_MAX_STEPS_OVERHEAD):
            one_step(model)

        assert not model._forward_hooks
        assert not model._forward_pre_hooks
        assert not model._backward_hooks
        assert _get_current_dispatch_mode() is None

    model_opt = torch.compile(model)
    model_opt_casted = cast(torch.nn.Module, model_opt)

    # Warmup
    for _ in range(2):
        one_step(model_opt_casted)

    # Run with profiler
    with xformers.profiler.profile(
        "test_profiler_overhead", module=model_opt_casted, schedule=TEST_SCHEDULE
    ):
        for _ in range(PROFILER_MAX_STEPS_OVERHEAD):
            one_step(model_opt_casted)

        assert not model_opt_casted._forward_hooks
        assert not model_opt_casted._forward_pre_hooks
        assert not model_opt_casted._backward_hooks
        assert _get_current_dispatch_mode() is None


@contextmanager
def assert_flops(
    error_msg: str,
    *,
    match: int = -1,
    at_least: int = -1,
    at_most: Union[int, float] = math.inf,
    fw=True,
    bw=True,
):
    try:
        with torch.profiler.profile(
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
            with_flops=True,
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        ) as p:
            yield
    finally:
        results = profile_analyzer.AnalyzedTrace.from_profile(
            p.profiler.kineto_results.events()
        )
        total_flops = 0.0
        if fw:
            total_flops += sum(results.operations_per_dtype_fw.values())
        if bw:
            total_flops += sum(results.operations_per_dtype_bw.values())
        if match != -1:
            # Some tolerance
            assert (
                total_flops * 0.99 < match < total_flops * 1.01
            ), f"{error_msg}: {total_flops} flops, expected {match}"
        assert total_flops >= at_least, error_msg
        assert total_flops <= at_most, error_msg


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float64, torch.float, torch.bfloat16]
)
@cuda_only
def test_analyze_prof(dtype) -> None:
    B, N = 64, 128
    w = torch.empty([128, 128], dtype=dtype, device="cuda", requires_grad=True)
    x = torch.ones([B, 1, N, 128], dtype=dtype, device="cuda", requires_grad=True)
    with assert_flops("Linear", match=2 * B * N * 128 * 128):
        x = x @ w
    with assert_flops("LinearBW", match=2 * B * N * 128 * 128 * 2, fw=False):
        x.backward(x)


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize(
    "backend",
    [
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.CUDNN_ATTENTION,
    ],
    ids=["mem-eff", "flash", "cudnn"],
)
@pytest.mark.parametrize("causal", [True, False], ids=["causal", ""])
@cuda_only
def test_analyze_prof_sdpa(dtype, backend, causal: bool) -> None:
    B, M, H, K = 64, 1024, 3, 128
    x = torch.ones([B, H, M, K], dtype=dtype, device="cuda", requires_grad=True)
    fw_flops = 2 * 2 * M * M * K
    fw_flops *= B * H
    if causal:
        fw_flops //= 2
    device_sm = torch.cuda.get_device_capability(x.device)
    if backend in [
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.FLASH_ATTENTION,
    ] and device_sm not in ((8, 0), (9, 0)):
        pytest.skip("not available")
    with sdpa_kernel(backend):
        with assert_flops("SDPA", match=fw_flops):
            x = nn.functional.scaled_dot_product_attention(x, x, x, is_causal=causal)
        with assert_flops("SDPA BW", match=fw_flops * 5 // 2):
            x.backward(x)


@pytest.mark.parametrize(
    "op",
    [
        (fmha.cutlass.FwOp, fmha.cutlass.BwOp),
        (fmha.flash.FwOp, fmha.flash.BwOp),
    ],
    ids=["cutlass", "flash"],
)
@pytest.mark.parametrize("causal", [True, False], ids=["causal", ""])
@cuda_only
def test_analyze_prof_memeff(op, causal: bool) -> None:
    dtype = torch.float16
    B, M, H, K = 64, 256, 3, 128
    x = torch.ones([B, M, H, K], dtype=dtype, device="cuda", requires_grad=True)
    fw_flops = 2 * 2 * M * M * K
    fw_flops *= B * H
    bias = None
    if causal:
        bias = fmha.attn_bias.LowerTriangularMask()
        fw_flops //= 2
    device_sm = torch.cuda.get_device_capability(x.device)
    if device_sm < op[0].CUDA_MINIMUM_COMPUTE_CAPABILITY:
        pytest.skip(f"Requires sm{op[0].CUDA_MINIMUM_COMPUTE_CAPABILITY}")
    with assert_flops("memory_efficient_attention", match=fw_flops):
        y = xops.memory_efficient_attention(x, x, x, attn_bias=bias, op=op)
    with assert_flops("memory_efficient_attention BW", match=fw_flops * 5 // 2):
        y.backward(y)
