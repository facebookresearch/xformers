# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode, _get_current_dispatch_mode

import xformers.profiler
from xformers.profiler.slow_ops_profiler import GemmOpComputeFlops, flop_mapping

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


# Not using the PyTorch profiler, as it causes segfaults
# in the CI ~30% of the time
TEST_SCHEDULE = tuple(
    x
    for x in xformers.profiler.api.DEFAULT_SCHEDULE
    if x[0] is not xformers.profiler.PyTorchProfiler
)


class GEMMShapeDispatcher(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.mnk = (0, 0, 0)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        compute_flops = flop_mapping[func._overloadpacket]
        if isinstance(compute_flops, GemmOpComputeFlops):
            self.mnk = compute_flops._get_mnk(args)
        return func(*args)


def test_gemm_flops() -> None:
    M, N, K = 13, 17, 53

    a = torch.empty([M, K])
    b = torch.empty([K, N])
    x = torch.empty([K])

    with GEMMShapeDispatcher() as disp:
        a @ b
        assert disp.mnk == (M, N, K)
    with GEMMShapeDispatcher() as disp:
        a @ x
        assert disp.mnk == (M, 1, K)
    with GEMMShapeDispatcher() as disp:
        torch.nn.functional.linear(a, b.transpose(0, 1))
        assert disp.mnk == (M, N, K)
    with GEMMShapeDispatcher() as disp:
        torch.addmm(torch.empty([1, 1]), a, b)
        assert disp.mnk == (M, N, K)

    B = 3
    ba = torch.empty([B, M, K])
    bb = torch.empty([B, K, N])
    with GEMMShapeDispatcher() as disp:
        ba @ bb
        assert disp.mnk == (B * M, N, K)
    with GEMMShapeDispatcher() as disp:
        ba @ bb[:1]
        assert disp.mnk == (B * M, N, K)
    with GEMMShapeDispatcher() as disp:
        ba[:1] @ bb
        assert disp.mnk == (B * M, N, K)
    with GEMMShapeDispatcher() as disp:
        ba @ bb[0]
        assert disp.mnk == (B * M, N, K)
    with GEMMShapeDispatcher() as disp:
        torch.addbmm(torch.empty([1, 1]), ba, bb)
        assert disp.mnk == (B * M, N, K)


@cuda_only
def test_profiler_dispatcher_stream_workaround() -> None:
    x = torch.zeros([10, 10], device="cuda")
    with xformers.profiler.profile(
        "test_profiler_dispatcher_stream_workaround", schedule=TEST_SCHEDULE
    ):
        for _ in range(20):
            x.record_stream(torch.cuda.Stream())  # type: ignore
            xformers.profiler.step()


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

    def one_step() -> None:
        model(inp).sum().backward()
        optim.step()
        optim.zero_grad()

    # Warmup
    for _ in range(2):
        one_step()

    # Run with profiler
    with xformers.profiler.profile(
        "test_profiler_overhead", module=model, schedule=TEST_SCHEDULE
    ):
        for _ in range(PROFILER_MAX_STEPS_OVERHEAD):
            one_step()

        assert not model._forward_hooks
        assert not model._forward_pre_hooks
        assert not model._backward_hooks
        assert _get_current_dispatch_mode() is None
