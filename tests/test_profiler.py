# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import xformers.profiler
from xformers.profiler.slow_ops_profiler import GemmOpComputeFlops, flop_mapping

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


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


@cuda_only
def test_profiler_dispatcher_stream_workaround() -> None:
    x = torch.zeros([10, 10], device="cuda")
    with xformers.profiler.profile("test_profiler"):
        for _ in range(20):
            x.record_stream(torch.cuda.Stream())  # type: ignore
            xformers.profiler.step()
