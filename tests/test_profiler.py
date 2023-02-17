# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

import xformers.profiler


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_profiler_dispatcher_stream_workaround() -> None:
    x = torch.zeros([10, 10], device="cuda")
    with xformers.profiler.profile("test_profiler"):
        for _ in range(20):
            x.record_stream(torch.cuda.Stream())  # type: ignore
            xformers.profiler.step()
