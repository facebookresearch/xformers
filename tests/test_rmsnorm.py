# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
from contextlib import nullcontext
from typing import Optional

import pytest
import torch
from torch import nn

from xformers.ops import RMSNorm

from .utils import assert_allclose

compute_capability = (0, 0)
if torch.cuda.is_available():
    compute_capability = torch.cuda.get_device_capability("cuda")
cuda_sm80_only = pytest.mark.skipif(
    compute_capability < (8, 0), reason="requires sm80+"
)

DTYPES = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}


class RMSNormPytorch(torch.nn.Module):
    def __init__(self, dim: int, include_weight: bool = True, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        if include_weight:
            self.weight: Optional[nn.Parameter] = nn.Parameter(torch.ones(dim))
        else:
            self.weight = None

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output.type_as(x)


@cuda_sm80_only
@pytest.mark.parametrize("K", [273, 4100])
@pytest.mark.parametrize("dtype", ["f16", "bf16", "f32"])
def test_forward(K: int, dtype: str):
    atol = 1e-8 if dtype == "f32" else 1e-4
    rtol = 1e-5 if dtype == "f32" else 0.01
    torch.manual_seed(1)
    B, M, K = 31, 27, K
    device = torch.device("cuda")

    rms_layer = RMSNorm(K).cuda()
    baseline_layer = RMSNormPytorch(K).cuda()
    x = torch.rand(B, M, K, device=device, dtype=DTYPES[dtype])
    torch.nn.init.normal_(rms_layer.weight)  # type: ignore
    with torch.no_grad():
        x_rms = rms_layer(x)
        assert x_rms.shape == x.shape
        baseline_layer.weight.copy_(rms_layer.weight)  # type: ignore
    baseline = baseline_layer(x)
    assert_allclose(x_rms, baseline, atol=atol, rtol=rtol)

    torch.nn.init.ones_(rms_layer.weight)  # type: ignore
    with torch.no_grad():
        x_rms1 = rms_layer(x)
    assert not torch.allclose(x_rms, x_rms1)
    rms1_layer = RMSNorm(K, include_weight=False)
    with torch.no_grad():
        x_rms_1 = rms1_layer(x)
    assert_allclose(x_rms1, x_rms_1, atol=atol, rtol=rtol)


@cuda_sm80_only
@pytest.mark.parametrize("K", [273, 4100])
@pytest.mark.parametrize("include_weight", [True, False])
@pytest.mark.parametrize("dtype", ["f16", "bf16", "f32"])
def test_increment(K: int, include_weight: bool, dtype: str):
    atol = 1e-8 if dtype == "f32" else 1e-4
    rtol = 1e-5 if dtype == "f32" else 0.01
    torch.manual_seed(1)
    B, M, K = 31, 27, K
    device = torch.device("cuda")
    dtype_ = DTYPES[dtype]

    rms_layer = RMSNorm(K, include_weight=include_weight).cuda()
    x_orig = torch.rand(B, M, K, device=device, dtype=dtype_)
    y_orig = torch.rand(B, M, K, device=device, dtype=dtype_)
    x = x_orig.clone()
    y = y_orig.clone()
    if include_weight:
        torch.nn.init.normal_(rms_layer.weight)  # type: ignore

    context = torch.no_grad() if include_weight else nullcontext()
    with context:  # type: ignore
        baseline = rms_layer(x_orig + y_orig)
        out = rms_layer.increment_and_forward_(x, y)
    assert_allclose(out, baseline, atol=atol, rtol=rtol)
    assert_allclose(x, x_orig + y_orig, atol=atol, rtol=rtol)
    assert_allclose(y, y_orig, atol=atol, rtol=rtol)
