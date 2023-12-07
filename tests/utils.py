# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import numpy as np
import torch


def assert_allclose(
    out: Optional[torch.Tensor],
    ref: Optional[torch.Tensor],
    msg: str = "failed",
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> None:
    assert out is not None, f"{msg}: output Tensor is None"
    assert ref is not None, f"{msg}: reference Tensor is None"
    assert out.shape == ref.shape, f"Shape: {out.shape} (expected: {ref.shape})"
    if out.dtype != ref.dtype:
        assert False, f"out dtype: {out.dtype}, ref dtype: {ref.dtype}"
    if out.numel() == 0:
        return
    flatten_diff = ((out - ref).abs() - atol - ref.abs() * rtol).flatten()
    max_pos = flatten_diff.argmax()
    max_location = np.unravel_index(int(max_pos), out.shape)
    max_diff = flatten_diff[max_pos]
    num_different = flatten_diff.numel() - torch.count_nonzero(flatten_diff <= 0)
    percentage = num_different / flatten_diff.numel()
    del flatten_diff
    assert torch.allclose(out, ref, rtol=rtol, atol=atol), (
        f"{msg}: "
        f"out={out.flatten()[max_pos]} and ref={ref.flatten()[max_pos]} (diff={max_diff} > 0)"
        f" at {max_location} of shape {tuple(out.shape)} / atol={atol}, rtol={rtol}"
        f"/ total failing elements: {num_different}, percentage={percentage}"
    )
