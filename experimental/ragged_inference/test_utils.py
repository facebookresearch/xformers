# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Dict, Tuple

import numpy as np
import torch

_DTYPE_PRECISIONS = {
    torch.float16: (1e-3, 1e-3),
    torch.bfloat16: (1e-1, 1e-3),
    torch.float32: (1e-4, 1e-5),
    torch.float64: (1e-5, 1e-8),
}


def _get_default_rtol_and_atol(
    actual: torch.Tensor, expected: torch.Tensor
) -> Tuple[float, float]:
    expected_rtol = expected_atol = actual_rtol = actual_atol = 0.0
    if isinstance(actual, torch.Tensor):
        actual_rtol, actual_atol = _DTYPE_PRECISIONS.get(actual.dtype, (0.0, 0.0))
    if isinstance(expected, torch.Tensor):
        expected_rtol, expected_atol = _DTYPE_PRECISIONS.get(expected.dtype, (0.0, 0.0))
    return max(actual_rtol, expected_rtol), max(actual_atol, expected_atol)


def assert_eq(actual, expected, msg="", rtol=None, atol=None):
    """Asserts two things are equal with nice support for lists and tensors

    It also gives prettier error messages than assert a == b
    """
    if not msg:
        msg = f"Values are not equal: \n\ta={actual} \n\tb={expected}"

    if isinstance(actual, torch.Size):
        actual = list(actual)
    if isinstance(expected, torch.Size):
        expected = list(expected)

    if isinstance(actual, tuple):
        actual = list(actual)
    if isinstance(expected, tuple):
        expected = list(expected)

    if isinstance(actual, torch.Tensor):
        if rtol is None and atol is None:
            rtol, atol = _get_default_rtol_and_atol(actual=actual, expected=expected)
        torch.testing.assert_allclose(actual, expected, msg=msg, rtol=rtol, atol=atol)
        return
    if isinstance(actual, np.ndarray):
        np.testing.assert_allclose(actual, expected, rtol=rtol or 0, atol=atol or 0)
        return
    if isinstance(actual, torch.Size) or isinstance(expected, torch.Size):
        assert actual == expected, msg
        return
    if isinstance(actual, dict):
        assert isinstance(expected, dict)
        assert actual.keys() == expected.keys(), msg
        for key in actual.keys():
            assert_eq(actual[key], expected[key], msg=msg, rtol=rtol, atol=atol)
        return
    if isinstance(actual, (tuple, list, set)):
        assert isinstance(expected, type(actual))
        assert len(actual) == len(expected), msg
        for ai, bi in zip(actual, expected):
            assert_eq(ai, bi, msg=msg, rtol=rtol, atol=atol)
        return

    if rtol is None and atol is None:
        assert actual == expected, f"{actual} != {expected}"
    else:
        atol = 0 if atol is None else atol
        rtol = 0 if rtol is None else rtol
        assert (
            abs(actual - expected) <= atol + expected * rtol
        ), f"{actual} != {expected}"


_gpu_is_old = None


def gpu_capabilities_older_than_70() -> bool:
    """Return True if the GPU's compute capability is older than SM70."""
    global _gpu_is_old
    if _gpu_is_old is None:
        for i in range(torch.cuda.device_count()):
            major, _ = torch.cuda.get_device_capability(f"cuda:{i}")
            if major < 7:
                _gpu_is_old = True
        if _gpu_is_old is None:
            _gpu_is_old = False
    return _gpu_is_old


def bf16_support():
    # Ampere cards support bf16
    return torch.cuda.is_available() and (
        "RTX" in torch.cuda.get_device_name() or "A100" in torch.cuda.get_device_name()
    )


def make_seq(n_ctx: int, value: int, d_model: int, dtype: Dict[str, Any]):
    return torch.full([n_ctx, d_model], value, **dtype)


def make_seq_arange(n_ctx: int, start_value: int, d_head: int, dtype: Dict[str, Any]):
    return (
        torch.full([n_ctx, d_head], start_value, **dtype)
        + torch.arange(n_ctx, **dtype)[:, None]
    )
