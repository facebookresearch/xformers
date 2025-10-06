# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any

import pytest
import torch
from xformers.fwbw_overlap import (
    before_forward,
    enter_comm,
    enter_compute,
    overlap_fw_bw,
)


def test_fwbw_overlap() -> None:
    class _JournalizedFunc(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: Any, journal: list[str], name: str, tensor: torch.Tensor
        ) -> Any:
            ctx.journal, ctx.name = journal, name
            journal.append(f"{name}_F")
            return tensor

        @staticmethod
        def backward(ctx: Any, *gtensors) -> Any:
            ctx.journal.append(f"{ctx.name}_B")
            return None, None, *gtensors

    def journalized_fn(d: list[str], n: str, x: torch.Tensor) -> torch.Tensor:
        return _JournalizedFunc.apply(d, n, x)  # type: ignore

    journal: list = []

    def f(x: torch.Tensor) -> torch.Tensor:
        x = x @ w1
        x = journalized_fn(journal, "compute1", x)

        compute_end_event, x = enter_comm(x, name="comm1")  # type: ignore
        x = journalized_fn(journal, "comm1", x)
        x = enter_compute(compute_end_event, x, name="compute2")

        x = journalized_fn(journal, "compute2", x)
        x = x @ w2

        compute_end_event, x = enter_comm(x, name="comm2")  # type: ignore
        x = journalized_fn(journal, "comm2", x)
        x = enter_compute(compute_end_event, x, name="end")
        return x

    w1 = torch.randn([128, 128], device="cuda", requires_grad=True)
    w2 = torch.randn([128, 128], device="cuda", requires_grad=True)
    x = torch.randn([128, 128], device="cuda", requires_grad=True)
    gy = torch.randn_like(x)

    # Disable everything
    before_forward(False)
    y = f(x)
    y.backward(gy)
    ref = dict(dw1=w1.grad, dw2=w2.grad, dx=x.grad)
    w1.grad = w2.grad = x.grad = None

    # Simple FW+BW
    # enable FWBW overlap, but don't actually use it
    before_forward(True)
    y = f(x)
    y.backward(gy)
    assert torch.allclose(ref["dx"], x.grad)  # type: ignore
    assert torch.allclose(ref["dw1"], w1.grad)  # type: ignore
    assert torch.allclose(ref["dw2"], w2.grad)  # type: ignore
    w1.grad = w2.grad = x.grad = None

    # Overlapped FW+BW
    before_forward(True)
    # warmup
    y = f(x)
    assert y is not None
    assert y.requires_grad
    # overlapped - BW first
    journal.clear()
    y = overlap_fw_bw(lambda: f(x), lambda: y.backward(gy), initial_bw_chunks=1)
    assert journal == [
        "compute1_F",
        "comm2_B",
        "comm1_F",
        "compute2_B",
        "compute2_F",
        "comm1_B",
        "comm2_F",
        "compute1_B",
    ]
    # overlapped - FW first
    journal.clear()
    y = overlap_fw_bw(lambda: f(x), lambda: y.backward(gy), initial_bw_chunks=0)
    assert journal == [
        "compute1_F",
        "comm1_F",
        "comm2_B",
        "compute2_F",
        "compute2_B",
        "comm2_F",
        "comm1_B",
        "compute1_B",
    ]
    # cooldown
    y.backward(gy)
    assert torch.allclose(3 * ref["dx"], x.grad)  # type: ignore
    assert torch.allclose(3 * ref["dw1"], w1.grad)  # type: ignore
    assert torch.allclose(3 * ref["dw2"], w2.grad)  # type: ignore


def test_fwbw_nothing_to_overlap() -> None:
    def f(x: torch.Tensor) -> torch.Tensor:
        x = x * x
        return x

    x = torch.randn([128], device="cuda", requires_grad=True)
    gy = torch.randn([128], device="cuda")

    before_forward(True)
    y = f(x)
    y = overlap_fw_bw(lambda: f(x), lambda: y.backward(gy), initial_bw_chunks=1)
    y = overlap_fw_bw(lambda: f(x), lambda: y.backward(gy), initial_bw_chunks=0)


class ExceptionInBW(Exception):
    pass


class ExceptionInBWOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        return x

    @staticmethod
    def backward(ctx: Any, gx: torch.Tensor) -> torch.Tensor:  # type: ignore
        raise ExceptionInBW()


def test_exception_in_bw_pass() -> None:
    def f(x: torch.Tensor) -> torch.Tensor:
        x = x * x
        compute_end_event, x = enter_comm(x, name="comm1")  # type: ignore
        x = ExceptionInBWOp.apply(x)  # type: ignore
        x = enter_compute(compute_end_event, x, name="compute2")
        return x * x

    x = torch.randn([128], device="cuda", requires_grad=True)
    gy = torch.randn([128], device="cuda")

    before_forward(True)
    y = f(x)
    with pytest.raises(ExceptionInBW):
        overlap_fw_bw(lambda: f(x), lambda: y.backward(gy), initial_bw_chunks=1)
    y = f(x)
    with pytest.raises(ExceptionInBW):
        overlap_fw_bw(lambda: f(x), lambda: y.backward(gy), initial_bw_chunks=0)


def test_exception_in_first_bw_pass() -> None:
    def f(x: torch.Tensor) -> torch.Tensor:
        x = x * x
        _, x = enter_comm(x, name="comm1")  # type: ignore
        return ExceptionInBWOp.apply(x)  # type: ignore

    x = torch.randn([128], device="cuda", requires_grad=True)
    gy = torch.randn([128], device="cuda")

    before_forward(True)
    y = f(x)
    with pytest.raises(ExceptionInBW):
        overlap_fw_bw(lambda: f(x), lambda: y.backward(gy), initial_bw_chunks=0)
