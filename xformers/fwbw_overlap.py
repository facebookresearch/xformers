# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import atexit
import logging
import os
import queue
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast, ClassVar, overload, TypeVar, Union

import torch
from typing_extensions import Unpack

try:
    from deep_ep.utils import EventHandle, EventOverlap  # type: ignore
except ImportError:
    # If DeepEP is not available, recreate ourself those structures
    class EventHandle:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self._event = torch.cuda.Event()

        def current_stream_wait(self) -> None:
            self._event.wait()

    class EventOverlap:  # type: ignore[no-redef]
        def __init__(self, event: Union[EventHandle, None] = None) -> None:
            self.event = event

        def current_stream_wait(self) -> None:
            assert self.event is not None
            self.event.current_stream_wait()


logger = logging.getLogger(__name__)


class EventOverlapHolder(torch.Tensor):
    """
    Holds a CUDAEvent. Why does it need to be a tensor?
    So that its `gradient` can also hold a CUDAEvent for
    overlaps in the BW pass
    """

    event_overlap: Union[EventOverlap, None]
    _name: str
    __slots__: list[str] = ["event_overlap", "_name"]

    @classmethod
    def capture(
        cls,
        device: torch.device,
        name: str = "",
    ) -> "EventOverlapHolder":
        return EventOverlapHolder(
            EventOverlap(EventHandle()),
            device=device,
            name=name,
            requires_grad=False,
        )

    @staticmethod
    def __new__(
        cls: "type[EventOverlapHolder]",
        event_overlap: Union[EventOverlap, None],
        device: torch.device,
        name: str = "",
        requires_grad: bool = True,
    ):
        return torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            size=[0],
            device=device,
            dtype=torch.float32,
            requires_grad=requires_grad,
        )

    def __init__(
        self,
        event_overlap: Union[EventOverlap, None],
        device: torch.device,
        name: str = "",
        requires_grad: bool = True,
    ):
        super().__init__()
        self.event_overlap = event_overlap
        self._name = name

    def __tensor_flatten__(self):
        return self.__slots__, ()

    def __repr__(self) -> str:  # type: ignore
        return f"{self.__class__.__name__}({self._name})"

    def current_stream_wait(self) -> None:
        if self.event_overlap is not None:
            self.event_overlap.current_stream_wait()

    __torch_function__ = torch._C._disabled_torch_function_impl  # type: ignore

    @classmethod
    def __torch_dispatch__(
        cls, func: Any, types: Any, args: Any = (), kwargs: Any = None
    ) -> Any:
        if func._overloadpacket in [
            torch.ops.aten.detach_,
            torch.ops.aten.detach,
            torch.ops.aten._to_copy,
            torch.ops.aten.view,
            torch.ops.aten._unsafe_view,
        ]:
            self = args[0]
            assert isinstance(self, EventOverlapHolder)
            return EventOverlapHolder(
                self.event_overlap,
                device=self.device,
                name=self._name,
                requires_grad=False,
            )
        if func._overloadpacket is torch.ops.aten.new_empty_strided:
            self, shape, strides = args
            return EventOverlapHolder(
                None,
                device=self.device,
                name="new_empty_strided",
                requires_grad=False,
            )

        raise NotImplementedError(f"{cls.__name__} does not support {func}")


class _ExitCompute(torch.autograd.Function):
    """
    Execution order:
    [compute]
    [exit_compute] <- here
      [bw chunk]
    [comms]
    """

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, *tensors: torch.Tensor):
        device = next(x.device for x in tensors if x is not None)
        holder = EventOverlapHolder.capture(device=device, name="exit_compute")
        ctx.set_materialize_grads(False)
        return (holder, *tensors)

    @staticmethod
    def backward(  # type: ignore
        ctx: torch.autograd.function.FunctionCtx,
        gholder: Union[EventOverlapHolder, None],
        *gtensors: torch.Tensor,
    ):
        # wait for comms to finish before doing the compute
        if gholder is not None:
            assert isinstance(gholder, EventOverlapHolder)
            gholder.current_stream_wait()
        return gtensors


class _EnterCompute(torch.autograd.Function):
    """
    Execution order:
    [comms]
      [bw chunk]
    [enter_compute] <- here
    [compute]
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        holder: EventOverlapHolder,
        *tensors: torch.Tensor,
    ):
        holder.current_stream_wait()
        ctx.set_materialize_grads(False)
        ctx.event_overlap_requires_grad = holder.requires_grad  # type: ignore
        if len(tensors) == 1:
            return tensors[0]
        return tensors

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, *gtensors: torch.Tensor):
        if ctx.event_overlap_requires_grad:  # type: ignore
            device = next(x.device for x in gtensors if x is not None)
            gholder = EventOverlapHolder.capture(device=device, name="enter_compute(B)")
        else:
            gholder = None
        return gholder, *gtensors


class _FillGradientForOverlapHolder(torch.autograd.Function):
    """
    If the OverlapHolder is not used (eg with 1 GPU) then it does
    not get any gradient, and the BW pass is stuck. This ensures it does
    not happen.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        holder: EventOverlapHolder,
        *tensors: torch.Tensor,
    ):
        assert isinstance(holder, EventOverlapHolder)
        holder = EventOverlapHolder(
            holder.event_overlap, device=holder.device, name=holder._name
        )
        ctx.set_materialize_grads(False)
        ctx.already_called = False  # type: ignore
        return holder, *tensors

    @staticmethod
    def backward(  # type: ignore
        ctx: torch.autograd.function.FunctionCtx,
        gholder: Union[EventOverlapHolder, None],
        *gtensors: torch.Tensor,
    ):
        assert not ctx.already_called, (  # type: ignore
            "this BW pass got called multiple times. This is "
            "most likely a *very nasty* bug with the FW+BW overlap"
        )
        ctx.already_called = True  # type: ignore

        if gholder is None:
            device = next(x.device for x in gtensors if x is not None)
            gholder = EventOverlapHolder(
                None,
                device=device,
                name="_FillGradientForOverlapHolder",
                requires_grad=False,
            )
        return gholder, *gtensors


def enter_comm(
    *tensors: torch.Tensor,
    name: str = "comm",
) -> tuple[EventOverlapHolder, Unpack[tuple[torch.Tensor, ...]]]:
    assert all(isinstance(x, torch.Tensor) for x in tensors)
    tensors = _ExitCompute.apply(*tensors)  # type: ignore
    tensors = enter_phase(name, *tensors)
    return _FillGradientForOverlapHolder.apply(*tensors)  # type: ignore


@overload
def enter_compute(
    __overlap_holder: EventOverlapHolder,
    __tensor0: torch.Tensor,
    __tensor1: torch.Tensor,
    *tensors: torch.Tensor,
    name: str = "compute",
) -> tuple[torch.Tensor, ...]:
    pass


@overload
def enter_compute(
    overlap_holder: EventOverlapHolder,
    tensor: torch.Tensor,
    *,
    name: str = "compute",
) -> torch.Tensor:
    pass


def enter_compute(  # type: ignore
    overlap_holder: EventOverlapHolder,
    *tensors: torch.Tensor,
    name: str = "compute",
) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
    assert all(isinstance(x, torch.Tensor) for x in tensors)
    # Ensure we have at least one op
    overlap_holder = overlap_holder.view_as(overlap_holder)  # type: ignore
    overlap_holder, *compute_tensors = enter_phase(name, overlap_holder, *tensors)  # type: ignore
    return _EnterCompute.apply(overlap_holder, *compute_tensors)  # type: ignore


@dataclass
class PhaseBoundary:
    fw_enter: str
    arrived_sem: threading.BoundedSemaphore
    unblock_sem: threading.BoundedSemaphore
    fw_previous_boundary: "Union[PhaseBoundary, None]"
    is_final: bool = True

    def __post_init__(self) -> None:
        if self.fw_previous_boundary is not None:
            self.fw_previous_boundary.is_final = False

    def __str__(self) -> str:
        prev_name = "None"
        if self.fw_previous_boundary is not None:
            prev_name = self.fw_previous_boundary.fw_enter
        return f"PhaseBoundary({prev_name} --> {self.fw_enter})"

    def __call__(self) -> None:
        with _CurrentForwardState.cv_on_bw:
            self.unblock_sem.release()
            _CurrentForwardState.cv_on_bw.wait()
        if _CurrentForwardState.bw_exception is not None:
            raise _CurrentForwardState.bw_exception
        elif self.fw_previous_boundary is not None:
            # wait for autograd to finish scheduling the next chunk
            if not self.fw_previous_boundary.arrived_sem.acquire(timeout=3800):
                raise RuntimeError(
                    f"FWBW overlap: autograd did not go from {self.fw_enter}-->{self.fw_previous_boundary.fw_enter}. "
                    "Did you make sure to call `before_forward` before every forward?"
                )
        else:
            # is there is no previous one, wait for autograd to finish entirely
            assert _CurrentForwardState.bw_done_semaphore is not None
            if not _CurrentForwardState.bw_done_semaphore.acquire(timeout=3800):
                raise RuntimeError(
                    f"FWBW overlap: autograd did not finish after crossing {self.fw_enter} in BW pass."
                    "Did you make sure to call `before_forward` before every forward?"
                )
        _CurrentForwardState.bw_last_boundary = self.fw_previous_boundary


class InitialBw:
    def __init__(self, trigger_bw: Callable[[], None]) -> None:
        self.trigger_bw = trigger_bw

    def __call__(self) -> None:
        _CurrentForwardState.bw_chunks_wait = True

        # trigger first BW chunk
        # + retrieve the first boundary in the BW pass
        with _CurrentForwardState.cv_on_bw:
            _CurrentForwardState.bw_last_boundary = None
            _CurrentForwardState.bw_done_semaphore = async_bw(self.trigger_bw)
            del self.trigger_bw
            _CurrentForwardState.cv_on_bw.wait()
            if _CurrentForwardState.bw_exception is not None:
                exception = _CurrentForwardState.bw_exception
                _CurrentForwardState.bw_exception = None
                raise exception


class _GlobalAutogradThread:
    thread: Union[threading.Thread, None] = None
    todo = queue.SimpleQueue()  # type: ignore
    # how many free BW slots we have
    # 0 if we are currently running a BW pass
    sem = threading.BoundedSemaphore()

    @classmethod
    def run(cls) -> None:
        while True:
            bw_fn, release_when_done = cls.todo.get()
            if bw_fn is None:  # exit signal
                return
            try:
                with cls.sem:
                    bw_fn()
            except Exception as exc:
                traceback.print_exc()
                _CurrentForwardState.bw_exception = exc
            finally:
                del bw_fn
                release_when_done.release()
                # edge case: if there is only a single chunk
                # in the BW pass. We need to wake-up the thread
                # waiting for the first chunk
                # If there are multiple chunks, `cv_on_bw` will be
                # already notified and be `None`
                with _CurrentForwardState.cv_on_bw:
                    _CurrentForwardState.bw_last_boundary = None
                    _CurrentForwardState.cv_on_bw.notify()

    @classmethod
    def cleanup_at_exit(cls) -> None:
        logger.info("Shutting down FWBW overlap background thread")
        cls.todo.put((None, None))
        if cls.thread is not None:
            cls.thread.join()
            cls.thread = None


def async_bw(backward_fn: Callable[[], None]) -> threading.Semaphore:
    """
    You can wait for the backward to finish with `done_semaphore.acquire()`
    """
    done_semaphore = threading.BoundedSemaphore()
    done_semaphore.acquire()
    _GlobalAutogradThread.todo.put((backward_fn, done_semaphore), block=False)
    return done_semaphore


class _WaitInBW(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        boundary: PhaseBoundary,
        *x: torch.Tensor,
    ) -> Any:
        ctx.boundary = boundary  # type: ignore
        ctx.backward_done = False  # type: ignore
        if len(x) == 1:
            return x[0]
        return x

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, *gx: torch.Tensor) -> Any:
        assert not ctx.backward_done  # type: ignore
        ctx.backward_done = True  # type: ignore

        if not _CurrentForwardState.bw_chunks_wait:
            return None, *gx

        boundary = cast(PhaseBoundary, ctx.boundary)  # type: ignore
        boundary.arrived_sem.release()

        with _CurrentForwardState.cv_on_bw:
            _CurrentForwardState.bw_last_boundary = boundary
            _CurrentForwardState.cv_on_bw.notify()

        if not boundary.unblock_sem.acquire(timeout=3800):
            raise RuntimeError(
                f"{boundary.fw_enter}: timed-out: unable to acquire semaphore to continue BW pass"
            )
        return None, *gx


class _CurrentForwardState:
    # if `True`, we record boundaries between chunks during the FW pass
    record_fw_chunks: ClassVar[bool] = False
    # if `True`, we wait in the BW pass
    bw_chunks_wait: ClassVar[bool] = False
    fw_previous_boundary: ClassVar[Union[PhaseBoundary, None]] = None
    bw_last_boundary: ClassVar[Union[PhaseBoundary, Callable[[], None], None]] = None
    bw_done_semaphore: ClassVar[Union[threading.Semaphore, None]] = None
    bw_exception: ClassVar[Union[Exception, None]] = None
    # triggered whenever a BW chunk is done (or exception happens)
    cv_on_bw: ClassVar[threading.Condition] = threading.Condition()


def before_forward(record_fw_chunks: bool) -> None:
    """call this before entering a new FW pass"""
    _CurrentForwardState.record_fw_chunks = record_fw_chunks
    _CurrentForwardState.bw_chunks_wait = False
    _CurrentForwardState.fw_previous_boundary = None
    _CurrentForwardState.bw_last_boundary = None
    _CurrentForwardState.bw_done_semaphore = None
    _CurrentForwardState.bw_exception = None


def enter_phase(enter: str, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    "Marks the transition to either comms or compute in the FW pass"
    if not _CurrentForwardState.record_fw_chunks:
        return tensors
    assert any(x.grad_fn is not None for x in tensors), (
        f"Entering phase `{enter}` but there is no `grad_fn` "
        "on any of the input tensors. This can mean that there was no"
        " operator in the previous phase?"
    )
    flush_single_bw_chunk()

    boundary = PhaseBoundary(
        fw_enter=enter,
        arrived_sem=threading.BoundedSemaphore(),
        unblock_sem=threading.BoundedSemaphore(),
        fw_previous_boundary=_CurrentForwardState.fw_previous_boundary,
    )
    boundary.arrived_sem.acquire()
    boundary.unblock_sem.acquire()
    tensors_after = _WaitInBW.apply(boundary, *tensors)  # type: ignore
    _CurrentForwardState.fw_previous_boundary = boundary

    return tuple(tensors_after)  # type: ignore


def flush_single_bw_chunk() -> bool:
    last_b = _CurrentForwardState.bw_last_boundary
    if last_b is None:
        return False
    last_b()  # trigger it
    return True


def flush_pending_bw() -> None:
    "Flush all BW chunks that we can compute recursively"
    while flush_single_bw_chunk():
        pass


T = TypeVar("T")


def overlap_fw_bw(
    trigger_fw: Callable[[], T],
    trigger_bw: Callable[[], None],
    initial_bw_chunks: int = 0,
) -> T:
    try:
        return _overlap_fw_bw(trigger_fw, trigger_bw, initial_bw_chunks)
    finally:
        _CurrentForwardState.bw_chunks_wait = False


def _overlap_fw_bw(
    trigger_fw: Callable[[], T],
    trigger_bw: Callable[[], None],
    initial_bw_chunks: int = 0,
) -> T:
    if _GlobalAutogradThread.thread is None:
        atexit.register(_GlobalAutogradThread.cleanup_at_exit)
        _GlobalAutogradThread.thread = threading.Thread(
            target=_GlobalAutogradThread.run,
            name="overlap_fw_bw_autograd",
        )
        _GlobalAutogradThread.thread.daemon = True
        _GlobalAutogradThread.thread.start()
        logger.info(
            f"Launched background autograd thread for FW+BW overlap\n"
            f"  parent_pid = {os.getpid()}\n"
            f"  thread_pid = {_GlobalAutogradThread.thread.native_id}"
        )

    assert initial_bw_chunks >= 0
    before_forward(True)
    _CurrentForwardState.bw_last_boundary = InitialBw(trigger_bw)
    for _ in range(initial_bw_chunks):
        flush_single_bw_chunk()
    outputs = trigger_fw()
    while flush_single_bw_chunk():
        pass
    assert _CurrentForwardState.bw_last_boundary is None, "BW never finished?"
    return outputs
