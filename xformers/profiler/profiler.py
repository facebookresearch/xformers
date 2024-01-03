# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import queue
import socket
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import torch.cuda.memory
import torch.cuda.nvtx
import torch.nn as nn
import torch.profiler
import torch.utils.hooks

logger = logging.getLogger(__name__)


def _normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x


class NsightProfiler:
    """Profiler that triggers start of NSight profiler.

    NOTE: you need to ensure that the script running this code actually is running with
    ``nsys profile`` and also has a flag ``--capture-range=cudaProfilerApi`` so the
    capturing is performed by this profiler during certain steps.
    """

    def __init__(self, main_profiler: "_Profiler") -> None:
        self.main_profiler = main_profiler
        # TODO figure out if there is a way to know if nsys is launched at this point

    def __enter__(self):
        self.main_profiler._install_hooks()
        torch.cuda.profiler.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.profiler.stop()
        self.main_profiler._remove_hooks()

    def step(self) -> None:
        pass


class PyTorchProfiler:
    """Profiler which relies on native Pytorch profiling. Current setting of the profiler
    captures traces, memory footprint and other info that could be read via TensorBoard.
    """

    ACTIVITIES = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]

    def __init__(self, main_profiler: "_Profiler") -> None:
        self.main_profiler = main_profiler
        activities_str = "_".join(a.name for a in self.ACTIVITIES)
        trace_handler = torch.profiler.tensorboard_trace_handler(
            dir_name=str(
                main_profiler.output_dir
                / f"profile_{activities_str}_{main_profiler.done_steps:06}"
            ),
            worker_name=main_profiler.worker_name,
            use_gzip=True,
        )
        self.hta = torch.profiler.profile(
            on_trace_ready=trace_handler,
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
            activities=self.ACTIVITIES,
        )
        self.done_steps = 0

    def __enter__(self):
        torch.cuda.synchronize()
        self.hta.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize()
        self.hta.__exit__(exc_type, exc_val, exc_tb)

    def step(self) -> None:
        self.hta.step()
        self.done_steps += 1


class PyTorchProfiler_CUDAOnly(PyTorchProfiler):
    # This profiler does not profile the CPU-side of things
    # so we expect it to have almost no overhead
    ACTIVITIES = [torch.profiler.ProfilerActivity.CUDA]


class MemSnapshotsProfiler:
    """Profiler that captures memory traces for allocation and deallocation of memory for
    tensors.
    """

    def __init__(self, main_profiler: "_Profiler") -> None:
        self.main_profiler = main_profiler
        self.enabled = False

    @property
    def _has_trace_plot(self) -> bool:
        return hasattr(torch.cuda._memory_viz, "trace_plot")

    def __enter__(self):
        if not self._has_trace_plot:
            return
        self.enabled = True
        # TODO: This does not show the previous memory allocations
        # We could at least have a placeholder with how much
        # memory was allocated before
        torch.cuda.memory._record_memory_history(
            True,
            # keep 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,
            # record stack information for the trace events
            trace_alloc_record_context=True,
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._has_trace_plot:
            self.main_profiler.summary.append(
                ("MemTrace", "(not available with your Pytorch version)")
            )
            return
        assert self.enabled
        snapshot = torch.cuda.memory._snapshot()
        torch.cuda.memory._record_memory_history(False)
        # No data was recorded - avoids a `ValueError` in `trace_plot`
        if all(len(t) == 0 for t in snapshot["device_traces"]):
            self.main_profiler.summary.append(("MemTrace", "(no allocation recorded)"))
            return
        # Dump to disk
        filename = self.main_profiler._create_output_filename("memory_trace_plot.html")
        self.main_profiler.summary.append(("MemTrace", filename))
        with open(filename, "w+") as fd:
            fd.write(
                torch.cuda._memory_viz.trace_plot(
                    snapshot, device=None, plot_segments=False
                )
            )

    def step(self) -> None:
        pass


@dataclass
class _ProfilerState:
    cls: Any
    iter_begin: int
    iter_end: int
    object: Any = None


class _Profiler:
    _CURRENT_PROFILER = None

    def __init__(
        self,
        output_dir: str,
        schedule: Sequence[Tuple[Any, int, int]],
        module: Optional[nn.Module],
    ) -> None:
        self.check_schedule(schedule)
        self.done_steps = 0
        self.output_dir = Path(output_dir).absolute()
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.worker_name = ""
        if torch.distributed.is_initialized():
            self.worker_name = "{}_{}".format(socket.gethostname(), str(os.getpid()))

        self.module = weakref.ref(module if module is not None else nn.Module())
        self.parents = ["Global"]
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.hooks_refcount = 0
        self.profilers: List[_ProfilerState] = sorted(
            [_ProfilerState(cls, begin, end) for cls, begin, end in schedule],
            key=lambda x: x.iter_begin,
        )
        self.last_step = self.profilers[-1].iter_end if self.profilers else 0
        self.summary: List[Tuple[str, str]] = []

    def check_schedule(self, schedule: Sequence[Tuple[Any, int, int]]) -> None:
        if len(schedule) == 0:
            logger.warning(
                "You specified empty schedule for profiling. No data will be captured."
            )

        pq: Any = queue.PriorityQueue()
        for cls, begin, end in schedule:
            assert (
                begin >= 0
            ), f"Begin step of profiler must be non-negative, found: {begin}"
            assert end > 0, f"End step of profiler must be positive, found: {end}"
            assert (
                begin < end
            ), f"Start must be before the end, found: begin={begin} and end={end}"

            pq.put((begin, end))

        prev_end = -1
        for begin, end in pq.queue:
            assert begin >= prev_end, (
                "There is some overlapping in profiler scheduling. Please do not"
                + " overlap profilers by step as they may affect each other. Schedule:"
                + f" {schedule}"
            )
            prev_end = end

    def update_profilers_on_step(self) -> None:
        for p in self.profilers:
            if p.iter_begin <= self.done_steps and self.done_steps < p.iter_end:
                if p.object is None:
                    o = p.cls(self)
                    logging.info(f"Starting {p.cls.__name__} profiler...")
                    o.__enter__()
                    p.object = o
                else:
                    p.object.step()
            else:
                if p.object is not None:
                    o = p.object
                    p.object = None
                    logging.info(f"Shutting down {p.cls.__name__} profiler...")
                    o.__exit__(None, None, None)

    def _create_output_filename(self, filename: str) -> Path:
        """
        Returns where to write a file with desired filename.
        Handles the case where we are in distributed settings, or when
        we need to output the same file multiple times (eg if a profiler
        runs for several steps)
        """
        if self.worker_name != "":
            file = Path(filename)
            folder = self.output_dir / file.stem
            folder.mkdir(parents=True, exist_ok=True)
            return folder / f"{self.done_steps:06}_{self.worker_name}{file.suffix}"
        return self.output_dir / f"{self.done_steps:06}_{filename}"

    def _install_hooks(self) -> None:
        self.hooks_refcount += 1
        # Already installed
        if self.hooks:
            return
        module = self.module()
        if module is None:
            return
        for name, sub_mod in module.named_modules():
            if name == "":
                continue
            name = name.split(".")[-1]
            self.hooks += [
                sub_mod.register_forward_pre_hook(self._enter_module_hook(name)),
                sub_mod.register_forward_hook(self._exit_module_hook(name)),
            ]

    def _remove_hooks(self) -> None:
        self.hooks_refcount -= 1
        if self.hooks_refcount == 0:
            for h in self.hooks:
                h.remove()

    def _enter_module_hook(self, name):
        class PopState(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                if len(args) == 1:
                    return args[0]
                return args

            @staticmethod
            def backward(ctx, *grad_outs):
                self._exit_module(name)
                return grad_outs

        def f(module, inputs):
            self._enter_module(name)
            inputs = _normalize_tuple(inputs)
            out = PopState.apply(*inputs)
            return out

        return f

    def _exit_module_hook(self, name):
        class PushState(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                if len(args) == 1:
                    return args[0]
                return args

            @staticmethod
            def backward(ctx, *grad_outs):
                self._enter_module(name)
                return grad_outs

        def f(module, inputs, outputs):
            self._exit_module(name)
            outputs = _normalize_tuple(outputs)
            return PushState.apply(*outputs)

        return f

    def _enter_module(self, name) -> None:
        self.parents.append(name)
        torch.cuda.nvtx.range_push(name)

    def _exit_module(self, name) -> None:
        torch.cuda.nvtx.range_pop()
        assert self.parents[-1] == name
        self.parents.pop()

    def start(self):
        self.__enter__()

    def stop(self, exc_type=None, exc_val=None, exc_tb=None):
        self.__exit__(exc_type, exc_val, exc_tb)

    def __enter__(self):
        if _Profiler._CURRENT_PROFILER is not None:
            raise ValueError("Only one xformers profiler can be active at a time")
        _Profiler._CURRENT_PROFILER = self
        self.update_profilers_on_step()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _Profiler._CURRENT_PROFILER = None

        for p in self.profilers:
            if p.object is not None:
                p.object.__exit__(exc_type, exc_val, exc_tb)

    def step(self) -> None:
        """Signals the profiler that the next profiling step has started."""
        self.done_steps += 1

        if self.done_steps <= self.last_step:
            self.parents = ["Global"]
            self.update_profilers_on_step()
        if self.done_steps == self.last_step:
            logger.info("xFormers profiler done. %s", self.format_summary())

    def format_summary(self) -> str:
        if len(self.summary) == 0:
            return ""
        pad_titles = max(len(title) for title, value in self.summary)
        return "summary:\n" + "\n".join(
            [f"  {title.ljust(pad_titles)}: {value}" for title, value in self.summary]
        )
