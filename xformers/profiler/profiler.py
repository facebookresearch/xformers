# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import csv
import logging
import os
import queue
import socket
import time
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch.cuda.memory
import torch.cuda.nvtx
import torch.distributed as dist
import torch.nn as nn
import torch.profiler

from .device_limits import get_device_limits
from .profile_analyzer import AnalyzedTrace

logger = logging.getLogger(__name__)


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
        torch.cuda.profiler.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.profiler.stop()

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
        self.num_steps = 0
        self.pytorch_profiler = torch.profiler.profile(
            on_trace_ready=self._on_trace,
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
            with_flops=True,
            activities=self.ACTIVITIES,
        )

    def _on_trace(self, prof: torch.profiler.profiler.profile) -> None:
        activities_str = "_".join(a.name for a in self.ACTIVITIES)
        dir_name = str(
            self.main_profiler.output_dir
            / f"profile_{activities_str}_{self.main_profiler.done_steps:06}"
        )
        worker_name = self.main_profiler.worker_name
        if worker_name == "":
            worker_name = f"{socket.gethostname()}_{os.getpid()}"
            if dist.is_available() and dist.is_initialized():
                # Left-pad rank with zeros to make them all of the same length
                rank = f"{dist.get_rank()}".zfill(len(f"{dist.get_world_size() - 1}"))
                worker_name = f"rank{rank}_{worker_name}"
        os.makedirs(dir_name, exist_ok=True)
        file_name = f"{worker_name}.{time.time_ns()}.pt.trace.json.gz"
        prof.export_chrome_trace(os.path.join(dir_name, file_name))
        csv_file_name = f"kernels_{worker_name}.{time.time_ns()}.csv"
        self._preprocess_trace(prof, os.path.join(dir_name, csv_file_name))
        try:
            self._analyze_trace(prof)
        except Exception as exc:
            self.main_profiler.summary.append(("TraceAnalysis", "Error"))
            logger.warning("Exception analyzing kineto trace", exc_info=exc)

    def _preprocess_trace(
        self, prof: torch.profiler.profiler.profile, file_name: str
    ) -> None:
        if prof.profiler is None or prof.profiler.kineto_results is None:
            return
        with open(file_name, "w", newline="") as file:
            writer = csv.writer(file)
            for e in prof.profiler.kineto_results.events():
                if (
                    e.device_type().name == "CUDA"
                    and not e.is_user_annotation()
                    and e.duration_ns() > 0
                ):
                    writer.writerow([e.name(), f"{e.duration_ns() / 1_000}"])

    def _analyze_trace(self, prof: torch.profiler.profiler.profile) -> None:
        if prof.profiler is None or prof.profiler.kineto_results is None:
            return
        results = AnalyzedTrace.from_profile(prof.profiler.kineto_results.events())
        limits = get_device_limits(torch.device("cuda"))
        hw_flops: Dict[torch.dtype, float] = {}
        if limits is not None:
            for dtype, tflops in limits.gemm_tflops.items():
                hw_flops[dtype] = tflops * (1000**4)
        total_hfu = results.compute_hfu(hw_flops)
        total_mfu = results.compute_mfu(hw_flops)
        total_flop = sum(
            results.compute_num_ops(dtype)
            for dtype in results.operations_per_dtype_fw.keys()
        )
        s = self.main_profiler.summary
        s.append(
            ("Step time (ms)", f"{int(results.total_time_s * 1000 / self.num_steps)}")
        )
        s.append(("TFlop/step", f"{total_flop / (self.num_steps * 1000**4):0.1f}"))
        s.append(("TFlops", f"{total_flop / (results.total_time_s * 1000**4):0.1f}"))
        s.append(("HFU", f"{total_hfu:0.3f}"))
        s.append(("MFU", f"{total_mfu:0.3f}"))

    def __enter__(self):
        torch.cuda.synchronize()
        self.pytorch_profiler.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize()
        self.pytorch_profiler.__exit__(exc_type, exc_val, exc_tb)

    def step(self) -> None:
        self.pytorch_profiler.step()
        self.num_steps += 1


class PyTorchProfiler_CUDAOnly(PyTorchProfiler):
    # This profiler does not profile the CPU-side of things
    # so we expect it to have almost no overhead
    ACTIVITIES = [torch.profiler.ProfilerActivity.CUDA]

    def _analyze_trace(self, prof: torch.profiler.profiler.profile) -> None:
        # Can't analyze trace without CPU trace for operator shapes etc...
        pass


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
        self.schedule = schedule
        self.done_steps = 0
        self.output_dir = Path(output_dir).absolute()
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.worker_name = ""
        if dist.is_available() and dist.is_initialized():
            # Left-pad rank with zeros to make them all of the same length
            rank = f"{dist.get_rank()}".zfill(len(f"{dist.get_world_size() - 1}"))
            self.worker_name = f"rank{rank}_{socket.gethostname()}_{os.getpid()}"

        self.module = weakref.ref(module if module is not None else nn.Module())
        self.init_schedule()

    def init_schedule(self, offset: int = 0) -> None:
        self.profilers: List[_ProfilerState] = sorted(
            [
                _ProfilerState(cls, begin + offset, end + offset)
                for cls, begin, end in self.schedule
            ],
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
                    # Make sure the profiler's `step` function is called
                    # $N times when we do $N steps with this profiler.
                    o.step()
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
            self.update_profilers_on_step()
        if self.done_steps == self.last_step:
            logger.info("xFormers profiler done. %s", self.format_summary())

        # Check if we triggered a manual profile step
        CHECK_TRIGGER_EVERY = 10
        if (
            self.done_steps > self.last_step
            and (self.done_steps % CHECK_TRIGGER_EVERY) == 0
        ):
            try:
                (self.output_dir / "trigger").unlink()
                (
                    self.output_dir
                    / f"trigger.{self.done_steps + CHECK_TRIGGER_EVERY:09}"
                ).write_text(self.worker_name)
            except FileNotFoundError:
                pass
            step_trigger = self.output_dir / f"trigger.{self.done_steps:09}"
            if step_trigger.exists():
                logger.info(
                    "xFormers profiler manually triggered at step %d", self.done_steps
                )
                self.init_schedule(offset=self.done_steps + 1)

    def format_summary(self) -> str:
        if len(self.summary) == 0:
            return ""
        pad_titles = max(len(title) for title, value in self.summary)
        return "summary:\n" + "\n".join(
            [f"  {title.ljust(pad_titles)}: {value}" for title, value in self.summary]
        )
