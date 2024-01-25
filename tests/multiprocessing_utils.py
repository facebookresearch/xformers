# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import concurrent.futures
import multiprocessing
import signal
import tempfile
from typing import List

import torch


class SafeMpContext:
    def __init__(self) -> None:
        self.mp_context = multiprocessing.get_context("spawn")
        self.processes: List[multiprocessing.context.SpawnProcess] = []

    def Process(self, *args, **kwargs) -> multiprocessing.context.SpawnProcess:
        p = self.mp_context.Process(*args, **kwargs)
        p.daemon = True
        self.processes.append(p)
        return p

    def kill_all_processes(self):
        for p in self.processes:
            p.terminate()
            p.join(1)
            if p.exitcode is None:
                p.kill()
                p.join()

    def log_bad_exit_codes(self):
        for rank, p in enumerate(self.processes):
            if p.exitcode == 0:
                continue
            if p.exitcode < 0:
                try:
                    signal_desc = f" (signal {signal.Signals(-p.exitcode).name})"
                except ValueError:
                    signal_desc = " (unrecognized signal)"
            else:
                signal_desc = ""
            print(
                f"Child process for rank #{rank} with PID {p.pid} exited with code {p.exitcode}{signal_desc}"
            )

    def __getattr__(self, name: str):
        return getattr(self.mp_context, name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.kill_all_processes()
        self.log_bad_exit_codes()


def _launch_subprocesses_fn_wrapper(
    init_method: str, rank: int, world_size: int, user_fn, args, kwargs
):
    torch._C._set_print_stack_traces_on_fatal_signal(True)

    if torch.cuda.device_count() >= world_size:
        backend = "nccl"
        torch.cuda.set_device(rank)
    else:
        # Use Gloo instead of NCCL so that we can run on a single GPU
        backend = "gloo"
    torch.distributed.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
        init_method=init_method,
    )
    return user_fn(*args, **kwargs)


def launch_subprocesses(world_size: int, fn, *args, **kwargs):
    with SafeMpContext() as mp_context, concurrent.futures.ProcessPoolExecutor(
        max_workers=world_size, mp_context=mp_context
    ) as e, tempfile.NamedTemporaryFile(mode="w+b", buffering=-1, delete=True) as rdv:
        futures = [
            e.submit(
                _launch_subprocesses_fn_wrapper,
                init_method=f"file://{rdv.name}",
                rank=rank,
                world_size=world_size,
                user_fn=fn,
                args=args,
                kwargs=kwargs,
            )
            for rank in range(world_size)
        ]
        done, _ = concurrent.futures.wait(
            futures, return_when=concurrent.futures.FIRST_EXCEPTION
        )
        for f in done:
            f.result()
