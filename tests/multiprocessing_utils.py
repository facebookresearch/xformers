# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import concurrent
import gc
import multiprocessing
import os
import signal
from tempfile import _TemporaryFileWrapper, NamedTemporaryFile
from typing import Dict, List, Tuple

import torch


class SafeMpContext(multiprocessing.context.BaseContext):
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

            # (https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Process.exitcode)
            # Even though the python documentation seems to say that after joining the exitcode should
            # become set, this is not what we have observed in practice. We therefore loop until it
            # becomes set.
            while p.exitcode is None:
                p.kill()
                p.join()

            assert p.exitcode is not None, f"{p} is still alive"

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


def init_process_group(init_method: str, rank: int, world_size: int):
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


def _launch_subprocesses_fn_wrapper(
    init_method: str,
    rank: int,
    world_size: int,
    parent_env_vars: Dict[str, str],
    user_fn,
    args,
    kwargs,
):
    # This function initializes the environment for spawned subprocesses by capturing and applying the current
    # environment variables from the parent process. By clearing and then updating `os.environ` with `parent_env_vars`,
    # we ensure that each spawned subprocess starts with an environment that mirrors the parent process at the time
    # of job submission. This approach guarantees consistency across subprocesses, reflecting the latest state of the
    # parent's environment variables even/especially when reusing the subprocesses for subsequent job executions.
    os.environ.clear()
    os.environ.update(parent_env_vars)

    # Check if the process group is already initialized
    if not torch.distributed.is_initialized():
        init_process_group(init_method, rank, world_size)
    try:
        return user_fn(*args, **kwargs)
    finally:
        # should free all memory used by PyTorch in the subprocesses
        gc.collect()
        torch.cuda.empty_cache()


# Global dictionary to keep track of executors and temporary files
EXECUTORS_AND_FILES: Dict[
    int, Tuple[_TemporaryFileWrapper, concurrent.futures.ProcessPoolExecutor]
] = {}


def get_global_pool_allocator(
    world_size: int,
) -> Tuple[_TemporaryFileWrapper, concurrent.futures.ProcessPoolExecutor]:
    global EXECUTORS_AND_FILES

    if world_size not in EXECUTORS_AND_FILES:
        rdv = NamedTemporaryFile(mode="w+b", buffering=-1, delete=False)
        mp_context = SafeMpContext()

        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=world_size, mp_context=mp_context
        )

        # Add the executor and temporary file to the global list
        EXECUTORS_AND_FILES[world_size] = (rdv, executor)
    else:
        rdv, executor = EXECUTORS_AND_FILES[world_size]

    return rdv, executor


class ProcessPoolExecutorManager:
    def __init__(self, world_size: int):
        self.world_size = world_size

    def __enter__(self):
        # when you start a subprocess you want to free memory used by PyTorch in the main process,
        # so the subprocess can have memory
        gc.collect()
        torch.cuda.empty_cache()

        self.rdv, self.executor = get_global_pool_allocator(self.world_size)
        return self

    def submit(self, fn, *args, **kwargs):
        return self.executor.submit(fn, *args, **kwargs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # One of the subprocesses jobs has failed
        if exc_val:
            # We want to avoid killing the processes while the executor was thinking that they were
            # still up and healthy (as this may have unintended consequences, such as the executor
            # restarting the processes, or reporting spurious errors).
            # Set the internal state of the executor and call cancel() on each issued task that is
            # not executing
            self.executor.shutdown(wait=False, cancel_futures=True)

            # Kill all remaining subprocesses
            mp_context = self.executor._mp_context
            mp_context.kill_all_processes()
            mp_context.log_bad_exit_codes()

            # We want to wait for all the futures to complete, so we need to shutdown twice
            self.executor.shutdown(wait=True)

            # Close the temporary file
            self.rdv.close()

            # Remove the executor from the global list.
            # This will recreate it next time a test is requiring this world_size
            assert self.world_size in EXECUTORS_AND_FILES
            del EXECUTORS_AND_FILES[self.world_size]

            print(
                f"Shutdown and remove the executor after subprocesses error. Executors cnt: {len(EXECUTORS_AND_FILES)}"
            )


def launch_subprocesses(world_size: int, fn, *args, **kwargs):
    # This custom manager allows each test execution to enter/exit the following context.
    # When entering the context, it creates/reuses a new/existing ProcessPoolExecutor with the given world size.
    # The context also allows to detect an exception upon exit, in which case it will kill all spawned processes,
    # delete the manager, recreate the manager upon following request and respawn processes.
    with ProcessPoolExecutorManager(world_size) as manager:
        futures = [
            manager.submit(
                _launch_subprocesses_fn_wrapper,
                init_method=f"file://{manager.rdv.name}",
                rank=rank,
                world_size=world_size,
                parent_env_vars=dict(os.environ),
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
