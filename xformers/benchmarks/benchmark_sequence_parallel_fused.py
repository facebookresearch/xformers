# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import contextlib
import dataclasses
import enum
import multiprocessing
import os
import random
from collections import deque
from statistics import mean, stdev
from typing import Callable

import torch

# torch._C._set_print_stack_traces_on_fatal_signal(True)


@dataclasses.dataclass
class Scenario:
    # The number of tokens, i.e., the batch size times the sequence length
    num_samples: int
    # The per-sample features outside of the MHA/FFN block, and inside of it
    outer_dim: int
    inner_dim: int
    # Simulate this many matmuls during the all-gather step
    num_ag_matrices: int


class Step(enum.Enum):
    AllGather = "ag"
    ReduceScatter = "rs"

    def __str__(self):
        return self.value


@dataclasses.dataclass
class Bench:
    ag: Callable[[], None]
    rs: Callable[[], None]

    def __getitem__(self, step: Step):
        if step is Step.AllGather:
            return self.ag
        elif step is Step.ReduceScatter:
            return self.rs
        else:
            raise KeyError(f"{step}")


LLAMA_07B_SLEN = 4096
LLAMA_07B_D = 4096

LLAMA_70B_SLEN = 2048
LLAMA_70B_D = 8192


def round_up_to_nearest_multiple(n: int, m: int) -> int:
    return m * ((n + m - 1) // m)


def llama_07B_MHA(world_size: int) -> Scenario:
    batch_size = 8
    return Scenario(
        num_samples=batch_size * LLAMA_07B_SLEN,
        outer_dim=LLAMA_07B_D,
        inner_dim=LLAMA_07B_D // world_size,
        num_ag_matrices=3,
    )


def llama_07B_FFN(world_size: int) -> Scenario:
    batch_size = 8
    return Scenario(
        num_samples=batch_size * LLAMA_07B_SLEN,
        outer_dim=LLAMA_07B_D,
        inner_dim=round_up_to_nearest_multiple(2 * (4 * LLAMA_07B_D) // 3, 256)
        // world_size,
        num_ag_matrices=2,
    )


def llama_70B_MHA(world_size: int) -> Scenario:
    batch_size = world_size
    return Scenario(
        num_samples=batch_size * LLAMA_70B_SLEN,
        outer_dim=LLAMA_70B_D,
        inner_dim=LLAMA_70B_D // world_size,
        num_ag_matrices=3,
    )


def llama_70B_FFN(world_size: int) -> Scenario:
    batch_size = world_size
    return Scenario(
        num_samples=batch_size * LLAMA_70B_SLEN,
        outer_dim=LLAMA_70B_D,
        inner_dim=round_up_to_nearest_multiple(2 * (4 * LLAMA_70B_D) // 3, 256)
        // world_size,
        num_ag_matrices=2,
    )


SCENARIOS = {
    "llama_07B_MHA": llama_07B_MHA,
    "llama_07B_FFN": llama_07B_FFN,
    "llama_70B_MHA": llama_70B_MHA,
    "llama_70B_FFN": llama_70B_FFN,
}

DTYPES = {
    "bfloat16": torch.bfloat16,
}


def run_one_rank(
    my_rank,
    world_size,
    scenario_name,
    step,
    dtype_str,
    num_rounds,
    num_warmup_iters,
    num_bench_iters,
    profile,
    conn_from_prev,
    conn_to_next,
):
    print(f"RANK {my_rank} started")

    torch.cuda.set_device(my_rank)
    my_device = torch.device(f"cuda:{my_rank}")

    os.environ["RANK"] = f"{my_rank}"
    os.environ["WORLD_SIZE"] = f"{world_size}"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    torch.distributed.init_process_group(backend="nccl", init_method="env://")

    subgroup = torch.distributed.new_group()
    subgroup_nowait = torch.distributed.new_group()
    subgroup_nowait_nomemcpy = torch.distributed.new_group()

    scenario = SCENARIOS[scenario_name](world_size)
    if step is Step.AllGather:
        M = scenario.num_samples
        N = scenario.inner_dim
        K = scenario.outer_dim
        num_matrices = scenario.num_ag_matrices
    elif step is Step.ReduceScatter:
        M = scenario.num_samples
        N = scenario.outer_dim
        K = scenario.inner_dim
        num_matrices = 1

    dtype = DTYPES[dtype_str]

    scattered_input = torch.randn((M // world_size, K), dtype=dtype, device=my_device)
    gathered_input = torch.randn((M, K), dtype=dtype, device=my_device)
    weights = [
        torch.randn((K, N), dtype=dtype, device=my_device) for _ in range(num_matrices)
    ]
    gathered_outputs = [
        torch.randn((M, N), dtype=dtype, device=my_device) for _ in range(num_matrices)
    ]
    scattered_outputs = [
        torch.randn((M // world_size, N), dtype=dtype, device=my_device)
        for _ in range(num_matrices)
    ]

    gathered_outputs_nccl_reference = [
        torch.randn((M, N), dtype=dtype, device=my_device) for _ in range(num_matrices)
    ]
    gathered_outputs_fused = [
        torch.randn((M, N), dtype=dtype, device=my_device) for _ in range(num_matrices)
    ]
    scattered_outputs_nccl_reference = [
        torch.randn((M // world_size, N), dtype=dtype, device=my_device)
        for _ in range(num_matrices)
    ]
    scattered_outputs_fused = [
        torch.randn((M // world_size, N), dtype=dtype, device=my_device)
        for _ in range(num_matrices)
    ]

    def run_compute_lower_bound_ag():
        for w, go in zip(weights, gathered_outputs):
            torch.matmul(gathered_input, w, out=go)

    def run_compute_lower_bound_rs():
        for w, go, so in zip(weights, gathered_outputs, scattered_outputs):
            torch.matmul(gathered_input, w, out=go)
            torch.sum(go.view((world_size, M // world_size, N)), dim=0, out=so)

    def run_comms_lower_bound_ag():
        torch.distributed.all_gather_into_tensor(gathered_input, scattered_input)

    def run_comms_lower_bound_rs():
        for so, go in zip(scattered_outputs, gathered_outputs):
            torch.distributed.reduce_scatter_tensor(so, go)

    def run_nccl_reference_ag():
        torch.distributed.all_gather_into_tensor(gathered_input, scattered_input)
        for w, go in zip(weights, gathered_outputs_nccl_reference):
            torch.matmul(gathered_input, w, out=go)

    def run_nccl_reference_rs():
        for w, go, so in zip(
            weights, gathered_outputs, scattered_outputs_nccl_reference
        ):
            torch.matmul(gathered_input, w, out=go)
            torch.distributed.reduce_scatter_tensor(so, go)

    def run_fused_ag():
        nonlocal gathered_outputs_fused
        from xformers.ops import fused_allgather_and_linear

        gathered_outputs_fused = fused_allgather_and_linear(
            scattered_input,
            [w.t() for w in weights],
            group=subgroup,
            num_stripes=2,
            timeout_s=10,
        )

    def run_fused_rs():
        nonlocal scattered_outputs_fused
        from xformers.ops import fused_linear_and_reducescatter

        scattered_outputs_fused = fused_linear_and_reducescatter(
            gathered_input,
            [w.t() for w in weights],
            group=subgroup,
            num_stripes=2,
            timeout_s=10,
        )

    def run_fused_nowait_ag():
        nonlocal gathered_outputs_fused
        from xformers.ops import fused_allgather_and_linear

        gathered_outputs_fused = fused_allgather_and_linear(
            scattered_input,
            [w.t() for w in weights],
            group=subgroup_nowait,
            num_stripes=2,
            _wait=False,
            timeout_s=10,
        )

    def run_fused_nowait_rs():
        nonlocal scattered_outputs_fused
        from xformers.ops import fused_linear_and_reducescatter

        scattered_outputs_fused = fused_linear_and_reducescatter(
            gathered_input,
            [w.t() for w in weights],
            group=subgroup_nowait,
            num_stripes=2,
            _wait=False,
            timeout_s=10,
        )

    def run_fused_nowait_nomemcpy_ag():
        nonlocal gathered_outputs_fused
        from xformers.ops import fused_allgather_and_linear

        gathered_outputs_fused = fused_allgather_and_linear(
            scattered_input,
            [w.t() for w in weights],
            group=subgroup_nowait_nomemcpy,
            num_stripes=2,
            _wait=False,
            _memcpy=False,
            timeout_s=10,
        )

    def run_fused_nowait_nomemcpy_rs():
        nonlocal scattered_outputs_fused
        from xformers.ops import fused_linear_and_reducescatter

        scattered_outputs_fused = fused_linear_and_reducescatter(
            gathered_input,
            [w.t() for w in weights],
            group=subgroup_nowait_nomemcpy,
            num_stripes=2,
            _wait=False,
            _memcpy=False,
            timeout_s=10,
        )

    print(f"Sizes: ({world_size}x{M // world_size})x({num_matrices}x{N})x{K}")

    if step is Step.AllGather:
        run_nccl_reference_ag()
        run_fused_ag()
        if my_rank == 0:
            print("fused:")
            print(
                "Are equal? "
                + " ".join(
                    str(torch.equal(ref, fus))
                    for ref, fus in zip(
                        gathered_outputs_nccl_reference, gathered_outputs_fused
                    )
                )
            )
            print(
                "Are allclose? "
                + " ".join(
                    str(torch.allclose(ref, fus))
                    for ref, fus in zip(
                        gathered_outputs_nccl_reference, gathered_outputs_fused
                    )
                )
            )

    elif step is Step.ReduceScatter:
        run_nccl_reference_rs()
        run_fused_rs()
        if my_rank == 0:
            print("fused:")
            print(
                "Are equal? "
                + " ".join(
                    str(torch.equal(ref, fus))
                    for ref, fus in zip(
                        scattered_outputs_nccl_reference, scattered_outputs_fused
                    )
                )
            )
            print(
                "Are allclose? "
                + " ".join(
                    str(torch.allclose(ref, fus))
                    for ref, fus in zip(
                        scattered_outputs_nccl_reference, scattered_outputs_fused
                    )
                )
            )

    # The above checks might still return False for, e.g., bfloat16 because they
    # have too little tolerance for its lower precision. This method, OTOH, uses
    # variable tolerances based on dtype.
    # for ref, fus in zip(gathered_outputs_nccl_reference, gathered_outputs_fused):
    #     torch.testing.assert_close(ref, fus)
    # for ref, fus in zip(scattered_outputs_nccl_reference, scattered_outputs_fused):
    #     torch.testing.assert_close(ref, fus)

    all_benchs = {
        "compute_lower_bound": Bench(
            ag=run_compute_lower_bound_ag, rs=run_compute_lower_bound_rs
        ),
        "comms_lower_bound": Bench(
            ag=run_comms_lower_bound_ag, rs=run_comms_lower_bound_rs
        ),
        "nccl_reference": Bench(ag=run_nccl_reference_ag, rs=run_nccl_reference_rs),
        "fused": Bench(ag=run_fused_ag, rs=run_fused_rs),
        "fused_nowait": Bench(ag=run_fused_nowait_ag, rs=run_fused_nowait_rs),
        "fused_nowait_nomemcpy": Bench(
            ag=run_fused_nowait_nomemcpy_ag, rs=run_fused_nowait_nomemcpy_rs
        ),
    }

    unused_events = deque(
        tuple(torch.cuda.Event(enable_timing=my_rank == 0) for _ in range(2))
        for f in range(len(all_benchs))
    )
    used_events = deque()

    timings = {}

    gen = random.Random(42)

    if profile:
        profiler = torch.profiler.profile()
    else:
        profiler = contextlib.nullcontext()

    with profiler as p:
        for method in gen.sample(
            list(all_benchs),
            k=num_rounds * len(all_benchs),
            counts=[num_rounds] * len(all_benchs),
        ):
            fun = all_benchs[method][step]

            if unused_events:
                start_ev, end_ev = unused_events.popleft()
            else:
                old_method, start_ev, end_ev = used_events.popleft()
                end_ev.synchronize()
                if my_rank == 0:
                    timings.setdefault(old_method, []).append(
                        start_ev.elapsed_time(end_ev) / num_bench_iters
                    )

            for _ in range(num_warmup_iters):
                fun()
            start_ev.record()
            for _ in range(num_bench_iters):
                fun()
            end_ev.record()

            used_events.append((method, start_ev, end_ev))

        torch.cuda.synchronize()

    if profile:
        p.export_chrome_trace(f"fusion_trace_{my_rank}.json")

    if my_rank == 0:
        for method, start_ev, end_ev in used_events:
            timings.setdefault(method, []).append(
                start_ev.elapsed_time(end_ev) / num_bench_iters
            )

        for method in all_benchs:
            print(
                f"{method} = {mean(timings[method]):g}ms (+/- {stdev(timings[method]):g})"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", choices=SCENARIOS.keys())
    parser.add_argument("step", choices=list(Step), type=Step)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--dtype", choices=DTYPES.keys(), default="bfloat16")
    parser.add_argument("--num-rounds", type=int, default=20)
    parser.add_argument("--num-warmup-iters", type=int, default=5)
    parser.add_argument("--num-bench-iters", type=int, default=50)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    conns_from_prev = [None] * args.world_size
    conns_to_next = [None] * args.world_size
    for rank in range(args.world_size):
        end1, end2 = multiprocessing.get_context("spawn").Pipe(duplex=True)
        conns_to_next[rank] = end1
        conns_from_prev[(rank + 1) % args.world_size] = end2

    processes = []
    for rank in range(args.world_size):
        p = multiprocessing.get_context("spawn").Process(
            target=run_one_rank,
            args=(
                rank,
                args.world_size,
                args.scenario,
                args.step,
                args.dtype,
                args.num_rounds,
                args.num_warmup_iters,
                args.num_bench_iters,
                args.profile,
                conns_from_prev[rank],
                conns_to_next[rank],
            ),
            daemon=True,
        )
        p.start()
        processes.append(p)

    print("LAUNCHED")

    for rank, p in enumerate(processes):
        p.join()
        print(f"Rank {rank} exited with {p.exitcode}")

    print("JOINED")


if __name__ == "__main__":
    main()
