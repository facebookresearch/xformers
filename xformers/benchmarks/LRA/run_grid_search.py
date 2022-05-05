# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import itertools
import os
import uuid
from datetime import date
from pathlib import Path
from typing import Dict, Iterable

import submitit

from xformers.benchmarks.LRA.run_with_submitit import (
    Trainer,
    get_init_file,
    get_shared_folder,
    parse_args,
)


def grid_parameters(grid: Dict):
    """
    Yield all combinations of parameters in the grid (as a dict)
    """
    grid_copy = dict(grid)
    # Turn single value in an Iterable
    for k in grid_copy:
        if not isinstance(grid_copy[k], Iterable):
            grid_copy[k] = [grid_copy[k]]
    for p in itertools.product(*grid_copy.values()):
        yield dict(zip(grid.keys(), p))


def grid_search(args):
    if args.checkpoint_dir == "":
        args.checkpoint_dir = get_shared_folder() / "%j"

    date_curr = date.today().strftime("%m-%d-%Y")
    orig_check_dir = os.path.join(args.checkpoint_dir, date_curr)

    # Create the executor
    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(
        folder=get_shared_folder() / "%j", slurm_max_num_timeout=30
    )
    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    args.world_size = args.nodes * args.ngpus
    partition = args.partition

    executor.update_parameters(
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=10,
        nodes=nodes,
        timeout_min=60 * 72,
        slurm_signal_delay_s=120,
        slurm_partition=partition,
    )
    executor.update_parameters(name="lra")

    if args.task == "text":
        grid_meta = {
            "training:learning_rate": (
                [1e-4, 2e-4, 3e-4, 5e-5],
                lambda val: f"lr{val}",
            ),
            "training:warmup": ([3000, 8000], lambda val: f"warmup{val}"),
            "training:seed": ([1234, 32, 1994], lambda val: f"seed{val}"),
            "training:weight_decay": ([0.02, 0.05, 0.01], lambda val: f"wd{val}"),
            "model:pooling_model": (["cls"], lambda val: f"pool-{val}"),
            "model:common:dropout": ([0, 0.05], lambda val: f"drop{val}"),
        }
    elif args.task == "retrieval":
        grid_meta = {
            "training:learning_rate": ([1e-4, 3e-4], lambda val: f"lr{val}"),
            "training:warmup": ([2000, 8000], lambda val: f"warmup{val}"),
            "training:seed": ([4096, 1234, 3, 15, 5], lambda val: f"seed{val}"),
            "training:weight_decay": ([0.01, 0], lambda val: f"wd{val}"),
            "model:pooling_model": (["cls"], lambda val: f"pool-{val}"),
            "model:common:dropout": ([0], lambda val: f"drop{val}"),
        }
    elif args.task == "listops":
        grid_meta = {
            "training:learning_rate": (
                [1e-4, 2e-4, 3e-4, 5e-5],
                lambda val: f"lr{val}",
            ),
            "training:warmup": ([3000, 2000], lambda val: f"warmup{val}"),
            "training:seed": (
                [
                    1234,
                ],
                lambda val: f"seed{val}",
            ),
            "training:weight_decay": ([0.02, 0.05, 0, 1], lambda val: f"wd{val}"),
            "model:pooling_model": (["cls"], lambda val: f"pool-{val}"),
            "model:common:dropout": ([0], lambda val: f"drop{val}"),
        }
    else:
        grid_meta = {
            "training:learning_rate": ([1e-4, 5e-5], lambda val: f"lr{val}"),
            "training:warmup": ([8000], lambda val: f"warmup{val}"),
            "training:seed": ([1234, 4321, 3], lambda val: f"seed{val}"),
            "training:weight_decay": ([0.01], lambda val: f"wd{val}"),
            "model:pooling_model": (["cls"], lambda val: f"pool-{val}"),
            "model:common:dropout": ([0.1], lambda val: f"drop{val}"),
        }

    grid = {k: v[0] for k, v in grid_meta.items()}
    save_key = {k: v[1] for k, v in grid_meta.items()}

    hyper_parameters = list(grid_parameters(grid))
    jobs = []

    for i, grid_data in enumerate(hyper_parameters):

        args.sweep_parameters = grid_data
        run_name = f"{args.attention}"
        # run_name = "paper_config"
        for k, v in grid_data.items():
            run_name += "prenorm-" + save_key[k](v)
        args.checkpoint_dir = os.path.join(
            orig_check_dir, f"{args.task}", "logs", run_name
        )
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        args.tb_dir = os.path.join(orig_check_dir, f"{args.task}", "tb", run_name)
        Path(args.tb_dir).mkdir(parents=True, exist_ok=True)

        # Chronos needs a different job name each time
        executor.update_parameters(name=f"lra_{args.task}_{i:02d}_{uuid.uuid4().hex}")

        args.dist_url = get_init_file().as_uri()
        args.temp_file = str(get_init_file())

        trainer = Trainer(args)
        job = executor.submit(trainer)
        jobs.append(job)
        print(f"Run {i:02d} submitted with train cfg: {args}")
    print(f"Submitted jobs ids: {','.join([str(job.job_id) for job in jobs])}")


if __name__ == "__main__":
    args = parse_args()
    grid_search(args)
