# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


"""
A script to run multinode training with submitit.
Almost copy-paste from https://github.com/facebookresearch/deit/blob/main/run_with_submitit.py
"""

import argparse
import os
import uuid
from pathlib import Path

import submitit

from xformers.benchmarks.LRA.run_tasks import benchmark, get_arg_parser


def parse_args():
    parser = argparse.ArgumentParser(
        "Submitit for LRA", parents=[get_arg_parser()], add_help=False
    )
    parser.add_argument(
        "--ngpus", default=1, type=int, help="Number of gpus to request on each node"
    )
    parser.add_argument(
        "--nodes", default=1, type=int, help="Number of nodes to request"
    )
    parser.add_argument("--timeout", default=2800, type=int, help="Duration of the job")

    parser.add_argument(
        "--partition", default="a100", type=str, help="Partition where to submit"
    )
    parser.add_argument(
        "--use_volta32", action="store_true", help="Big models? Use this"
    )
    parser.add_argument(
        "--enforce_host_memory", action="store_true", help="Use if the host OOMs"
    )

    parser.add_argument(
        "--comment",
        default="",
        type=str,
        help="Comment to pass to scheduler, e.g. priority message",
    )
    return parser.parse_args()


def get_shared_folder() -> Path:
    user = os.getenv("USER")
    checkpoint_paths = ["/checkpoint", "/checkpoints"]
    for checkpoint_path in checkpoint_paths:
        if Path(checkpoint_path).is_dir():
            p = Path(f"{checkpoint_path}/{user}/xformers/submitit")
            p.mkdir(exist_ok=True, parents=True)
            return p
    raise RuntimeError(f"No shared folder available - considering {checkpoint_paths}")


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        self._setup_gpu_args()
        benchmark(self.args)

    def checkpoint(self):
        self.args.dist_url = get_init_file().as_uri()
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        job_env = submitit.JobEnvironment()
        self.args.checkpoint_dir = Path(
            str(self.args.checkpoint_dir).replace("%j", str(job_env.job_id))
        )
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    args = parse_args()
    if args.checkpoint_dir == "":
        args.checkpoint_dir = get_shared_folder() / "%j"
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(
        folder=args.checkpoint_dir, slurm_max_num_timeout=30
    )

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout
    args.world_size = args.nodes * args.ngpus

    partition = args.partition

    kwargs = {
        "gpus_per_node": num_gpus_per_node,
        "tasks_per_node": num_gpus_per_node,  # one task per GPU
        "cpus_per_task": 10,
        "nodes": nodes,
        "timeout_min": timeout_min,  # max is 60 * 72
        # Below are cluster dependent parameters
        "slurm_partition": partition,
        "slurm_signal_delay_s": 120,
    }

    if args.enforce_host_memory:
        kwargs["mem_gb"] = (40 * num_gpus_per_node,)

    if args.use_volta32:
        kwargs["slurm_constraint"] = "volta32gb"

    if args.comment:
        kwargs["slurm_comment"] = args.comment

    executor.update_parameters(
        **kwargs,
    )

    executor.update_parameters(name="lra")

    args.dist_url = get_init_file().as_uri()
    args.temp_file = str(get_init_file())

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print(f"Submitted job_id: {job.job_id}")
    print(f"Logs and checkpoints will be saved at: {args.checkpoint_dir}")
    with open(Path(f"{args.checkpoint_dir}") / Path("jobs.txt"), "a") as jobfile:
        jobfile.write(f"{job.job_id}\n")


if __name__ == "__main__":
    main()
