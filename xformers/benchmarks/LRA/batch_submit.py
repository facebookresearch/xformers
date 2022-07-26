# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import os
from pathlib import Path

from xformers.benchmarks.LRA.run_tasks import Task
from xformers.components.attention import ATTENTION_REGISTRY


def get_default_shared_folder() -> str:
    checkpoint_paths = ["/checkpoint", "/checkpoints"]
    for checkpoint_path in checkpoint_paths:
        if Path(checkpoint_path).is_dir():
            return checkpoint_path

    return "."


if __name__ == "__main__":
    default_checkpoint_path = get_default_shared_folder()

    # Get the user requests
    parser = argparse.ArgumentParser(
        "Benchmark different attention mechanisms on various sequence lengths"
    )
    parser.add_argument("-c", "--config_path", required=True)
    parser.add_argument("-ck", "--checkpoint_path", required=True)
    parser.add_argument(
        "-a", "--attentions", nargs="+", default=list(ATTENTION_REGISTRY.keys())
    )
    parser.add_argument("-t", "--tasks", nargs="+", default=[t.value for t in Task])
    parser.add_argument(
        "--partition", default="a100", type=str, help="Partition where to submit"
    )
    args = parser.parse_args()

    for attention in args.attentions:
        for task in args.tasks:
            os.system(
                "python3 run_with_submitit.py"
                + f" --attention {attention}  --task {task} --config {args.config_path}"
                + f" --checkpoint_dir {args.checkpoint_path}/{attention}/{task}"
                + f" --partition {args.partition}"
            )
