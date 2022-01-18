# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import json
from pathlib import Path
from typing import Any, Dict

reference_steps = {
    "image": 35176,
    "listops": 10000,
    "pathfinder32-curv_contour_length_14": 62400,
    "pathfinder32-curv_baseline": 62400,
    "pathfinder32-curv_contour_length_9": 62400,
    "text": 20000,
    "retrieval": 30000,
}

if __name__ == "__main__":
    # Get the user requests
    parser = argparse.ArgumentParser(
        "Collect results from a given batch of distributed results"
    )
    parser.add_argument("-ck", "--checkpoint_path", required=True)
    args = parser.parse_args()

    # Go through all the data in the given repo, try to find the end results
    root = Path(args.checkpoint_path)

    # - list all the mechanisms being benchmarked
    results: Dict[str, Any] = {}

    for attention in filter(lambda x: x.is_dir(), root.iterdir()):
        print(f"\nFound results for {attention.stem}")
        task_logs = attention.glob("*.log")
        results[attention.stem] = {}

        for task in filter(lambda x: "__0" in str(x), task_logs):
            task_name = task.stem.split("__")[0]
            print(f"Logs found for task: {task_name}")
            results[attention.stem][task_name] = -1

            # - collect the individual results
            with open(task, "r") as result_file:
                for line in reversed(result_file.readlines()):
                    if '"component": "test"' in line:
                        # Check that all the steps are done
                        res = json.loads(line)
                        if res["train_step_idx"] == reference_steps[task_name]:
                            results[attention.stem][task_name] = res["best_accu"]
                            print(
                                f"Final result found for {task_name}: {results[attention.stem][task_name]}"
                            )
                        else:
                            print(
                                "Current step: {}/{}. Not finished".format(
                                    res["train_step_idx"], reference_steps[task_name]
                                )
                            )
                        break

    print(f"\nCollected results: {json.dumps(results, indent=2)}")

    #  - reduction: compute the average
    tasks = set(t for v in results.values() for t in v.keys())
    # -- fill in the possible gaps
    for att in results.keys():
        for t in tasks:
            if t not in results[att].keys():
                results[att][t] = 0.0

    # -- add the average value
    for att in results.keys():
        results[att]["AVG"] = round(sum(results[att][t] for t in tasks) / len(tasks), 2)

    # - Format as an array, markdown style
    tasks_sort = sorted(
        set(t for v in results.values() for t in v.keys()), reverse=True
    )
    print(
        "{0:<20}".format("") + "".join("{0:<20}   ".format(t[:10]) for t in tasks_sort)
    )

    for att in results.keys():
        print(
            "{0:<20}".format(att)
            + "".join("{0:<20}   ".format(results[att][t]) for t in tasks_sort)
        )
