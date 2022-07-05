# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

if __name__ == "__main__":
    # Get the user requests
    parser = argparse.ArgumentParser(
        "Collect results from a given batch of distributed results"
    )
    parser.add_argument("-ck", "--checkpoint_path", required=True)
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    # Go through all the data in the given repo, try to find the end results
    root = Path(args.checkpoint_path)

    # - list all the mechanisms being benchmarked
    results: Dict[str, Any] = {}

    for attention in filter(lambda x: x.is_dir(), root.iterdir()):
        logging.info(f"\nFound results for {attention.stem}")
        task_jsons = attention.glob("*/test_eval_summary.json")
        results[attention.stem] = {}

        for task in task_jsons:
            task_name = task.stem.split("__")[0]
            logging.info(f"Logs found for task: {task_name}")
            results[attention.stem][task_name] = -1
            found_result = False

            # - collect the individual results
            with open(task, "r") as result_file:
                dct = json.load(result_file)
                if "test_accu_mean" in dct:
                    found_result = True
                    results[attention.stem][task_name] = dct["test_accu_mean"]

                    logging.info(
                        f"Final result found for {task_name} at epoch {dct['train_step_idx']}: "
                        f"{results[attention.stem][task_name]}"
                    )
                else:
                    break

            # - report an error if no result was found
            if not found_result:
                ERR_TAIL = 30

                logging.warning(
                    f"No result found for {task_name}, showing the error log in {task.parent}"
                )
                err_log = Path(task.parent).glob("*.err")
                print("*****************************************************")
                with open(next(err_log), "r") as err_file:
                    for i, line in enumerate(reversed(err_file.readlines())):
                        print(line, end="")
                        if i > ERR_TAIL:
                            break
                print("*****************************************************")

    logging.info(f"\nCollected results: {json.dumps(results, indent=2)}")

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
