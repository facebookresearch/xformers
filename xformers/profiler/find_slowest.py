# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import glob
import gzip
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np


def read_gzipped_json(file_path):
    with gzip.open(file_path, "rt") as f:
        return json.load(f)


# Extract detailed event durations
def extract_detailed_info(log_data):
    events = []
    rank = log_data["distributedInfo"]["rank"]

    for event in log_data["traceEvents"]:
        if "name" in event and "dur" in event:
            events.append(
                {
                    "name": event["name"],
                    "start_time": event["ts"],
                    "duration_ms": (event["dur"] / 1000),  # convert to milliseconds
                    "rank": f"GPU {rank}",
                    "trace_name": log_data["traceName"],
                    "cat": event["cat"],
                    "log_name": log_data["log_name"],
                }
            )
    return events


def print_json_as_dataframe(json_list):
    if not json_list:
        print("Empty list")
        return

    # Extract the headers from the keys of the first dictionary
    headers = list(json_list[0].keys())

    # Determine the width of each column
    col_widths = {header: max(len(header), 10) for header in headers}
    for row in json_list:
        for header in headers:
            col_widths[header] = max(col_widths[header], len(str(row[header])))

    # Create the header row
    header_row = "  ".join(f"{header:<{col_widths[header]}}" for header in headers)
    print(header_row)
    print("-" * len(header_row))

    # Create each data row
    for row in json_list:
        data_row = "  ".join(
            f"{str(row[header]):<{col_widths[header]}}" for header in headers
        )
        print(data_row)


def compute_std_dev_of_event_durations_over_ranks(events, top=5):
    # Step 1: Group by 'rank' and 'kernel' and sum the 'duration_ms'
    grouped_data: defaultdict[str, defaultdict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for event in events:
        grouped_data[event["name"]][event["rank"]] += event["duration_ms"]

    # Step 2: Calculate the standard deviation across ranks for each kernel
    std_devs = []
    for name, ranks in grouped_data.items():
        durations = np.array(list(ranks.values()))
        std_dev = np.std(durations, ddof=1)
        std_devs.append({"name": name, "std_dev": std_dev})

    # Step 3: Sort by standard deviation in descending order
    std_devs.sort(key=lambda x: x["std_dev"], reverse=True)
    for r in std_devs:
        r["std_dev"] = f"{r['std_dev']:.2f} ms"

    return std_devs[:top]


def sort_nccl_events(
    nccl_events, top_k: int = 3, last_k: int = 3
) -> List[Dict[str, str]]:
    # Step 1: Group by 'log_name' and sum the 'duration_ms'
    grouped_data: Dict[str, float] = defaultdict(float)
    for event in nccl_events:
        key = event["log_name"]
        grouped_data[key] += event["duration_ms"]

    # Step 2: Create a sorted list of tuples by 'duration_ms' in descending order
    sorted_list = sorted(grouped_data.items(), key=lambda x: x[1], reverse=True)

    # Step 3: Format the sorted list
    formatted_list: List[Dict[str, str]] = [
        {"log_name": log_name, "nccl_ms": f"{duration:.2f} ms"}
        for log_name, duration in sorted_list
    ]

    # Step 4: Get top_k and last_k items
    top_k_list = formatted_list[:top_k]
    last_k_list = formatted_list[-last_k:]

    return top_k_list + last_k_list


def print_profiling_info(cuda_profile_dir: str):
    has_json_gz_files = None

    cuda_profile_path_name = f"{cuda_profile_dir}/*trace.json.gz"

    profile_files = glob.glob(cuda_profile_path_name)

    if len(profile_files) == 0:
        cuda_profile_path_name = f"{cuda_profile_dir}/*.json"

        profile_files = glob.glob(cuda_profile_path_name)
        has_json_gz_files = False
    else:
        has_json_gz_files = True

    if len(profile_files) == 0:
        raise Exception(
            f"Couldnt find any profiling trace in the specified directory: {cuda_profile_dir}"
        )

    # Extract detailed NCCL event durations for all logs
    events_details = []
    total_files = len(profile_files)
    for index, profile_trace_path in enumerate(profile_files):
        print(f"Processing file {index + 1}/{total_files}", end="\r")
        sys.stdout.flush()

        if has_json_gz_files:
            log_data = read_gzipped_json(profile_trace_path)
        else:
            with open(profile_trace_path, "r") as f:
                log_data = json.loads(f.read())

        log_data["log_name"] = os.path.basename(profile_trace_path)
        events_details.extend(extract_detailed_info(log_data))
    print()

    kernel_events = [e for e in events_details if e["cat"] == "kernel"]
    communication_kernels = [e for e in kernel_events if "nccl" in e["name"]]
    computation_kernels = [e for e in kernel_events if "nccl" not in e["name"]]

    print("The longest and shortest communication_kernels:")
    print_json_as_dataframe(sort_nccl_events(communication_kernels))
    print("\n\n")

    std_df = compute_std_dev_of_event_durations_over_ranks(communication_kernels)
    print("The standard deviation of nccl kernels durations across ranks:")
    print_json_as_dataframe(std_df)
    print("\n\n")

    std_df = compute_std_dev_of_event_durations_over_ranks(computation_kernels)
    print("The standard deviation of computation kernels durations across ranks:")
    print_json_as_dataframe(std_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CUDA profile directory.")
    parser.add_argument("cuda_profile_dir", type=str, help="The CUDA profile directory")

    args = parser.parse_args()

    print_profiling_info(args.cuda_profile_dir)
