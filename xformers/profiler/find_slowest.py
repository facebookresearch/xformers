# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import concurrent.futures
import glob
import os
import subprocess
from shlex import quote
from typing import Dict, List

import pandas as pd


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
    grouped_sorted_events = list(
        events.filter(items=["name", "log_name", "dur"])
        .groupby(["name", "log_name"])
        .sum()
        .groupby(["name"])
        .std()
        .sort_values(["dur"], ascending=False)
        .iterrows()
    )

    return [
        {"name": idx, "std_dev": f"{row.dur / 1000:.2f} ms"}
        for idx, row in grouped_sorted_events[:top]
    ]


def sort_nccl_events(
    nccl_events, top_k: int = 3, last_k: int = 3
) -> List[Dict[str, str]]:
    grouped_sorted_events = list(
        nccl_events.filter(items=["log_name", "dur"])
        .groupby(["log_name"])
        .sum()
        .sort_values(["dur"])
        .iterrows()
    )

    return [
        {"log_name": idx, "nccl_ms": f"{row.dur / 1000:.2f} ms"}
        for idx, row in (
            grouped_sorted_events[:top_k] + grouped_sorted_events[-last_k:]
        )
    ]


def read_one_file(profile_trace_path: str) -> pd.DataFrame:
    if profile_trace_path.endswith(".csv"):
        return pd.read_csv(profile_trace_path, names=["name", "dur"])

    jq_pipe = '.traceEvents[] | select(.cat == "kernel") | [.name, .dur] | @csv'
    if profile_trace_path.endswith(".gz"):
        cmd = (
            f"gunzip -c {quote(profile_trace_path)} | jq --raw-output {quote(jq_pipe)}"
        )
    else:
        cmd = f"jq --raw-output {quote(jq_pipe)} {quote(profile_trace_path)}"

    subp = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    try:
        kernel_events = pd.read_csv(subp.stdout, names=["name", "dur"])
    except Exception:
        subp.terminate()
        raise
    finally:
        assert subp.wait() == 0

    return kernel_events


def parse_one_file(profile_trace_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    kernel_events = read_one_file(profile_trace_path)
    kernel_events["log_name"] = os.path.basename(profile_trace_path)

    communication_kernels = kernel_events[kernel_events.name.str.startswith("nccl")]
    computation_kernels = kernel_events[~(kernel_events.name.str.startswith("nccl"))]

    return communication_kernels, computation_kernels


def print_profiling_info(cuda_profile_dir: str):
    cuda_profile_path_name = f"{cuda_profile_dir}/kernels_*.csv"
    profile_files = glob.glob(cuda_profile_path_name)

    if len(profile_files) == 0:
        cuda_profile_path_name = f"{cuda_profile_dir}/*.pt.trace.json.gz"
        profile_files = glob.glob(cuda_profile_path_name)

    if len(profile_files) == 0:
        cuda_profile_path_name = f"{cuda_profile_dir}/*.json"
        profile_files = glob.glob(cuda_profile_path_name)

    if len(profile_files) == 0:
        raise Exception(
            f"Couldnt find any profiling trace in the specified directory: {cuda_profile_dir}"
        )

    # Extract detailed NCCL event durations for all logs
    communication_kernels = []
    computation_kernels = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        for index, (comm_ks, comp_ks) in enumerate(
            executor.map(parse_one_file, profile_files)
        ):
            print(
                f"Processed file {index + 1}/{len(profile_files)}", end="\r", flush=True
            )
            communication_kernels.append(comm_ks)
            computation_kernels.append(comp_ks)
    communication_kernels = pd.concat(communication_kernels)
    computation_kernels = pd.concat(computation_kernels)
    print()

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
