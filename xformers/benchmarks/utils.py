# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import contextlib
import copy
import glob
import logging
import math
import os
import pickle
import tempfile
from collections import defaultdict, namedtuple
from dataclasses import replace
from typing import Any, Dict, Generator, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import tqdm
from torch.utils import benchmark

sns.set()

TestCase = namedtuple("TestCase", ["function", "name"])


_triton_is_available = torch.cuda.is_available()
if _triton_is_available:
    try:
        import triton
    except ImportError as e:
        logging.warning(f"Triton is not available: {e}.\nbench_functions")
        _triton_is_available = False


def pretty_print(results, title, units):
    """Printout the contents of a dict as a human-readable and Markdown compatible array"""
    print(title)
    header = " Units: {:<45}".format(units)
    print("| " + header + "|" + "".join("{0:<20}|".format(k) for k in results.keys()))

    offset = len(header)
    print(
        "|-{}|".format("-" * offset)
        + "".join("{}|".format("-" * 20) for _ in results.keys())
    )

    workloads: Dict[str, Any] = {k: [] for v in results.values() for k in v.keys()}
    for v in results.values():
        for k in v.keys():
            workloads[k].append(v[k])

    for k, w in workloads.items():
        print(
            "| {0:<{offset}}|".format(k, offset=offset)
            + "".join("{:<20}|".format(v) for v in w)
        )

    print("")


def pretty_plot(
    results, title, units: str, filename=None, dash_key="", legend_loc="lower right"
):
    """Graph out the contents of a dict.
    Dash key means that if the result label has this key, then it will be displayed with a dash"""

    if not filename:
        filename = title + ".png"

    # Sanitize the filename
    filename = (
        filename.replace(" ", "_").replace("/", "_").replace("-", "_").replace(":", "")
    )

    # Gather all the results in "collumns"
    workloads: Dict[str, Any] = {k: [] for v in results.values() for k in v.keys()}
    for v in results.values():
        for k in v.keys():
            workloads[k].append(float(v[k]))

    # Make sure that the plot is big enough
    f = plt.figure()
    f.set_figwidth(6)
    f.set_figheight(6)

    # Display the collections
    for k, v in workloads.items():
        if dash_key and dash_key in k:
            plt.plot(list(results.keys()), v, "--")
        else:
            plt.plot(list(results.keys()), v)

    plt.title(title)
    plt.legend(list(workloads.keys()), loc=legend_loc)
    plt.ylabel(units)
    plt.xticks(rotation=45)

    plt.savefig(filename, bbox_inches="tight")
    plt.close(f)


if _triton_is_available:

    def bench_functions(
        test_cases: List[TestCase], shapes, metric_transform, unit, title=""
    ):
        device = torch.device("cuda")

        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            results: Dict[str, Any] = {}

            for B, M, K in shapes:
                a = torch.rand(B, M, K, device=device, dtype=dtype, requires_grad=True)

                for testcase in test_cases:
                    time = triton.testing.do_bench(lambda: testcase.function(a))[0]

                    metric = metric_transform(a, time)

                    key = f"B={B}, M={M}, K={K}"
                    if key not in results:
                        results[key] = {}

                    results[key][testcase.name] = f"{metric:.1f}"

            pretty_print(
                results,
                title=" ------------- Type: {} ------------- ".format(dtype),
                units=unit,
            )
            pretty_plot(results, title + str(dtype), unit, dash_key="pytorch")


def pretty_barplot(results, title, units: str, filename=None, dash_key=""):
    """Graph out the contents of a dict.
    Dash key means that if the result label has this key, then it will be displayed with a dash"""

    if not filename:
        filename = title + ".png"

    # Sanitize the filename
    filename = (
        filename.replace(" ", "_").replace("/", "_").replace("-", "_").replace(":", "")
    )

    xlabels = list(results.keys())
    # Gather all the results in "collumns"
    workloads: Dict[str, Any] = {k: [] for v in results.values() for k in v.keys()}
    for v in results.values():
        for k in v.keys():
            workloads[k].append(float(v[k]))

    options = list(workloads.keys())
    group_len = len(options)
    for key in workloads.keys():
        num_groups = len(workloads[key])
        break
    group_width = group_len + 1

    # Make sure that the plot is big enough
    f = plt.figure()
    f.set_figwidth(6)
    f.set_figheight(6)

    for idx in range(group_len):
        option = options[idx]
        values = workloads[option]
        xloc = np.arange(1 + idx, group_width * num_groups, group_width)
        plt.bar(xloc, values, width=1, edgecolor="black")

    plt.title(title)
    plt.legend(list(workloads.keys()), loc="upper right")
    plt.ylabel(units)

    ax = plt.gca()
    xticks_loc = np.arange(
        1 + (group_len - 1) / 2.0, group_width * num_groups, group_width
    )
    ax.set_xticks(xticks_loc, xlabels)
    plt.xticks(rotation=45)

    plt.setp(ax.xaxis.get_majorticklabels(), ha="right")
    ax.set_axisbelow(True)
    ax.yaxis.grid(color="gray", linestyle="dashed")
    ax.xaxis.grid(color="gray", linestyle="dashed")

    plt.savefig(filename, bbox_inches="tight")
    plt.close(f)


def rmf(filename: str) -> None:
    """Remove a file like rm -f."""
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass


@contextlib.contextmanager
def temp_files_ctx(num: int) -> Generator:
    """A context to get tempfiles and ensure they are cleaned up."""
    files = [tempfile.mkstemp()[1] for _ in range(num)]

    yield tuple(files)

    # temp files could have been removed, so we use rmf.
    for name in files:
        rmf(name)


META_ALGORITHM = "algorithm"
META_IS_REFERENCE = "is_ref"


def _finalize_results(results: List[Tuple[Dict[str, Any], Any]]) -> List[Any]:
    """
    Returns a `benchmark.Compare` object, except that if we have runs
    with different algorithms, we also add the algorithm name
    in the column titles
    """
    all_algorithms: Set[str] = set()
    all_description: Set[str] = set()
    for (metadata, r) in results:
        algo = metadata.get(META_ALGORITHM, None)
        if algo is not None:
            all_algorithms.add(algo)
        all_description.add(r.task_spec.description)
    display_algo = len(all_algorithms) > 1
    display_descr = len(all_description) > 1

    display_results = []
    for (metadata, r) in results:
        algo = metadata.get(META_ALGORITHM, None)
        if algo is None:
            display_results.append(r)
        else:
            r = copy.copy(r)
            description = ""
            if display_descr:
                description = r.task_spec.description
            if display_algo:
                if display_descr:
                    description += "["
                description += algo
                if display_descr:
                    description += "]"
            r.task_spec = replace(r.task_spec, description=description)
            display_results.append(r)
    return display_results


BASELINE_DESCRIPTIONS = ["eager", "vanilla"]


def _render_bar_plot(results: List[Any], store_results_folder: str) -> None:
    runtime: Dict[str, Dict[str, float]] = defaultdict(dict)
    memory_usage: Dict[str, Dict[str, float]] = defaultdict(dict)
    all_descriptions: List[str] = []
    for r in results:
        # Hacky: use a list to preserve order
        if r.task_spec.description not in all_descriptions:
            if r.task_spec.description in BASELINE_DESCRIPTIONS:
                all_descriptions.insert(0, r.task_spec.description)
            else:
                all_descriptions.append(r.task_spec.description)
        runtime[r.task_spec.sub_label][r.task_spec.description] = r.mean
        memory_usage[r.task_spec.sub_label][r.task_spec.description] = r.mem_use
    all_data_mem: List[Any] = []
    all_data_run: List[Any] = []
    for key, runtime_values in runtime.items():
        memory_values = memory_usage[key]
        all_data_mem.append(
            [key]
            + [
                memory_values.get(d, 0)
                / memory_values.get(all_descriptions[0], math.inf)
                for d in all_descriptions
            ]
        )
        all_data_run.append(
            [key]
            + [
                runtime_values.get(all_descriptions[0], 0)
                / runtime_values.get(d, math.inf)
                for d in all_descriptions
            ]
        )
    if all_descriptions[0] == "":
        all_descriptions[0] = "baseline"
    else:
        all_descriptions[0] = f"{all_descriptions[0]} (baseline)"

    for data, filename, title in [
        (all_data_mem, "mem.png", "Memory usage (vs baseline, lower is better)"),
        (
            all_data_run,
            "runtime.png",
            "Runtime speedup (vs baseline, higher is better)",
        ),
    ]:
        df = pd.DataFrame(data, columns=["Configuration"] + all_descriptions)
        df.plot(
            x="Configuration",
            kind="bar",
            stacked=False,
            title=title,
        )
        plt.tight_layout()
        filename_full = os.path.join(store_results_folder, filename)
        plt.savefig(filename_full)
        print(f"Saved plot: {filename_full}")


def benchmark_main_helper(
    benchmark_fn, cases: List[Dict[str, Any]], *, min_run_time: int = 2
) -> None:
    """
    Helper function to run benchmarks.
    Supports loading previous results for comparison, and saving current results to file.
    """
    SKIP_VANILLA_TASKS_IF_ALREADY_DONE = True

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fn", default=None, type=str, help="Only benchmark this function"
    )
    parser.add_argument(
        "--label", default=None, type=str, help="Store results to a file"
    )
    parser.add_argument(
        "--compare",
        default=None,
        type=str,
        help="Compare to previously stored benchmarks (coma separated)",
    )
    args = parser.parse_args()

    if args.fn is not None and args.fn != benchmark_fn.__name__:
        print(f'Skipping benchmark "{benchmark_fn.__name__}"')
        return

    results_compare_to = []
    results = []

    store_results_folder = os.path.expanduser(
        os.path.join("~", ".cache", "xformers", "benchmarks", benchmark_fn.__name__)
    )
    optimized_label = "optimized" if args.label is None else args.label

    try:
        env = (
            torch.cuda.get_device_name(torch.cuda.current_device())
            .replace(" ", "_")
            .replace("-", "_")
        )
    except RuntimeError:  # No GPU
        env = "cpu"

    os.makedirs(store_results_folder, exist_ok=True)

    # Load runs that we want to compare to
    skip_vanilla_tasks = set()
    if args.compare is not None:
        for name in args.compare.split(","):
            for filename in glob.glob(
                os.path.join(store_results_folder, f"{name}.*.pkl")
            ):
                with open(filename, "rb") as fd:
                    for row in pickle.load(fd):
                        if isinstance(row, tuple):
                            metadata, r = row
                        else:
                            # Backward compatibility
                            metadata, r = {}, row
                        spec = r.task_spec
                        if r.task_spec.description not in BASELINE_DESCRIPTIONS:
                            # (in case the file was renamed)
                            r.task_spec = replace(r.task_spec, description=name)
                        elif spec.env == env:
                            if SKIP_VANILLA_TASKS_IF_ALREADY_DONE:
                                skip_vanilla_tasks.add(
                                    (spec.sub_label, spec.num_threads)
                                )
                            else:
                                continue
                        results_compare_to.append((metadata, r))

    pbar = tqdm.tqdm(cases, leave=False)
    for case in pbar:
        # pbar.set_description(str(case))
        pbar.write(f"====== {str(case)} ======")
        try:
            benchmarks_generator = benchmark_fn(**case)
        except NotImplementedError:
            # pbar.write(f"Skipped (NotImplementedError)")
            continue
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e):
                raise
            pbar.write("Skipped (OOM)")
            continue

        name = None
        try:
            for benchmark_object in benchmarks_generator:
                is_optimized = (
                    benchmark_object._task_spec.description not in BASELINE_DESCRIPTIONS
                )
                if benchmark_object is None:
                    continue
                metadata = {}
                if is_optimized:
                    metadata[META_ALGORITHM] = benchmark_object._task_spec.description
                    benchmark_object._task_spec = replace(
                        benchmark_object._task_spec, description=optimized_label
                    )
                elif (
                    benchmark_object._task_spec.sub_label,
                    benchmark_object._task_spec.num_threads,
                ) in skip_vanilla_tasks:
                    continue

                memory = math.inf
                try:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    benchmark_object._task_spec = replace(
                        benchmark_object._task_spec, env=env
                    )
                    measurement = benchmark_object.blocked_autorange(
                        min_run_time=min_run_time
                    )
                    torch.cuda.synchronize()
                    results.append((metadata, measurement))
                    name = measurement.task_spec.description
                    memory = torch.cuda.max_memory_allocated() / 2**20
                    measurement.mem_use = memory
                except RuntimeError as e:
                    if "CUDA out of memory" not in str(e):
                        raise
                    pbar.write("Skipped (OOM)")
                finally:
                    del benchmark_object
                pbar.write(f"{name}: memory used: {memory} MB")
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e):
                raise
            pbar.write("Skipped (OOM)")
        # Display results for benchmarks we just calculated
        if name is not None:

            def matches_current(r):
                return (
                    r[1].task_spec.sub_label == results[-1][1].task_spec.sub_label
                    and r[1].task_spec.label == results[-1][1].task_spec.label
                )

            pbar.write(
                str(
                    benchmark.Compare(
                        _finalize_results(
                            list(filter(matches_current, results))
                            + list(filter(matches_current, results_compare_to))
                        )
                    )
                )
            )

    results_for_print = _finalize_results(results + results_compare_to)
    benchmark.Compare(results_for_print).print()
    _render_bar_plot(results_for_print, store_results_folder)

    # Save runs to a file
    if args.label is not None:
        write_to_path = os.path.join(
            store_results_folder, f"{optimized_label}.{env}.pkl"
        )
        with open(write_to_path, "wb+") as fd:
            pickle.dump(results, fd)
        print(f"Saved results to {write_to_path}")
