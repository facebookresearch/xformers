# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import contextlib
import copy
import csv
import functools
import glob
import itertools
import logging
import math
import os
import tempfile
from collections import defaultdict, namedtuple
from dataclasses import replace
from typing import Any, Dict, Generator, Iterator, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import tqdm
from torch.utils import benchmark

sns.set()

TestCase = namedtuple("TestCase", ["function", "name"])


class NotSupportedInputError(Exception):
    pass


_triton_is_available = torch.cuda.is_available()
if _triton_is_available:
    try:
        import triton
    except ImportError as e:
        logging.warning(f"Triton is not available: {e}.\nbench_functions")
        _triton_is_available = False


def get_func_name(fn):
    if isinstance(fn, functools.partial):
        return fn.func.__name__
    return fn.__name__


def pretty_print(results, title, units) -> None:
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
    Dash key means that if the result label has this key, then it will be displayed with a dash
    """

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
    Dash key means that if the result label has this key, then it will be displayed with a dash
    """

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
BASELINE_DESCRIPTIONS = ["eager", "vanilla", "pytorch"]


# Serialize/unserialize to CSV
# We could use pkl, but resort to CSV for readability
def _benchmark_results_from_csv(filename: str) -> List[Tuple[Dict[str, Any], Any]]:
    parts = os.path.basename(filename).split(".")
    env = ""
    description = ""
    if len(parts) == 3:
        env = parts[1]
        description = parts[0]

    data = []
    with open(filename, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if description != "" and row["description"] not in BASELINE_DESCRIPTIONS:
                row["description"] = description
            task_spec = benchmark.utils.common.TaskSpec(
                stmt="",
                setup="",
                global_setup="",
                label=row["label"],
                sub_label=row["sub_label"],
                description=row["description"],
                env=env,
                num_threads=int(row["num_threads"]),
            )
            measurement = benchmark.utils.common.Measurement(
                number_per_run=1,
                raw_times=[float(row["runtime_us"]) / (1000.0 * 1000)],
                task_spec=task_spec,
            )
            measurement.mem_use = float(row["mem_use_mb"])  # type: ignore
            data.append(
                (
                    {
                        META_ALGORITHM: (
                            row["algorithm"] if row["algorithm"] != "" else None
                        ),
                    },
                    measurement,
                )
            )
    return data


def _benchmark_results_to_csv(
    filename: str, results: List[Tuple[Dict[str, Any], Any]]
) -> None:
    data = [
        {
            "sub_label": r.task_spec.sub_label,
            "label": r.task_spec.label,
            "num_threads": r.task_spec.num_threads,
            "algorithm": metadata.get(META_ALGORITHM, ""),
            "description": (
                r.task_spec.description
                if r.task_spec.description in BASELINE_DESCRIPTIONS
                else ""
            ),
            "runtime_us": int(1000 * 1000 * r.mean),
            "mem_use_mb": r.mem_use,
        }
        for metadata, r in results
    ]
    with open(filename, "w+", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(data[0].keys()))
        writer.writeheader()
        for d in data:
            writer.writerow(d)


def _finalize_results(results: List[Tuple[Dict[str, Any], Any]]) -> List[Any]:
    """
    Returns a `benchmark.Compare` object, except that if we have runs
    with different algorithms, we also add the algorithm name
    in the column titles
    """
    all_algorithms: Set[str] = set()
    all_description: Set[str] = set()
    for metadata, r in results:
        algo = metadata.get(META_ALGORITHM, None)
        if algo is not None:
            all_algorithms.add(algo)
        all_description.add(r.task_spec.description)
    display_algo = len(all_algorithms) > 1
    display_descr = len(all_description) > 1

    display_results = []
    for metadata, r in results:
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


def _render_bar_plot(results: List[Any], store_results_folder: str) -> None:
    if not results:
        return
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
        denom = memory_values.get(all_descriptions[0], math.inf)
        if denom == 0:
            all_data_mem.append([key] + [0] * len(all_descriptions))
        else:
            all_data_mem.append(
                [key] + [memory_values.get(d, 0) / denom for d in all_descriptions]
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


def create_argparser() -> argparse.ArgumentParser:
    """
    Create CLI argument parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fn", default=None, type=str, help="Only benchmark this function"
    )
    parser.add_argument(
        "--label", default=None, type=str, help="Store results to a file"
    )
    parser.add_argument(
        "--fail_if_regression",
        action="store_true",
        help="Enabled in CI to check against performance regressions",
    )
    parser.add_argument(
        "--compare",
        default=None,
        type=str,
        help="Compare to previously stored benchmarks (coma separated)",
    )
    parser.add_argument(
        "--omit-baselines",
        action="store_true",
        help="Do not run the (potentially slow) baselines",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Skip intermediate results and progress bar",
    )
    return parser


def benchmark_main_helper(
    benchmark_fn, cases: List[Dict[str, Any]], arg_parser=None, **kwargs
) -> None:
    """
    Helper function to run benchmarks.
    Supports loading previous results for comparison, and saving current results to file.
    """
    arg_parser = arg_parser or create_argparser()
    args = arg_parser.parse_args()

    if args.fn is not None and args.fn != get_func_name(benchmark_fn):
        print(f'Skipping benchmark "{get_func_name(benchmark_fn)}"')
        return
    benchmark_run_and_compare(
        benchmark_fn=benchmark_fn,
        cases=cases,
        optimized_label="optimized" if args.label is None else args.label,
        fail_if_regression=args.fail_if_regression,
        compare=args.compare.split(",") if args.compare is not None else [],
        quiet=args.quiet,
        omit_baselines=args.omit_baselines,
        **kwargs,
    )


def benchmark_run_and_compare(
    benchmark_fn,
    cases: List[Dict[str, Any]],
    compare: List[str],
    omit_baselines: bool = False,
    fail_if_regression: bool = False,
    quiet: bool = False,
    optimized_label: str = "optimized",
    *,
    min_run_time: float = 2.0,
    atol_s: float = 30e-6,
    rtol: float = 0.05,
) -> None:
    SKIP_VANILLA_TASKS_IF_ALREADY_DONE = True
    results_compare_to = []
    results = []

    store_results_folder = os.path.expanduser(
        os.path.join(
            os.environ.get(
                "XFORMERS_BENCHMARKS_CACHE",
                os.path.join("~", ".cache", "xformers", "benchmarks"),
            ),
            get_func_name(benchmark_fn),
        )
    )

    try:
        env = (
            torch.cuda.get_device_name(torch.cuda.current_device())
            .replace(" ", "_")
            .replace("-", "_")
            .replace(".", "_")
            .replace("/", "_")
        )
    except (RuntimeError, AssertionError):  # No GPU
        env = "cpu"
    assert (
        "." not in optimized_label
    ), f"label=`{optimized_label}` should not contain dots"
    assert "." not in env, f"env=`{env}` should not contain dots"

    os.makedirs(store_results_folder, exist_ok=True)

    # Load runs that we want to compare to
    skip_vanilla_tasks = set()
    for cmp_name in compare:
        name_with_env = cmp_name if "." in cmp_name else f"{cmp_name}.*"
        for filename in glob.glob(
            os.path.join(store_results_folder, f"{name_with_env}.csv")
        ):
            loaded = _benchmark_results_from_csv(filename)
            for m, r in loaded:
                if m.get(META_ALGORITHM) is not None:
                    m[META_ALGORITHM] = m[META_ALGORITHM].partition("@")[0]
                if r.task_spec.env == env and SKIP_VANILLA_TASKS_IF_ALREADY_DONE:
                    skip_vanilla_tasks.add(
                        (r.task_spec.sub_label, r.task_spec.num_threads)
                    )
            results_compare_to += loaded

    if not quiet:
        pbar = tqdm.tqdm(cases, leave=False)
        cases = pbar
    for case in cases:
        if quiet:
            print(str(case))
        else:
            pbar.write(f"====== {str(case)} ======")
        try:
            benchmarks_generator = benchmark_fn(**case)
        except NotImplementedError:
            # pbar.write(f"Skipped (NotImplementedError)")
            continue
        except RuntimeError as e:
            if not _is_oom_error(e):
                raise
            if not quiet:
                pbar.write("Skipped (OOM)")
            continue

        name = None
        try:
            for benchmark_object in benchmarks_generator:
                is_optimized = (
                    benchmark_object._task_spec.description not in BASELINE_DESCRIPTIONS
                )
                metadata = {}
                if is_optimized:
                    metadata[META_ALGORITHM] = benchmark_object._task_spec.description
                    benchmark_object._task_spec = replace(
                        benchmark_object._task_spec, description=optimized_label
                    )
                elif (
                    omit_baselines
                    or (
                        benchmark_object._task_spec.sub_label,
                        benchmark_object._task_spec.num_threads,
                    )
                    in skip_vanilla_tasks
                ):
                    continue

                memory = math.inf
                try:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    mem_begin = torch.cuda.max_memory_allocated() / 2**20
                    benchmark_object._task_spec = replace(
                        benchmark_object._task_spec, env=env
                    )
                    measurement = benchmark_object.blocked_autorange(
                        min_run_time=min_run_time
                    )
                    torch.cuda.synchronize()
                    results.append((metadata, measurement))
                    name = measurement.task_spec.description
                    memory = torch.cuda.max_memory_allocated() / 2**20 - mem_begin
                    measurement.mem_use = memory
                except RuntimeError as e:
                    if not _is_oom_error(e):
                        raise
                    if not quiet:
                        pbar.write("Skipped (OOM)")
                finally:
                    del benchmark_object
                if not quiet:
                    pbar.write(f"{name}: memory used: {memory} MB")
        except RuntimeError as e:
            if not _is_oom_error(e):
                raise
            if not quiet:
                pbar.write("Skipped (OOM)")
        # Display results for benchmarks we just calculated
        if name is not None and not quiet:

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
    if results and optimized_label is not None:
        write_to_path = os.path.join(
            store_results_folder, f"{optimized_label}.{env}.csv"
        )
        _benchmark_results_to_csv(write_to_path, results)
        print(f"Saved results to {write_to_path}")

    if fail_if_regression:
        _fail_if_regressions(
            results, reference=results_compare_to, atol_s=atol_s, rtol=rtol
        )


def _is_oom_error(e):
    return isinstance(
        e, (torch.cuda.OutOfMemoryError, triton.runtime.autotuner.OutOfResources)
    )


def _fail_if_regressions(
    results: List[Any], reference: List[Any], atol_s: float, rtol: float
) -> None:
    def get_measurement_id(r):
        return (
            r[0].get(META_ALGORITHM, "").partition("@")[0],
            r[1].task_spec.label,
            r[1].task_spec.sub_label,
            r[1].task_spec.env,
        )

    id_to_result = {}
    for r in results:
        id_to_result[get_measurement_id(r)] = r[1]

    num_better = 0
    num_worse = 0
    num_nochange = 0
    num_unk = 0
    reference_set = set()
    for ref in reference:
        if ref[1].task_spec.description in BASELINE_DESCRIPTIONS:
            continue
        benchmark_id = get_measurement_id(ref)
        if benchmark_id in reference_set:
            raise ValueError(f"Duplicate benchmark in reference for {benchmark_id}")
        reference_set.add(benchmark_id)
        if benchmark_id not in id_to_result:
            num_unk += 1
            continue
        res = id_to_result[benchmark_id]
        # If significative change
        if abs(ref[1].mean - res.mean) - rtol * ref[1].mean > atol_s:
            is_now_better = res.mean < ref[1].mean
            if is_now_better:
                num_better += 1
            else:
                num_worse += 1
            cmp = "IMPROVED" if is_now_better else "REGRESS "
            print(cmp, benchmark_id, f"ref={ref[1].mean}", f"now={res.mean}")
        else:
            num_nochange += 1

    print("Regression test summary:")
    print(f"  Better   : {num_better}")
    print(f"  No change: {num_nochange}")
    print(f"  Worse    : {num_worse}")
    if num_unk > 0:
        print(f"  (no ref) : {num_unk}")
    benchmarks_run = num_better + num_nochange + num_worse
    if num_worse > 1:
        raise RuntimeError("At least one benchmark regressed!")
    elif num_unk == benchmarks_run:
        raise RuntimeError("No reference found")
    elif benchmarks_run == 0:
        raise RuntimeError("No benchmark was run")


def benchmark_main_helper2(
    name: str,
    functions,
    fw: bool = False,
    bw: bool = False,
    cuda_graph: bool = True,
    **kwargs,
) -> None:
    assert fw or bw

    def handle_case(**case) -> Iterator[benchmark.Timer]:
        for k, benchmark_cls in functions.items():
            try:
                benchmark_object = benchmark_cls(**case, bw=bw)
            except NotSupportedInputError:
                continue
            label = benchmark_object.label
            label += "fw" if fw else ""
            label += "bw" if bw else ""

            def run_one():
                if fw:
                    benchmark_object.fw()
                if bw:
                    benchmark_object.bw()

            if cuda_graph:
                run_one()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    run_one()

                def run_one():
                    g.replay()

            yield benchmark.Timer(
                stmt="fn()",
                globals={
                    "fn": run_one,
                },
                label=label,
                description=k,
                sub_label=benchmark_object.sub_label,
            )

    handle_case.__name__ = name
    benchmark_main_helper(handle_case, **kwargs)


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


DTYPE2STR = {
    torch.bfloat16: "b16",
    torch.half: "f16",
    torch.float32: "f32",
}
