# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
import subprocess

import xformers.benchmarks.utils as utils


class NamedObject:
    def __init__(self, name) -> None:
        self.__name__ = name


def git_file_at(filename: str, ref: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "show", f"{ref}:{filename}"], text=True
        ).strip()
    except subprocess.CalledProcessError:
        return ""  # File does not exist in that revision


GITHUB_BASE_REF = subprocess.check_output(
    ["git", "rev-parse", "origin/" + os.environ["GITHUB_BASE_REF"]], text=True
).strip()
XFORMERS_BENCHMARKS_CACHE = os.environ["XFORMERS_BENCHMARKS_CACHE"]
GITHUB_CURRENT_REF = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], text=True
).strip()

for f in glob.glob(os.path.join(XFORMERS_BENCHMARKS_CACHE, "*", "*.csv")):
    before = git_file_at(f, ref=GITHUB_BASE_REF)
    now = git_file_at(f, ref=GITHUB_CURRENT_REF)
    if before == "" or before == now:
        continue
    benchmark_name = os.path.basename(os.path.dirname(f))

    print("#" * 100)
    print(f"# UPDATED: {f}")
    print("#" * 100)

    filename_before = f.replace("reference", "before")
    filename_now = f.replace("reference", "now")
    with open(filename_before, "w+") as fd:
        fd.write(before)
    with open(filename_now, "w+") as fd:
        fd.write(now)
    utils.benchmark_run_and_compare(
        benchmark_fn=NamedObject(benchmark_name),
        cases=[],
        compare=[
            os.path.basename(filename_before)[: -len(".csv")],
            os.path.basename(filename_now)[: -len(".csv")],
        ],
    )
