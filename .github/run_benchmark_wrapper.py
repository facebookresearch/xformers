# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
import shlex
import subprocess
import sys

import torch

import xformers

# Build failed - return early
if not xformers._has_cpp_library:
    print("xFormers wasn't built correctly - can't run benchmarks")
    sys.exit(0)

benchmark_script = os.path.join("xformers", "benchmarks", sys.argv[1])
benchmark_fn = sys.argv[2]
label = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()[:8]
cmd = [
    sys.executable,
    benchmark_script,
    "--label",
    label,
    "--fn",
    benchmark_fn,
    "--fail_if_regression",
    "--quiet",
]
env = (
    torch.cuda.get_device_name(torch.cuda.current_device())
    .replace(" ", "_")
    .replace("-", "_")
    .replace(".", "_")
)

# Figure out the name of the baseline
pattern = os.path.join(os.environ["XFORMERS_BENCHMARKS_CACHE"], benchmark_fn, "*.csv")
ref_names = glob.glob(pattern)
baseline_names = set(
    os.path.basename(s)[: -len(".csv")]
    for s in ref_names
    # Only compare to benchmark data on same hardware
    if env in os.path.basename(s)
)
if baseline_names:
    if len(baseline_names) > 1:
        raise RuntimeError(
            f"Supplied more than one reference for this benchmark: {','.join(baseline_names)}"
        )
    cmd += ["--compare", ",".join(baseline_names)]

print("EXEC:", shlex.join(cmd))

retcode = 0
try:
    subprocess.check_call(cmd)
except subprocess.CalledProcessError as e:
    retcode = e.returncode

# Remove original benchmark files
for f in ref_names:
    os.remove(f)
# Rename new ones as 'ref'
for f in glob.glob(pattern):
    os.rename(f, f.replace(label, "reference"))

sys.exit(retcode)
