# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import subprocess
from pathlib import Path

# TODO: consolidate with the code in build_conda.py
THIS_PATH = Path(__file__).resolve()
version = (THIS_PATH.parents[1] / "version.txt").read_text().strip()


try:
    tag = subprocess.check_output(["git", "describe", "--tags"], text=True).strip()
except subprocess.CalledProcessError:  # no tag
    tag = ""

if tag:
    assert version == tag, "The version in version.txt does not match the given tag"
    print(tag, end="")
    exit(0)


num_commits = subprocess.check_output(
    ["git", "rev-list", "--count", "HEAD"], text=True
).strip()
# increment patch
last_part = version.rindex(".") + 1
version = version[:last_part] + str(1 + int(version[last_part:]))

print(f"{version}rc{num_commits}", end="")
