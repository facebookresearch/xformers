# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import subprocess
from pathlib import Path
from typing import Optional

# TODO: consolidate with the code in build_conda.py
THIS_PATH = Path(__file__).resolve()
version_from_file = (THIS_PATH.parents[1] / "version.txt").read_text().strip()


def get_tagged_version() -> Optional[str]:
    """
    Return whether we are at an exact version (namely the version variable).
    """
    try:
        tag = subprocess.check_output(
            ["git", "describe", "--tags", "--exact-match", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:  # no tag
        return None

    if not tag.startswith("v"):
        return None
    return tag[1:]


def get_dev_version() -> str:
    assert ".dev" not in version_from_file
    num_commits = subprocess.check_output(
        ["git", "rev-list", "--count", "HEAD"], text=True
    ).strip()
    return f"{version_from_file}.dev{num_commits}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", choices=["tag", "dev", "tag,dev"], required=False, default="tag,dev"
    )
    args = parser.parse_args()

    if "tag" in args.source:
        tagged_version = get_tagged_version()
        if args.source == "tag" and tagged_version is None:
            raise ValueError("No tag found")
    else:
        tagged_version = None
    if tagged_version is not None:
        print(tagged_version, end="")
    else:
        print(get_dev_version(), end="")
