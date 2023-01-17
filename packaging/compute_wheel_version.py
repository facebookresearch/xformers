# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import subprocess
from pathlib import Path
from typing import Optional

from packaging import version

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

    # Should match the version in `version.txt`
    # (except for the suffix like `rc` tag)
    assert (
        version.parse(version_from_file).release == version.parse(tag[1:]).release
    ), f"The version in version.txt ({version_from_file}) does not match the given tag ({tag})"
    return tag[1:]


def get_dev_version() -> str:
    num_commits = subprocess.check_output(
        ["git", "rev-list", "--count", "HEAD"], text=True
    ).strip()
    return f"{version_from_file}.dev{num_commits}"


if __name__ == "__main__":
    tagged_version = get_tagged_version()
    if tagged_version is not None:
        print(tagged_version, end="")
    else:
        print(get_dev_version(), end="")
