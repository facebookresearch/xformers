# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import compute_wheel_version

THIS_PATH = Path(__file__).resolve()
SOURCE_ROOT_DIR = THIS_PATH.parents[1]

PYTHON_VERSIONS = ["3.9", "3.10"]
PYTORCH_TO_CUDA_VERSIONS = {
    "1.11.0": ["10.2", "11.1", "11.3", "11.5"],
    "1.12.0": ["10.2", "11.3", "11.6"],
    "1.12.1": ["10.2", "11.3", "11.6"],
    "1.13": ["11.6", "11.7"],
}


def conda_docker_image_for_cuda(cuda_version: str) -> str:
    """
    Given a cuda version, return a docker image we could
    build in.
    """

    if cuda_version in ("10.1", "10.2", "11.1"):
        return "pytorch/conda-cuda"
    if cuda_version == "11.3":
        return "pytorch/conda-builder:cuda113"
    if cuda_version == "11.5":
        return "pytorch/conda-builder:cuda115"
    if cuda_version == "11.6":
        return "pytorch/conda-builder:cuda116"
    if cuda_version == "11.7":
        return "pytorch/conda-builder:cuda117"
    raise ValueError(f"Unknown cuda version {cuda_version}")


@dataclass
class Build:
    """
    Represents one configuration of a build, i.e.
    a set of versions of dependent libraries.

    Members:
        conda_always_copy: avoids hard linking which can behave weirdly.
        conda_debug: get added information about package search
        conda_dirty: see intermediate files after build
        build_inside_tree: output in build/ not ../build
        is_release: whether this is an official versioned release
    """

    python_version: str
    pytorch_version: str
    pytorch_channel: str
    cuda_version: str
    cuda_dep_runtime: str

    conda_always_copy: bool = True
    conda_debug: bool = False
    conda_dirty: bool = False
    build_inside_tree: bool = False
    tagged_version: Optional[str] = field(
        default_factory=compute_wheel_version.get_tagged_version
    )

    def _get_build_version(self) -> str:
        if self.tagged_version is not None:
            return self.tagged_version
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        dev_version = compute_wheel_version.get_dev_version()
        return f"{dev_version}+git.{git_hash}"

    def _set_env_for_build(self) -> None:
        """
        NOTE: Variables set here won't be visible in `setup.py`
        UNLESS they are also specified in meta.yaml
        """
        os.environ["BUILD_VERSION"] = self._get_build_version()
        tag = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        os.environ["GIT_TAG"] = tag
        os.environ["PYTORCH_VERSION"] = self.pytorch_version
        os.environ["CU_VERSION"] = self.cuda_version
        os.environ["SOURCE_ROOT_DIR"] = str(SOURCE_ROOT_DIR)

        # At build time, the same major/minor (otherwise we might get a CPU pytorch ...)
        cuda_constraint_build = "=" + ".".join(self.cuda_version.split(".")[:2])
        pytorch_version_tuple = tuple(
            int(v) for v in self.pytorch_version.split(".")[:2]
        )
        if pytorch_version_tuple < (1, 13):
            os.environ[
                "CONDA_CUDA_CONSTRAINT_BUILD"
            ] = f"cudatoolkit{cuda_constraint_build}"
            os.environ[
                "CONDA_CUDA_CONSTRAINT_RUN"
            ] = f"cudatoolkit{self.cuda_dep_runtime}"
        else:
            os.environ[
                "CONDA_CUDA_CONSTRAINT_BUILD"
            ] = f"pytorch-cuda{cuda_constraint_build}"
            os.environ[
                "CONDA_CUDA_CONSTRAINT_RUN"
            ] = f"pytorch-cuda{self.cuda_dep_runtime}"

        if self.conda_always_copy:
            os.environ["CONDA_ALWAYS_COPY"] = "true"

    def _get_build_args(self) -> List[str]:
        args = [
            "conda",
            "build",
            "-c",
            self.pytorch_channel,
            "-c",
            "nvidia",
            "--python",
            self.python_version,
            "--no-anaconda-upload",
        ]
        if self.conda_debug:
            args += ["--debug"]
        if self.conda_dirty:
            args += ["--dirty"]
        if not self.build_inside_tree:
            args += ["--croot", "../build"]
        return args + ["packaging/xformers"]

    def do_build(self) -> None:
        self._set_env_for_build()
        args = self._get_build_args()
        print(args)
        subprocess.check_call(args)

    def move_artifacts_to_store(self, store_pytorch_package: bool) -> None:
        """
        Run after a build to move the built package, and, if using nightly, the
        used PyTorch package, to a location where they will be recognized
        as build artifacts.
        """
        print("moving artifacts")
        assert not self.build_inside_tree
        artifacts = Path("packages")
        artifacts.mkdir(exist_ok=True)
        for filename in Path("../build/linux-64").resolve().glob("*.tar.bz2"):
            print("moving", filename, "to", artifacts)
            shutil.move(filename, artifacts)
        if store_pytorch_package:
            for filename in Path("/opt/conda/pkgs").glob("pytorch-[12].*.tar.bz2"):
                print("moving", filename, "to", artifacts)
                shutil.move(filename, artifacts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the conda package.")
    parser.add_argument(
        "--python", metavar="3.X", required=True, help="python version e.g. 3.10"
    )
    parser.add_argument(
        "--cuda-dep-runtime", metavar="1X.Y", required=True, help="eg '>=11.7,<11.9"
    )
    parser.add_argument(
        "--cuda", metavar="1X.Y", required=True, help="cuda version e.g. 11.3"
    )
    parser.add_argument(
        "--pytorch", metavar="1.Y.Z", required=True, help="PyTorch version e.g. 1.11.0"
    )
    parser.add_argument(
        "--build-inside-tree",
        action="store_true",
        help="Build in build/ instead of ../build/",
    )
    parser.add_argument(
        "--store",
        action="store_true",
        help="position artifact to store",
    )
    parser.add_argument(
        "--store-pytorch-package",
        action="store_true",
        help="position artifact to store",
    )
    parser.add_argument(
        "--pytorch-channel", default="pytorch", help="Use 'pytorch-nightly' for nightly"
    )
    args = parser.parse_args()

    pkg = Build(
        pytorch_channel=args.pytorch_channel,
        python_version=args.python,
        pytorch_version=args.pytorch,
        cuda_version=args.cuda,
        build_inside_tree=args.build_inside_tree,
        cuda_dep_runtime=args.cuda_dep_runtime,
    )

    pkg.do_build()
    pkg.move_artifacts_to_store(store_pytorch_package=args.store_pytorch_package)


# python packaging/conda/build_conda.py  --cuda 11.6 --python 3.10 --pytorch 1.12.1
# python packaging/conda/build_conda.py  --cuda 11.3 --python 3.9 --pytorch 1.12.1  # <= the dino one
# python packaging/conda/build_conda.py  --cuda 11.6 --python 3.10 --pytorch 1.11.0

# Note this does the build outside the root of the tree.

# TODO:
# - Make a local conda package cache available inside docker
# - do we need builds for both _GLIBCXX_USE_CXX11_ABI values?
# - how to prevent some cpu only builds of pytorch from being discovered?
