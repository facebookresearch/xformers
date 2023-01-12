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


def conda_docker_image_for_cuda(cuda_version):
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


def version_constraint(version):
    """
    Given version "11.3" returns " >=11.3,<11.4"
    """
    last_part = version.rindex(".") + 1
    upper = version[:last_part] + str(1 + int(version[last_part:]))
    return f" >={version},<{upper}"


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
        upload: whether to upload to xformers on anaconda
        is_release: whether this is an official versioned release
    """

    python_version: str
    pytorch_version: str
    cuda_version: str

    conda_always_copy: bool = True
    conda_debug: bool = False
    conda_dirty: bool = False
    build_inside_tree: bool = False
    upload: bool = False
    is_release: bool = field(default_factory=compute_wheel_version.is_exact_version)

    def _get_build_version(self):
        if self.is_release:
            return compute_wheel_version.code_version
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        dev_version = compute_wheel_version.get_dev_version()
        return f"{dev_version}+git.{git_hash}"

    def _set_env_for_build(self):
        if "CUDA_HOME" not in os.environ:
            if "FAIR_ENV_CLUSTER" in os.environ:
                cuda_home = "/public/apps/cuda/" + self.cuda_version
            else:
                # E.g. inside docker
                cuda_home = "/usr/local/cuda-" + self.cuda_version
            assert Path(cuda_home).is_dir
            os.environ["CUDA_HOME"] = cuda_home

        os.environ["TORCH_CUDA_ARCH_LIST"] = "5.0+PTX 6.0 6.1 7.0 7.5 8.0 8.6"
        os.environ["BUILD_VERSION"] = self._get_build_version()
        tag = subprocess.check_output(["git", "describe", "--tags"], text=True).strip()
        os.environ["GIT_TAG"] = tag
        os.environ["PYTORCH_VERSION"] = self.pytorch_version
        os.environ["CU_VERSION"] = self.cuda_version
        os.environ["SOURCE_ROOT_DIR"] = str(SOURCE_ROOT_DIR)
        os.environ["XFORMERS_BUILD_TYPE"] = "Release"
        os.environ["XFORMERS_PACKAGE_FROM"] = "conda"
        cuda_constraint = version_constraint(self.cuda_version)
        pytorch_version_tuple = tuple(int(v) for v in self.pytorch_version.split("."))
        if pytorch_version_tuple < (1, 13):
            os.environ["CONDA_CUDA_CONSTRAINT"] = f"cudatoolkit{cuda_constraint}"
        else:
            os.environ["CONDA_CUDA_CONSTRAINT"] = f"pytorch-cuda{cuda_constraint}"
        os.environ["FORCE_CUDA"] = "1"

        if self.conda_always_copy:
            os.environ["CONDA_ALWAYS_COPY"] = "true"

    def _get_build_args(self):
        args = [
            "conda",
            "build",
            "-c",
            "pytorch",
            "-c",
            "nvidia",
            "--no-anaconda-upload",
            "--python",
            self.python_version,
        ]
        if self.conda_debug:
            args += ["--debug"]
        if self.conda_dirty:
            args += ["--dirty"]
        if not self.build_inside_tree:
            args += ["--croot", "../build"]
        if self.upload:
            if self.is_release:
                args += ["--user", "xformers"]
            else:
                args += ["--user", "xformers", "--label", "dev"]
        return args + ["packaging/xformers"]

    def do_build(self):
        self._set_env_for_build()
        if self.upload:
            subprocess.check_call(
                ["conda", "config", "--set", "anaconda_upload", "yes"]
            )
        args = self._get_build_args()
        print(args)
        subprocess.check_call(args)

    def move_artifacts_to_store(self):
        """run after a build to move artifacts elsewhere"""
        assert not self.build_inside_tree
        artifacts = Path("artifacts")
        artifacts.mkdir(exist_ok=True)
        for filename in Path("../build").resolve().glob("*.bz2"):
            shutil.move(filename, artifacts)

    def build_in_docker(self) -> None:
        filesystem = subprocess.check_output("stat -f -c %T .", shell=True).strip()
        if filesystem in (b"nfs", b"tmpfs"):
            raise ValueError(
                "Cannot run docker here. "
                + "Please work on a local filesystem, e.g. /raid."
            )
        image = conda_docker_image_for_cuda(self.cuda_version)
        args = ["sudo", "docker", "run", "-it", "--rm", "-w", "/m"]
        args += ["-v", f"{str(SOURCE_ROOT_DIR)}:/m", image]
        args += ["python3", str(THIS_PATH.relative_to(SOURCE_ROOT_DIR))]
        self_args = [
            "--cuda",
            self.cuda_version,
            "--pytorch",
            self.pytorch_version,
            "--python",
            self.python_version,
        ]
        args += self_args
        print(args)
        subprocess.check_call(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the conda package.")
    parser.add_argument(
        "--python", metavar="3.X", required=True, help="python version e.g. 3.10"
    )
    parser.add_argument(
        "--cuda", metavar="1X.Y", required=True, help="cuda version e.g. 11.3"
    )
    parser.add_argument(
        "--pytorch", metavar="1.Y.Z", required=True, help="PyTorch version e.g. 1.11.0"
    )
    parser.add_argument(
        "--docker", action="store_true", help="Call this script inside docker."
    )
    parser.add_argument(
        "--build-inside-tree",
        action="store_true",
        help="Build in build/ instead of ../build/",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="whether to upload to xformers anaconda",
    )
    parser.add_argument(
        "--upload-or-store",
        action="store_true",
        help="upload to xformers anaconda if FACEBOOKRESEARCH, else position artifact to store",
    )
    args = parser.parse_args()

    facebookresearch = os.getenv("CIRCLE_PROJECT_USERNAME", "") == "facebookresearch"

    pkg = Build(
        python_version=args.python,
        pytorch_version=args.pytorch,
        cuda_version=args.cuda,
        build_inside_tree=args.build_inside_tree,
        upload=args.upload or (facebookresearch and args.upload_or_store),
    )

    if args.docker:
        pkg.build_in_docker()
    else:
        pkg.do_build()
        if args.upload_or_store and not facebookresearch:
            pkg.move_artifacts_to_store()


# python packaging/conda/build_conda.py  --cuda 11.6 --python 3.10 --pytorch 1.12.1
# python packaging/conda/build_conda.py  --cuda 11.3 --python 3.9 --pytorch 1.12.1  # <= the dino one
# python packaging/conda/build_conda.py  --cuda 11.6 --python 3.10 --pytorch 1.11.0

# Note this does the build outside the root of the tree.

# TODO:
# - Make a local conda package cache available inside docker
# - do we need builds for both _GLIBCXX_USE_CXX11_ABI values?
# - how to prevent some cpu only builds of pytorch from being discovered?
