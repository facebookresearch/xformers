#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import distutils.command.clean
import glob
import os
import re
import shutil
import sys

import setuptools
import torch
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

this_dir = os.path.dirname(os.path.abspath(__file__))


def fetch_requirements():
    with open("requirements.txt") as f:
        reqs = f.read().strip().split("\n")
    return reqs


# https://packaging.python.org/guides/single-sourcing-package-version/
def find_version(version_file_path):
    with open(version_file_path) as version_file:
        version_match = re.search(
            r"^__version__ = ['\"]([^'\"]*)['\"]", version_file.read(), re.M
        )
        # The following is used to build release packages.
        # Users should never use it.
        suffix = os.getenv("XFORMERS_VERSION_SUFFIX", "")
        if version_match:
            return version_match.group(1) + suffix
        raise RuntimeError("Unable to find version string.")


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(
        this_dir, "xformers", "components", "attention", "csrc"
    )

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))

    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp")) + glob.glob(
        os.path.join(extensions_dir, "autograd", "*.cpp")
    )

    sources = main_file + source_cpu
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sputnik_dir = os.path.join(this_dir, "third_party", "sputnik")

    extension = CppExtension

    define_macros = []

    extra_compile_args = {"cxx": ["-O3"]}
    if sys.platform == "win32":
        define_macros += [("xformers_EXPORTS", None)]
        extra_compile_args["cxx"].append("/MP")
    elif "OpenMP not found" not in torch.__config__.parallel_info():
        extra_compile_args["cxx"].append("-fopenmp")

    include_dirs = [extensions_dir]

    if (torch.cuda.is_available() and ((CUDA_HOME is not None))) or os.getenv(
        "FORCE_CUDA", "0"
    ) == "1":
        extension = CUDAExtension
        sources += source_cuda
        include_dirs += [sputnik_dir]
        nvcc_flags = os.getenv("NVCC_FLAGS", "")
        if nvcc_flags == "":
            nvcc_flags = []
        else:
            nvcc_flags = nvcc_flags.split(" ")
        extra_compile_args["nvcc"] = nvcc_flags

    sources = [os.path.join(extensions_dir, s) for s in sources]

    ext_modules = [
        extension(
            "xformers._C",
            sorted(sources),
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


class clean(distutils.command.clean.clean):  # type: ignore
    def run(self):
        if os.path.exists(".gitignore"):
            with open(".gitignore", "r") as f:
                ignores = f.read()
                for wildcard in filter(None, ignores.split("\n")):
                    for filename in glob.glob(wildcard):
                        try:
                            os.remove(filename)
                        except OSError:
                            shutil.rmtree(filename, ignore_errors=True)

        # It's an old-style class in Python 2.7...
        distutils.command.clean.clean.run(self)


if __name__ == "__main__":
    setuptools.setup(
        name="xformers",
        description="XFormers: A collection of composable Transformer building blocks.",
        version=find_version(os.path.join(this_dir, "xformers", "__init__.py")),
        setup_requires=[],
        install_requires=fetch_requirements(),
        packages=setuptools.find_packages(exclude=("tests", "tests.*")),
        ext_modules=get_extensions(),
        cmdclass={
            "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
            "clean": clean,
        },
        url="https://facebookresearch.github.io/xformers/",
        python_requires=">=3.6",
        author="Facebook AI Research",
        author_email="lefaudeux@fb.com",
        long_description="XFormers: A collection of composable Transformer building blocks."
        + "XFormers aims at being able to reproduce most architectures in the Transformer-family SOTA,"
        + "defined as compatible and combined building blocks as opposed to monolithic models",
        long_description_content_type="text/markdown",
        classifiers=[
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "License :: OSI Approved :: BSD License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Operating System :: OS Independent",
        ],
        zip_safe=False,
    )
