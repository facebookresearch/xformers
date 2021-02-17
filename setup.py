#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import os
import re

import setuptools

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
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")


if __name__ == "__main__":
    setuptools.setup(
        name="xformers",
        description="XFormers: A collection of composable Transformer building blocks.",
        version=find_version("xformers/__init__.py"),
        setup_requires=[],
        install_requires=fetch_requirements(),
        include_package_data=True,
        packages=setuptools.find_packages(exclude=("tests", "tests.*")),
        ext_modules=[],
        cmdclass={},
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
    )
