# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from setuptools import setup

setup(
    name="xformers_experimental",
    author="Facebook AI Research",
    version="0.0.1",
    packages=["ragged_inference", "mem_efficient_attention"],
    install_requires=[],
    scripts=[],
    python_requires=">=3.6",
)
