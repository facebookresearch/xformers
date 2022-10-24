#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import distutils.command.clean
import glob
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

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


if os.getenv("BUILD_VERSION"):
    version = os.getenv("BUILD_VERSION")
else:
    version_txt = os.path.join(this_dir, "version.txt")
    with open(version_txt) as f:
        version = f.readline().strip()


def write_version_file():
    version_path = os.path.join(this_dir, "xformers", "version.py")
    with open(version_path, "w") as f:
        f.write("# noqa: C801\n")
        f.write(f'__version__ = "{version}"\n')
        tag = os.getenv("GIT_TAG")
        if tag is not None:
            f.write(f'git_tag = "{tag}"\n')


def get_cuda_version(cuda_dir) -> int:
    nvcc_bin = "nvcc" if cuda_dir is None else cuda_dir + "/bin/nvcc"
    raw_output = subprocess.check_output([nvcc_bin, "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = int(release[0])
    bare_metal_minor = int(release[1][0])

    assert bare_metal_minor < 100
    return bare_metal_major * 100 + bare_metal_minor


def get_flash_attention_extensions(cuda_version: int, extra_compile_args):
    # Figure out default archs to target
    DEFAULT_ARCHS_LIST = ""
    if cuda_version > 1100:
        DEFAULT_ARCHS_LIST = "7.5;8.0;8.6"
    elif cuda_version == 1100:
        DEFAULT_ARCHS_LIST = "7.5;8.0"
    else:
        return []

    if os.getenv("XFORMERS_DISABLE_FLASH_ATTN", "0") != "0":
        return []

    archs_list = os.environ.get("TORCH_CUDA_ARCH_LIST", DEFAULT_ARCHS_LIST)
    nvcc_archs_flags = []
    for arch in archs_list.replace(" ", ";").split(";"):
        assert len(arch) >= 3, f"Invalid sm version: {arch}"

        num = 10 * int(arch[0]) + int(arch[2])
        # Need at least 7.5
        if num < 75:
            continue
        nvcc_archs_flags.append(f"-gencode=arch=compute_{num},code=sm_{num}")
        if arch.endswith("+PTX"):
            nvcc_archs_flags.append(f"-gencode=arch=compute_{num},code=compute_{num}")
    if not nvcc_archs_flags:
        return []

    this_dir = os.path.dirname(os.path.abspath(__file__))
    flash_root = os.path.join(this_dir, "third_party", "flash-attention")
    if not os.path.exists(flash_root):
        raise RuntimeError(
            "flashattention submodule not found. Did you forget "
            "to run `git submodule update --init --recursive` ?"
        )

    nvcc_platform_dependant_args: List[str] = []
    if sys.platform == "win32":
        nvcc_platform_dependant_args.append("-std=c++17")

    return [
        CUDAExtension(
            name="xformers._C_flashattention",
            sources=[
                os.path.join(this_dir, "third_party", "flash-attention", path)
                for path in [
                    "csrc/flash_attn/fmha_api.cpp",
                    "csrc/flash_attn/src/fmha_fprop_fp16_kernel.sm80.cu",
                    "csrc/flash_attn/src/fmha_dgrad_fp16_kernel_loop.sm80.cu",
                    "csrc/flash_attn/src/fmha_block_fprop_fp16_kernel.sm80.cu",
                    "csrc/flash_attn/src/fmha_block_dgrad_fp16_kernel_loop.sm80.cu",
                ]
            ],
            extra_compile_args={
                **extra_compile_args,
                "nvcc": extra_compile_args.get("nvcc", [])
                + [
                    "-O3",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "--ptxas-options=-v",
                    "-lineinfo",
                ]
                + nvcc_platform_dependant_args
                + nvcc_archs_flags,
            },
            include_dirs=[
                Path(flash_root) / "csrc" / "flash_attn",
                Path(flash_root) / "csrc" / "flash_attn" / "src",
                #            Path(flash_root) / 'csrc' / 'flash_attn' / 'cutlass' / 'include',
                Path(this_dir) / "third_party" / "cutlass" / "include",
            ],
        )
    ]


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "xformers", "components")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))

    source_cpu = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"), recursive=True)

    sources = main_file + source_cpu

    source_cuda = glob.glob(
        os.path.join(extensions_dir, "**", "cuda", "**", "*.cu"), recursive=True
    )

    sputnik_dir = os.path.join(this_dir, "third_party", "sputnik")
    cutlass_dir = os.path.join(this_dir, "third_party", "cutlass", "include")
    cutlass_examples_dir = os.path.join(this_dir, "third_party", "cutlass", "examples")
    if not os.path.exists(cutlass_dir):
        raise RuntimeError(
            f"CUTLASS submodule not found at {cutlass_dir}. "
            "Did you forget to run "
            "`git submodule update --init --recursive` ?"
        )

    extension = CppExtension

    define_macros = []

    extra_compile_args = {"cxx": ["-O3"]}
    if sys.platform == "win32":
        define_macros += [("xformers_EXPORTS", None)]
        extra_compile_args["cxx"].extend(["/MP", "/Zc:lambda", "/Zc:preprocessor"])
    elif "OpenMP not found" not in torch.__config__.parallel_info():
        extra_compile_args["cxx"].append("-fopenmp")

    include_dirs = [extensions_dir]
    ext_modules = []

    if (torch.cuda.is_available() and ((CUDA_HOME is not None))) or os.getenv(
        "FORCE_CUDA", "0"
    ) == "1":
        extension = CUDAExtension
        sources += source_cuda
        include_dirs += [sputnik_dir, cutlass_dir, cutlass_examples_dir]
        nvcc_flags = [
            "-DHAS_PYTORCH",
            "--use_fast_math",
            "--generate-line-info",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "--extended-lambda",
        ]
        if os.getenv("XFORMERS_ENABLE_DEBUG_ASSERTIONS", "0") != "1":
            nvcc_flags.append("-DNDEBUG")
        nvcc_flags += shlex.split(os.getenv("NVCC_FLAGS", ""))
        cuda_version = get_cuda_version(CUDA_HOME)
        if cuda_version >= 1102:
            nvcc_flags += [
                "--threads",
                "4",
                "--ptxas-options=-v",
            ]
        if sys.platform == "win32":
            nvcc_flags += [
                "-std=c++17",
                "-Xcompiler",
                "/Zc:lambda",
                "-Xcompiler",
                "/Zc:preprocessor",
            ]
        extra_compile_args["nvcc"] = nvcc_flags

        ext_modules += get_flash_attention_extensions(
            cuda_version=cuda_version, extra_compile_args=extra_compile_args
        )

    sources = [os.path.join(extensions_dir, s) for s in sources]

    ext_modules.append(
        extension(
            "xformers._C",
            sorted(sources),
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    )

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
    write_version_file()
    setuptools.setup(
        name="xformers",
        description="XFormers: A collection of composable Transformer building blocks.",
        version=version,
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
