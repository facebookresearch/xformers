#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import distutils.command.clean
import glob
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Dict, List

import setuptools
import setuptools.command.build_py
import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDA_HOME,
    CUDAExtension,
)

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:
    _bdist_wheel = object

this_dir = os.path.dirname(__file__)
PKG_NAME = "xformers"


def get_extra_nvcc_flags_for_build_type(cuda_version: int) -> List[str]:
    build_type = os.environ.get("XFORMERS_BUILD_TYPE", "RelWithDebInfo").lower()
    if build_type == "relwithdebinfo":
        if cuda_version >= 1201 and cuda_version < 1202:
            print(
                "Looks like we are using CUDA 12.1 which segfaults when provided with"
                " the -generate-line-info flag. Disabling it."
            )
            return []
        return ["--generate-line-info"]
    elif build_type == "release":
        return []
    elif build_type == "debug":
        return ["--device-debug"]
    else:
        raise ValueError(f"Unknown build type: {build_type}")


def fetch_requirements():
    with open("requirements.txt") as f:
        reqs = f.read().strip().split("\n")
    return reqs


def get_local_version_suffix() -> str:
    if not (Path(__file__).parent / ".git").is_dir():
        # Most likely installing from a source distribution
        return ""
    date_suffix = datetime.datetime.now().strftime("%Y%m%d")
    git_hash = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent
    ).decode("ascii")[:-1]
    return f"+{git_hash}.d{date_suffix}"


def generate_version_py(version: str) -> str:
    content = "# noqa: C801\n"
    content += f'__version__ = "{version}"\n'
    tag = os.getenv("GIT_TAG")
    if tag is not None:
        content += f'git_tag = "{tag}"\n'
    return content


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


def is_fairinternal_only(path: str) -> bool:
    """Whether the sync to the open-source repo strips this file."""
    return any("fairinternal" in part for part in Path(path).parts)


def get_extensions():
    extensions_dir = os.path.join("xformers", "csrc")

    sources = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"), recursive=True)
    source_cuda = glob.glob(os.path.join(extensions_dir, "**", "*.cu"), recursive=True)

    if "XFORMERS_SELECTIVE_BUILD" in os.environ:
        pattern = os.environ["XFORMERS_SELECTIVE_BUILD"]
        source_cuda = [f for f in source_cuda if pattern in str(f)]

    # Only the fairinternal kernels still need compiling. What is left of
    # xformers/csrc in the open-source repo is there to support them and is
    # unused without them, so the open-source build has no extension at all
    # and produces a pure-Python wheel.
    if not any(is_fairinternal_only(f) for f in sources + source_cuda):
        return [], None

    cutlass_dir = os.path.join(this_dir, "third_party", "cutlass", "include")
    cutlass_util_dir = os.path.join(
        this_dir, "third_party", "cutlass", "tools", "util", "include"
    )
    cutlass_examples_dir = os.path.join(this_dir, "third_party", "cutlass", "examples")
    if not os.path.exists(cutlass_dir):
        raise RuntimeError(
            f"CUTLASS submodule not found at {cutlass_dir}. "
            "Did you forget to run "
            "`git submodule update --init --recursive` ?"
        )

    extension = CppExtension

    define_macros = []

    extra_compile_args = {"cxx": ["-O3", "-std=c++17"]}
    if sys.platform == "win32":
        if os.getenv("DISTUTILS_USE_SDK") == "1":
            extra_compile_args = {"cxx": ["-O2", "/std:c++17"]}
        define_macros += [("xformers_EXPORTS", None)]
        extra_compile_args["cxx"].extend(
            ["/MP", "/Zc:lambda", "/Zc:preprocessor", "/Zc:__cplusplus"]
        )
    elif "OpenMP not found" not in torch.__config__.parallel_info():
        extra_compile_args["cxx"].append("-fopenmp")

    include_dirs = [extensions_dir]
    ext_modules = []
    cuda_version = None
    stable_args = [
        "-DTORCH_STABLE_ONLY",
        "-DTORCH_TARGET_VERSION=0x020a000000000000",
    ]
    extra_compile_args["cxx"].extend(stable_args)

    if (
        (
            torch.cuda.is_available()
            and (CUDA_HOME is not None)
            and (torch.version.cuda is not None)
        )
        or os.getenv("FORCE_CUDA", "0") == "1"
        or os.getenv("TORCH_CUDA_ARCH_LIST", "") != ""
    ):
        cuda_version = get_cuda_version(CUDA_HOME)
        extension = CUDAExtension
        sources += source_cuda
        if cuda_version < 1205:
            # swiglu_fairinternal.cu uses cuda::ptx::cp_async_bulk which requires
            # CUDA 12.5
            sources.remove(os.path.join(extensions_dir, "swiglu_fairinternal.cu"))
        include_dirs += [
            cutlass_dir,
            cutlass_util_dir,
            cutlass_examples_dir,
        ]
        nvcc_flags = [
            "-DHAS_PYTORCH",
            "--use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "--extended-lambda",
            "-D_ENABLE_EXTENDED_ALIGNED_STORAGE",
            "-std=c++17",
        ] + get_extra_nvcc_flags_for_build_type(cuda_version)
        if os.getenv("XFORMERS_ENABLE_DEBUG_ASSERTIONS", "0") != "1":
            nvcc_flags.append("-DNDEBUG")
        nvcc_flags += shlex.split(os.getenv("NVCC_FLAGS", ""))
        if cuda_version >= 1102:
            nvcc_flags += [
                "--threads",
                "4",
                "--ptxas-options=-v",
            ]
        if sys.platform == "win32":
            nvcc_flags += [
                "-Xcompiler",
                "/Zc:lambda",
                "-Xcompiler",
                "/Zc:preprocessor",
                "-Xcompiler",
                "/Zc:__cplusplus",
            ]
        extra_compile_args["nvcc"] = nvcc_flags

        extra_compile_args["nvcc"].extend(stable_args + ["-DUSE_CUDA"])

        if (
            "--device-debug" not in nvcc_flags and "-G" not in nvcc_flags
        ):  # (incompatible with -G)
            extra_compile_args["nvcc"] += [
                # Workaround for a regression with nvcc > 11.6
                # See https://github.com/facebookresearch/xformers/issues/712
                "--ptxas-options=-O2",
                "--ptxas-options=-allow-expensive-optimizations=true",
            ]
    ext_modules.append(
        extension(
            "xformers._C",
            sorted(sources),
            include_dirs=[os.path.abspath(p) for p in include_dirs],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    )

    return ext_modules, {
        "version": {
            "cuda": cuda_version,
            "hip": None,
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "env": {
            k: os.environ.get(k)
            for k in [
                "TORCH_CUDA_ARCH_LIST",
                "PYTORCH_ROCM_ARCH",
                "XFORMERS_BUILD_TYPE",
                "XFORMERS_ENABLE_DEBUG_ASSERTIONS",
                "NVCC_FLAGS",
                "XFORMERS_PACKAGE_FROM",
            ]
        },
    }


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


class bdist_wheel_abi_none(_bdist_wheel):
    """
    Custom wheel builder that tags wheels as ABI-independent despite containing compiled code.
    The compiled extensions are plain shared libraries (.so/.dll) that use only PyTorch's
    TORCH_LIBRARY mechanism, with no Python C API dependencies. This allows the same wheel
    to work across different Python versions and variants (including free-threaded builds).
    """

    def finalize_options(self) -> None:
        super().finalize_options()
        if not self.plat_name_supplied and not self.distribution.ext_modules:
            # Without an extension the wheel is pure and would be tagged
            # `any`. Keep naming it after the platform it was built on, so
            # that wheel filenames don't change while CI still builds and
            # publishes one wheel per platform and toolkit. Only the tag is
            # affected: root_is_pure stays true, so the layout is unchanged.
            self.plat_name = sysconfig.get_platform()
            self.plat_name_supplied = True

    def get_tag(self):
        if _bdist_wheel is object:
            raise RuntimeError("wheel package is required to build wheels")

        # Get the default tags from parent class
        python_tag, abi_tag, plat_tag = super().get_tag()

        # Override ABI tag to 'none' since our .so files have no Python ABI dependency
        # Use 'py39' as python tag to indicate minimum Python version (3.9+)
        # Keep platform tag since we have platform-specific compiled code
        return "py39", "none", plat_tag


class BuildPyWithExtraFiles(setuptools.command.build_py.build_py):
    """A `build_py` that also writes our generated files (`version.py`, and
    `cpp_lib.json` when we build an extension).

    These used to be written by `build_ext`, but `build_ext.run()` returns
    immediately when there are no extensions, which is the case for the
    open-source build.
    """

    @classmethod
    def with_options(cls, **options):
        # Same trick as torch's BuildExtension, which setuptools' commands
        # don't provide: setuptools instantiates the cmdclass itself, so bind
        # our arguments here.
        def init_with_options(*args, **kwargs):
            return cls(*args, **{**kwargs, **options})

        return init_with_options

    def __init__(self, *args, **kwargs) -> None:
        self.extra_files: Dict[str, str] = kwargs.pop("extra_files")
        super().__init__(*args, **kwargs)

    def run(self) -> None:
        super().run()
        self._write_extra_files(os.path.join(self.build_lib, PKG_NAME))
        if getattr(self, "editable_mode", False):
            # An editable install imports from the source tree, so the
            # generated files have to land there too. Both destinations
            # are gitignored.
            self._write_extra_files(self.get_package_dir(PKG_NAME))

    def _write_extra_files(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        for filename, content in self.extra_files.items():
            with open(os.path.join(directory, filename), "w") as fp:
                fp.write(content)


class BuildExtensionNoPythonAbi(BuildExtension):
    def get_export_symbols(self, ext):
        # Don't export PyInit_* symbols since our extension doesn't use the
        # Python C API. It registers operators with PyTorch via
        # STABLE_TORCH_LIBRARY_FRAGMENT and is loaded via torch.ops.load_library().
        return []

    def get_ext_filename(self, ext_name):
        # Return plain .so/.pyd names without Python version tags
        # This creates ABI-independent binaries that work with any Python version
        ext_path = ext_name.split(".")
        ext_basename = ext_path[-1]
        ext_dir = os.path.join(*ext_path[:-1]) if len(ext_path) > 1 else ""

        if sys.platform == "win32":
            # Windows: use .pyd extension (required for importlib to find it)
            filename = f"{ext_basename}.pyd"
        else:
            # Linux/Mac: use plain .so extension
            filename = f"{ext_basename}.so"

        return os.path.join(ext_dir, filename) if ext_dir else filename


if __name__ == "__main__":
    if os.getenv("BUILD_VERSION"):  # In CI
        version = os.getenv("BUILD_VERSION", "0.0.0")
    else:
        version_txt = os.path.join(this_dir, "version.txt")
        with open(version_txt) as f:
            version = f.readline().strip()
        version += get_local_version_suffix()

    extensions, extensions_metadata = get_extensions()
    extra_files = {"version.py": generate_version_py(version)}
    if extensions:
        extra_files["cpp_lib.json"] = json.dumps(extensions_metadata)

    cmdclass: Dict[str, type] = {
        "clean": clean,
        "bdist_wheel": bdist_wheel_abi_none,
        "build_py": BuildPyWithExtraFiles.with_options(extra_files=extra_files),
    }
    if extensions:
        cmdclass["build_ext"] = BuildExtensionNoPythonAbi.with_options(
            no_python_abi_suffix=True
        )

    setuptools.setup(
        name="xformers",
        description="XFormers: A collection of composable Transformer building blocks.",
        version=version,
        install_requires=fetch_requirements(),
        packages=setuptools.find_packages(exclude=("tests*", "benchmarks*")),
        ext_modules=extensions,
        cmdclass=cmdclass,
        url="https://facebookresearch.github.io/xformers/",
        python_requires=">=3.9",
        author="Facebook AI Research",
        author_email="oncall+xformers@xmail.facebook.com",
        long_description="XFormers: A collection of composable Transformer building blocks."
        + "XFormers aims at being able to reproduce most architectures in the Transformer-family SOTA,"
        + "defined as compatible and combined building blocks as opposed to monolithic models",
        long_description_content_type="text/markdown",
        classifiers=[
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "License :: OSI Approved :: BSD License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Operating System :: OS Independent",
        ],
        zip_safe=False,
    )
