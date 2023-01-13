#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import distutils.command.clean
import glob
import importlib.util
import json
import os
import platform
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

this_dir = os.path.dirname(__file__)


def get_extra_nvcc_flags_for_build_type() -> List[str]:
    build_type = os.environ.get("XFORMERS_BUILD_TYPE", "RelWithDebInfo").lower()
    if build_type == "relwithdebinfo":
        return ["--generate-line-info"]
    elif build_type == "release":
        return []
    else:
        raise ValueError(f"Unknown build type: {build_type}")


def fetch_requirements():
    with open("requirements.txt") as f:
        reqs = f.read().strip().split("\n")
    return reqs


def get_local_version_suffix() -> str:
    date_suffix = datetime.datetime.now().strftime("%Y%m%d")
    git_hash = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent
    ).decode("ascii")[:-1]
    return f"+{git_hash}.d{date_suffix}"


def write_version_file(version: str):
    version_path = os.path.join(this_dir, "xformers", "version.py")
    with open(version_path, "w") as f:
        f.write("# noqa: C801\n")
        f.write(f'__version__ = "{version}"\n')
        tag = os.getenv("GIT_TAG")
        if tag is not None:
            f.write(f'git_tag = "{tag}"\n')


def symlink_package(name: str, path: Path, is_building_wheel: bool) -> None:
    cwd = Path(__file__).parent
    path_from = cwd / path
    path_to = os.path.join(cwd, *name.split("."))

    try:
        if os.path.islink(path_to):
            os.unlink(path_to)
        elif os.path.isdir(path_to):
            shutil.rmtree(path_to)
        else:
            os.remove(path_to)
    except FileNotFoundError:
        pass
    # OSError: [WinError 1314] A required privilege is not held by the client
    # Windows requires special permission to symlink. Fallback to copy
    # When building wheels for linux 3.7 and 3.8, symlinks are not included
    # So we force a copy, see #611
    use_symlink = os.name != "nt" and not is_building_wheel
    if use_symlink:
        os.symlink(src=path_from, dst=path_to)
    else:
        shutil.copytree(src=path_from, dst=path_to)


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

    flash_root = os.path.join(this_dir, "third_party", "flash-attention")
    if not os.path.exists(flash_root):
        raise RuntimeError(
            "flashattention submodule not found. Did you forget "
            "to run `git submodule update --init --recursive` ?"
        )

    return [
        CUDAExtension(
            name="xformers._C_flashattention",
            sources=[
                os.path.join("third_party", "flash-attention", path)
                for path in [
                    "csrc/flash_attn/fmha_api.cpp",
                    "csrc/flash_attn/src/fmha_fwd_hdim32.cu",
                    "csrc/flash_attn/src/fmha_fwd_hdim64.cu",
                    "csrc/flash_attn/src/fmha_fwd_hdim128.cu",
                    "csrc/flash_attn/src/fmha_bwd_hdim32.cu",
                    "csrc/flash_attn/src/fmha_bwd_hdim64.cu",
                    "csrc/flash_attn/src/fmha_bwd_hdim128.cu",
                    "csrc/flash_attn/src/fmha_block_fprop_fp16_kernel.sm80.cu",
                    "csrc/flash_attn/src/fmha_block_dgrad_fp16_kernel_loop.sm80.cu",
                ]
            ],
            extra_compile_args={
                **extra_compile_args,
                "nvcc": extra_compile_args.get("nvcc", [])
                + [
                    "-O3",
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "--ptxas-options=-v",
                ]
                + nvcc_archs_flags
                + get_extra_nvcc_flags_for_build_type(),
            },
            include_dirs=[
                p.absolute()
                for p in [
                    Path(flash_root) / "csrc" / "flash_attn",
                    Path(flash_root) / "csrc" / "flash_attn" / "src",
                    Path(this_dir) / "third_party" / "cutlass" / "include",
                ]
            ],
        )
    ]


def get_extensions():
    extensions_dir = os.path.join("xformers", "csrc")

    sources = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"), recursive=True)
    source_cuda = glob.glob(os.path.join(extensions_dir, "**", "*.cu"), recursive=True)

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
    cuda_version = None

    if (
        (torch.cuda.is_available() and ((CUDA_HOME is not None)))
        or os.getenv("FORCE_CUDA", "0") == "1"
        or os.getenv("TORCH_CUDA_ARCH_LIST", "") != ""
    ):
        extension = CUDAExtension
        sources += source_cuda
        include_dirs += [sputnik_dir, cutlass_dir, cutlass_examples_dir]
        nvcc_flags = [
            "-DHAS_PYTORCH",
            "--use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "--extended-lambda",
            "-D_ENABLE_EXTENDED_ALIGNED_STORAGE",
        ] + get_extra_nvcc_flags_for_build_type()
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
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "env": {
            k: os.environ.get(k)
            for k in [
                "TORCH_CUDA_ARCH_LIST",
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


class BuildExtensionWithMetadata(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        self.xformers_build_metadata = kwargs.pop("xformers_build_metadata")
        self.pkg_name = "xformers"
        self.metadata_json = "cpp_lib.json"
        super().__init__(*args, **kwargs)

    def build_extensions(self) -> None:
        super().build_extensions()
        with open(
            os.path.join(self.build_lib, self.pkg_name, self.metadata_json), "w+"
        ) as fp:
            json.dump(self.xformers_build_metadata, fp)

    def copy_extensions_to_source(self):
        """
        Used for `pip install -e .`
        Copies everything we built back into the source repo
        """
        build_py = self.get_finalized_command("build_py")
        package_dir = build_py.get_package_dir(self.pkg_name)
        inplace_file = os.path.join(package_dir, self.metadata_json)
        regular_file = os.path.join(self.build_lib, self.pkg_name, self.metadata_json)
        self.copy_file(regular_file, inplace_file, level=self.verbose)
        super().copy_extensions_to_source()


if __name__ == "__main__":

    try:
        # when installing as a source distribution, the version module should exist
        # Let's import it manually to not trigger the load of the C++
        # library - which does not exist yet, and creates a WARNING
        spec = importlib.util.spec_from_file_location(
            "xformers_version", os.path.join(this_dir, "xformers", "version.py")
        )
        if spec is None or spec.loader is None:
            raise FileNotFoundError()
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        version = module.__version__
    except FileNotFoundError:
        if os.getenv("BUILD_VERSION"):  # In CI
            version = os.getenv("BUILD_VERSION", "0.0.0")
        else:
            version_txt = os.path.join(this_dir, "version.txt")
            with open(version_txt) as f:
                version = f.readline().strip()
            version += get_local_version_suffix()
        write_version_file(version)

    is_building_wheel = "bdist_wheel" in sys.argv
    # Embed a fixed version of flash_attn
    # NOTE: The correct way to do this would be to use the `package_dir`
    # parameter in `setuptools.setup`, but this does not work when
    # developing in editable mode
    # See: https://github.com/pypa/pip/issues/3160 (closed, but not fixed)
    symlink_package(
        "xformers._flash_attn",
        Path("third_party") / "flash-attention" / "flash_attn",
        is_building_wheel,
    )
    extensions, extensions_metadata = get_extensions()
    setuptools.setup(
        name="xformers",
        description="XFormers: A collection of composable Transformer building blocks.",
        version=version,
        install_requires=fetch_requirements(),
        packages=setuptools.find_packages(
            exclude=("tests*", "benchmarks*", "experimental*")
        ),
        ext_modules=extensions,
        cmdclass={
            "build_ext": BuildExtensionWithMetadata.with_options(
                no_python_abi_suffix=True, xformers_build_metadata=extensions_metadata
            ),
            "clean": clean,
        },
        url="https://facebookresearch.github.io/xformers/",
        python_requires=">=3.7",
        author="Facebook AI Research",
        author_email="oncall+xformers@xmail.facebook.com",
        long_description="XFormers: A collection of composable Transformer building blocks."
        + "XFormers aims at being able to reproduce most architectures in the Transformer-family SOTA,"
        + "defined as compatible and combined building blocks as opposed to monolithic models",
        long_description_content_type="text/markdown",
        classifiers=[
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "License :: OSI Approved :: BSD License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Operating System :: OS Independent",
        ],
        zip_safe=False,
    )
