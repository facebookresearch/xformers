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
from typing import List, Optional

import setuptools
import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDA_HOME,
    CUDAExtension,
    ROCM_HOME,
)

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:
    _bdist_wheel = None

this_dir = os.path.dirname(__file__)
pt_attn_compat_file_path = os.path.join(
    this_dir, "xformers", "ops", "fmha", "torch_attention_compat.py"
)

# Define the module name
module_name = "torch_attention_compat"

# Load the module
spec = importlib.util.spec_from_file_location(module_name, pt_attn_compat_file_path)
assert spec is not None
attn_compat_module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = attn_compat_module
assert spec.loader is not None
spec.loader.exec_module(attn_compat_module)


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


def get_hip_version(rocm_dir) -> Optional[str]:
    hipcc_bin = "hipcc" if rocm_dir is None else os.path.join(rocm_dir, "bin", "hipcc")
    try:
        raw_output = subprocess.check_output(
            [hipcc_bin, "--version"], universal_newlines=True
        )
    except Exception as e:
        print(
            f"hip installation not found: {e} ROCM_PATH={os.environ.get('ROCM_PATH')}"
        )
        return None
    for line in raw_output.split("\n"):
        if "HIP version" in line:
            return line.split()[-1]
    return None


def rename_cpp_cu(cpp_files):
    for entry in cpp_files:
        shutil.copy(entry, os.path.splitext(entry)[0] + ".cu")


def get_extensions():
    extensions_dir = os.path.join("xformers", "csrc")

    sources = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"), recursive=True)
    source_cuda = glob.glob(os.path.join(extensions_dir, "**", "*.cu"), recursive=True)
    fmha_source_cuda = glob.glob(
        os.path.join(extensions_dir, "**", "fmha", "**", "*.cu"), recursive=True
    )
    exclude_files = ["small_k.cu", "decoder.cu", "attention_cutlass_rand_uniform.cu"]
    fmha_source_cuda = [
        c
        for c in fmha_source_cuda
        if not any(exclude_file in c for exclude_file in exclude_files)
    ]

    source_hip = glob.glob(
        os.path.join(extensions_dir, "attention", "hip_*", "**", "*.cpp"),
        recursive=True,
    )

    source_hip_generated = glob.glob(
        os.path.join(extensions_dir, "attention", "hip_*", "**", "*.cu"),
        recursive=True,
    )
    # avoid the temporary .cu files generated under xformers/csrc/attention/hip_fmha
    source_cuda = list(set(source_cuda) - set(source_hip_generated))
    sources = list(set(sources) - set(source_hip))

    xformers_pt_cutlass_attn = os.getenv("XFORMERS_PT_CUTLASS_ATTN")
    # By default, we try to link to torch internal CUTLASS attention implementation
    # and silently switch to local CUTLASS attention build if no compatibility
    # If we force 'torch CUTLASS switch' then setup will fail when no compatibility
    if (
        xformers_pt_cutlass_attn is None or xformers_pt_cutlass_attn == "1"
    ) and attn_compat_module.is_pt_cutlass_compatible(
        force=xformers_pt_cutlass_attn == "1"
    ):
        source_cuda = list(set(source_cuda) - set(fmha_source_cuda))

    if "XFORMERS_SELECTIVE_BUILD" in os.environ:
        pattern = os.environ["XFORMERS_SELECTIVE_BUILD"]
        source_cuda = [f for f in source_cuda if pattern in str(f)]

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
    hip_version = None

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

        # For now we enforce the PyTorch stable ABI only for CUDA builds (not HIP).
        stable_args = [
            "-DTORCH_STABLE_ONLY",
            "-DTORCH_TARGET_VERSION=0x020a000000000000",
        ]
        extra_compile_args["cxx"].extend(stable_args)
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
    elif (
        torch.version.hip
        and os.getenv("XFORMERS_CK_FLASH_ATTN", "1") == "1"
        and (torch.cuda.is_available() or os.getenv("HIP_ARCHITECTURES", "") != "")
    ):
        rename_cpp_cu(source_hip)
        hip_version = get_hip_version(ROCM_HOME)

        source_hip_cu = []
        for ff in source_hip:
            source_hip_cu += [ff.replace(".cpp", ".cu")]

        extension = CUDAExtension
        sources += source_hip_cu
        include_dirs += [
            Path(this_dir) / "xformers" / "csrc" / "attention" / "hip_fmha",
            Path(this_dir) / "xformers" / "csrc" / "attention" / "hip_decoder",
        ]

        include_dirs += [
            Path(this_dir) / "third_party" / "composable_kernel_tiled" / "include"
        ]

        cc_flag = ["-DBUILD_PYTHON_PACKAGE"]
        use_rtn_bf16_convert = os.getenv("ENABLE_HIP_FMHA_RTN_BF16_CONVERT", "0")
        if use_rtn_bf16_convert == "1":
            cc_flag += ["-DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=3"]

        arch_list = os.getenv("HIP_ARCHITECTURES", "native").split()

        offload_compress_flag = []
        if hip_version >= "6.2.":
            offload_compress_flag = ["--offload-compress"]

        extra_compile_args["nvcc"] = [
            "-O3",
            "-std=c++17",
            *[f"--offload-arch={arch}" for arch in arch_list],
            *offload_compress_flag,
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-DCK_TILE_FMHA_FWD_FAST_EXP2=1",
            "-fgpu-flush-denormals-to-zero",
            "-Werror",
            "-Wc++11-narrowing",
            "-Woverloaded-virtual",
            "-mllvm",
            "-enable-post-misched=0",
            "-mllvm",
            "-amdgpu-early-inline-all=true",
            "-mllvm",
            "-amdgpu-function-calls=false",
            "-mllvm",
            "-greedy-reverse-local-assignment=1",
        ] + cc_flag

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
            "hip": hip_version,
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


class bdist_wheel_abi_none(_bdist_wheel if _bdist_wheel else object):  # type: ignore[misc]
    """
    Custom wheel builder that tags wheels as ABI-independent despite containing compiled code.
    The compiled extensions are plain shared libraries (.so/.dll) that use only PyTorch's
    TORCH_LIBRARY mechanism, with no Python C API dependencies. This allows the same wheel
    to work across different Python versions and variants (including free-threaded builds).
    """

    def get_tag(self):
        if _bdist_wheel is None:
            raise RuntimeError("wheel package is required to build wheels")

        # Get the default tags from parent class
        python_tag, abi_tag, plat_tag = super().get_tag()

        # Override ABI tag to 'none' since our .so files have no Python ABI dependency
        # Use 'py39' as python tag to indicate minimum Python version (3.9+)
        # Keep platform tag since we have platform-specific compiled code
        return "py39", "none", plat_tag


class BuildExtensionWithExtraFiles(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        self.xformers_build_metadata = kwargs.pop("extra_files")
        self.pkg_name = "xformers"
        super().__init__(*args, **kwargs)

    def get_export_symbols(self, ext):
        # Don't export PyInit_* symbols since our extension doesn't use the
        # Python C API. It registers operators with PyTorch via
        # STABLE_TORCH_LIBRARY_FRAGMENT and is loaded via torch.ops.load_library().
        return []

    def build_extensions(self) -> None:
        super().build_extensions()

        for filename, content in self.xformers_build_metadata.items():
            with open(
                os.path.join(self.build_lib, self.pkg_name, filename), "w+"
            ) as fp:
                fp.write(content)

    def copy_extensions_to_source(self) -> None:
        """
        Used for `pip install -e .`
        Copies everything we built back into the source repo
        """
        build_py = self.get_finalized_command("build_py")
        package_dir = build_py.get_package_dir(self.pkg_name)

        for filename in self.xformers_build_metadata.keys():
            inplace_file = os.path.join(package_dir, filename)
            regular_file = os.path.join(self.build_lib, self.pkg_name, filename)
            self.copy_file(regular_file, inplace_file, level=self.verbose)
        super().copy_extensions_to_source()

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
    setuptools.setup(
        name="xformers",
        description="XFormers: A collection of composable Transformer building blocks.",
        version=version,
        install_requires=fetch_requirements(),
        packages=setuptools.find_packages(exclude=("tests*", "benchmarks*")),
        ext_modules=extensions,
        cmdclass={
            "build_ext": BuildExtensionWithExtraFiles.with_options(
                no_python_abi_suffix=True,
                extra_files={
                    "cpp_lib.json": json.dumps(extensions_metadata),
                    "version.py": generate_version_py(version),
                },
            ),
            "bdist_wheel": bdist_wheel_abi_none,
            "clean": clean,
        },
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
