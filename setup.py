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
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import setuptools
import torch
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

this_dir = os.path.dirname(__file__)
pt_attn_compat_file_path = os.path.join(
    this_dir, "xformers", "ops", "fmha", "torch_attention_compat.py"
)

# Define the module name
module_name = "torch_attention_compat"

# Load the module
spec = importlib.util.spec_from_file_location(module_name, pt_attn_compat_file_path)
attn_compat_module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = attn_compat_module
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


def get_flash_version() -> str:
    flash_dir = Path(__file__).parent / "third_party" / "flash-attention"
    try:
        return subprocess.check_output(
            ["git", "describe", "--tags", "--always"],
            cwd=flash_dir,
        ).decode("ascii")[:-1]
    except subprocess.CalledProcessError:
        version = flash_dir / "version.txt"
        if version.is_file():
            return version.read_text().strip()
        return "v?"


def generate_version_py(version: str) -> str:
    content = "# noqa: C801\n"
    content += f'__version__ = "{version}"\n'
    tag = os.getenv("GIT_TAG")
    if tag is not None:
        content += f'git_tag = "{tag}"\n'
    return content


def symlink_package(name: str, path: Path, is_building_wheel: bool) -> None:
    cwd = Path(__file__).resolve().parent
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


def get_flash_attention_nvcc_archs_flags(cuda_version: int):
    # XXX: Not supported on windows for cuda<12
    # https://github.com/Dao-AILab/flash-attention/issues/345
    if platform.system() != "Linux" and cuda_version < 1200:
        return []
    # Figure out default archs to target
    DEFAULT_ARCHS_LIST = ""
    if cuda_version >= 1108:
        DEFAULT_ARCHS_LIST = "8.0;8.6;9.0"
    elif cuda_version > 1100:
        DEFAULT_ARCHS_LIST = "8.0;8.6"
    elif cuda_version == 1100:
        DEFAULT_ARCHS_LIST = "8.0"
    else:
        return []

    if os.getenv("XFORMERS_DISABLE_FLASH_ATTN", "0") != "0":
        return []

    # Supports `9.0`, `9.0+PTX`, `9.0a+PTX` etc...
    PARSE_CUDA_ARCH_RE = re.compile(
        r"(?P<major>[0-9]+)\.(?P<minor>[0-9])(?P<suffix>[a-zA-Z]{0,1})(?P<ptx>\+PTX){0,1}"
    )
    archs_list = os.environ.get("TORCH_CUDA_ARCH_LIST", DEFAULT_ARCHS_LIST)
    nvcc_archs_flags = []
    for arch in archs_list.replace(" ", ";").split(";"):
        match = PARSE_CUDA_ARCH_RE.match(arch)
        assert match is not None, f"Invalid sm version: {arch}"
        num = 10 * int(match.group("major")) + int(match.group("minor"))
        # Need at least Sm80
        if num < 80:
            continue
        # Sm90 requires nvcc 11.8+
        if num >= 90 and cuda_version < 1108:
            continue
        suffix = match.group("suffix")
        nvcc_archs_flags.append(
            f"-gencode=arch=compute_{num}{suffix},code=sm_{num}{suffix}"
        )
        if match.group("ptx") is not None:
            nvcc_archs_flags.append(
                f"-gencode=arch=compute_{num}{suffix},code=compute_{num}{suffix}"
            )

    return nvcc_archs_flags


def get_flash_attention_extensions(cuda_version: int, extra_compile_args):
    nvcc_archs_flags = get_flash_attention_nvcc_archs_flags(cuda_version)

    if not nvcc_archs_flags:
        return []

    flash_root = os.path.join(this_dir, "third_party", "flash-attention")
    cutlass_inc = os.path.join(flash_root, "csrc", "cutlass", "include")
    if not os.path.exists(flash_root) or not os.path.exists(cutlass_inc):
        raise RuntimeError(
            "flashattention submodule not found. Did you forget "
            "to run `git submodule update --init --recursive` ?"
        )

    sources = ["csrc/flash_attn/flash_api.cpp"]
    for f in glob.glob(os.path.join(flash_root, "csrc", "flash_attn", "src", "*.cu")):
        sources.append(str(Path(f).relative_to(flash_root)))
    common_extra_compile_args = ["-DFLASHATTENTION_DISABLE_ALIBI"]
    return [
        CUDAExtension(
            name="xformers._C_flashattention",
            sources=[os.path.join(flash_root, path) for path in sources],
            extra_compile_args={
                "cxx": extra_compile_args.get("cxx", []) + common_extra_compile_args,
                "nvcc": extra_compile_args.get("nvcc", [])
                + [
                    "-O3",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "--ptxas-options=-v",
                ]
                + nvcc_archs_flags
                + common_extra_compile_args
                + get_extra_nvcc_flags_for_build_type(cuda_version),
            },
            include_dirs=[
                p.absolute()
                for p in [
                    Path(flash_root) / "csrc" / "flash_attn",
                    Path(flash_root) / "csrc" / "flash_attn" / "src",
                    Path(flash_root) / "csrc" / "cutlass" / "include",
                ]
            ],
        )
    ]


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
        os.path.join(extensions_dir, "attention", "hip_fmha", "**", "*.cpp"),
        recursive=True,
    )
    source_hip_generated = glob.glob(
        os.path.join(extensions_dir, "attention", "hip_fmha", "**", "*.cu"),
        recursive=True,
    )
    # avoid the temporary .cu files generated under xformers/csrc/attention/hip_fmha
    source_cuda = list(set(source_cuda) - set(source_hip_generated))
    sources = list(set(sources) - set(source_hip))

    sputnik_dir = os.path.join(this_dir, "third_party", "sputnik")

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
    flash_version = "0.0.0"
    use_pt_flash = False

    if (
        (torch.cuda.is_available() and ((CUDA_HOME is not None)))
        or os.getenv("FORCE_CUDA", "0") == "1"
        or os.getenv("TORCH_CUDA_ARCH_LIST", "") != ""
    ):
        cuda_version = get_cuda_version(CUDA_HOME)
        extension = CUDAExtension
        sources += source_cuda
        include_dirs += [
            sputnik_dir,
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

        flash_extensions = []
        xformers_pt_flash_attn = os.getenv("XFORMERS_PT_FLASH_ATTN")

        # check if the current device supports flash_attention
        nvcc_archs_flags = get_flash_attention_nvcc_archs_flags(cuda_version)
        if not nvcc_archs_flags:
            if xformers_pt_flash_attn == "1":
                raise ValueError(
                    "Current Torch Flash-Attention is not available on this device"
                )
        else:
            # By default, we try to link to torch internal flash attention implementation
            # and silently switch to local flash attention build if no compatibility
            # If we force 'torch FA switch' then setup will fail when no compatibility
            if (
                xformers_pt_flash_attn is None or xformers_pt_flash_attn == "1"
            ) and attn_compat_module.is_pt_flash_compatible(
                force=xformers_pt_flash_attn == "1"
            ):
                flash_version = torch.nn.attention._get_flash_version() + "-pt"
                use_pt_flash = True
            else:
                flash_extensions = get_flash_attention_extensions(
                    cuda_version=cuda_version, extra_compile_args=extra_compile_args
                )
                if flash_extensions:
                    flash_version = get_flash_version()

        ext_modules += flash_extensions

        # NOTE: This should not be applied to Flash-Attention
        # see https://github.com/Dao-AILab/flash-attention/issues/359
        extra_compile_args["nvcc"] += [
            # Workaround for a regression with nvcc > 11.6
            # See https://github.com/facebookresearch/xformers/issues/712
            "--ptxas-options=-O2",
            "--ptxas-options=-allow-expensive-optimizations=true",
        ]
    elif torch.cuda.is_available() and torch.version.hip:
        rename_cpp_cu(source_hip)
        rocm_home = os.getenv("ROCM_PATH")
        hip_version = get_hip_version(rocm_home)

        source_hip_cu = []
        for ff in source_hip:
            source_hip_cu += [ff.replace(".cpp", ".cu")]

        extension = CUDAExtension
        sources += source_hip_cu
        include_dirs += [
            Path(this_dir) / "xformers" / "csrc" / "attention" / "hip_fmha"
        ]

        include_dirs += [
            Path(this_dir) / "third_party" / "composable_kernel_tiled" / "include"
        ]

        generator_flag = []

        cc_flag = ["-DBUILD_PYTHON_PACKAGE"]
        extra_compile_args = {
            "cxx": ["-O3", "-std=c++17"] + generator_flag,
            "nvcc": [
                "-O3",
                "-std=c++17",
                f"--offload-arch={os.getenv('HIP_ARCHITECTURES', 'native')}",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-DCK_FMHA_FWD_FAST_EXP2=1",
                "-fgpu-flush-denormals-to-zero",
                "-Werror",
                "-Woverloaded-virtual",
            ]
            + generator_flag
            + cc_flag,
        }

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
            "flash": flash_version,
            "use_torch_flash": use_pt_flash,
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


class BuildExtensionWithExtraFiles(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        self.xformers_build_metadata = kwargs.pop("extra_files")
        self.pkg_name = "xformers"
        super().__init__(*args, **kwargs)

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


if __name__ == "__main__":
    if os.getenv("BUILD_VERSION"):  # In CI
        version = os.getenv("BUILD_VERSION", "0.0.0")
    else:
        version_txt = os.path.join(this_dir, "version.txt")
        with open(version_txt) as f:
            version = f.readline().strip()
        version += get_local_version_suffix()

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
