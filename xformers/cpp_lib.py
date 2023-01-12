# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

logger = logging.getLogger("xformers")

UNAVAILABLE_FEATURES_MSG = (
    "    Memory-efficient attention, SwiGLU, blocksparse and more won't be available."
)


class xFormersWasNotBuiltException(Exception):
    def __str__(self) -> str:
        return (
            "Need to compile C++ extensions to use all xFormers features.\n"
            "    Please install xformers properly "
            "(see https://github.com/facebookresearch/xformers#installing-xformers)\n"
            + UNAVAILABLE_FEATURES_MSG
        )


class xFormersInvalidLibException(Exception):
    def __str__(self) -> str:
        return (
            "xFormers was built for a different version of PyTorch or Python.\n"
            "    Please reinstall xformers "
            "(see https://github.com/facebookresearch/xformers#installing-xformers)\n"
            + UNAVAILABLE_FEATURES_MSG
        )


def _register_extensions():
    import importlib
    import os

    import torch

    # load the custom_op_library and register the custom ops
    lib_dir = os.path.dirname(__file__)
    if os.name == "nt":
        # Register the main torchvision library location on the default DLL path
        import ctypes
        import sys

        kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
        with_load_library_flags = hasattr(kernel32, "AddDllDirectory")
        prev_error_mode = kernel32.SetErrorMode(0x0001)

        if with_load_library_flags:
            kernel32.AddDllDirectory.restype = ctypes.c_void_p

        if sys.version_info >= (3, 8):
            os.add_dll_directory(lib_dir)
        elif with_load_library_flags:
            res = kernel32.AddDllDirectory(lib_dir)
            if res is None:
                err = ctypes.WinError(ctypes.get_last_error())
                err.strerror += f' Error adding "{lib_dir}" to the DLL directories.'
                raise err

        kernel32.SetErrorMode(prev_error_mode)

    loader_details = (
        importlib.machinery.ExtensionFileLoader,
        importlib.machinery.EXTENSION_SUFFIXES,
    )

    extfinder = importlib.machinery.FileFinder(lib_dir, loader_details)
    ext_specs = extfinder.find_spec("_C")
    if ext_specs is None:
        raise xFormersWasNotBuiltException()
    try:
        torch.ops.load_library(ext_specs.origin)
    except OSError as exc:
        raise xFormersInvalidLibException() from exc


_cpp_library_load_exception = None

try:
    _register_extensions()
except (xFormersInvalidLibException, xFormersWasNotBuiltException) as e:
    ENV_VAR_FOR_DETAILS = "XFORMERS_MORE_DETAILS"
    if os.environ.get(ENV_VAR_FOR_DETAILS, False):
        logger.warning(f"WARNING[XFORMERS]: {e}", exc_info=e)
    else:
        logger.warning(
            f"WARNING[XFORMERS]: {e}\n    Set {ENV_VAR_FOR_DETAILS}=1 for more details"
        )
    _cpp_library_load_exception = e
