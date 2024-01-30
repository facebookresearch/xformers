# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import torch

from . import _cpp_lib
from .checkpoint import (  # noqa: E402, F401
    checkpoint,
    get_optimal_checkpoint_policy,
    list_operators,
    selective_checkpoint_wrapper,
)

try:
    from .version import __version__  # noqa: F401
except ImportError:
    __version__ = "0.0.0"


logger = logging.getLogger("xformers")

_has_cpp_library: bool = _cpp_lib._cpp_library_load_exception is None

_is_opensource: bool = True


def compute_once(func):
    value = None

    def func_wrapper():
        nonlocal value
        if value is None:
            value = func()
        return value

    return func_wrapper


@compute_once
def _is_triton_available():
    if not torch.cuda.is_available():
        return False
    if os.environ.get("XFORMERS_FORCE_DISABLE_TRITON", "0") == "1":
        return False
    # We have many errors on V100 with recent triton versions
    # Let's just drop support for triton kernels below A100
    if torch.cuda.get_device_capability("cuda") < (8, 0):
        return False
    try:
        from xformers.triton.softmax import softmax as triton_softmax  # noqa

        return True
    except (ImportError, AttributeError):
        logger.warning(
            "A matching Triton is not available, some optimizations will not be enabled",
            exc_info=True,
        )
        return False


@compute_once
def get_python_lib():
    return torch.library.Library("xformers_python", "DEF")


# end of file
