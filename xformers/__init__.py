# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch

from . import _cpp_lib

try:
    from .version import __version__  # noqa: F401
except ImportError:
    __version__ = "0.0.0"


logger = logging.getLogger("xformers")

_has_cpp_library: bool = _cpp_lib._cpp_library_load_exception is None

# Set to true to utilize functorch
_is_functorch_available: bool = False
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
    try:
        from xformers.triton.softmax import softmax as triton_softmax  # noqa

        return True
    except (ImportError, AttributeError) as e:
        logger.warning(
            f"A matching Triton is not available, some optimizations will not be enabled.\nError caught was: {e}"
        )
        return False


if _is_functorch_available:
    try:
        from xformers.components.nvfuser import NVFusedBiasActivationDropout  # noqa
    except ImportError as e:
        logger.warning(
            f"Functorch is not available, some optimizations will not be enabled.\nError caught was: {e}"
        )
        _is_functorch_available = False
