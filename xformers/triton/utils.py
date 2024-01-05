# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger("xformers")


_oldest_gpu: Optional[Tuple[int, int]] = None


def _get_oldest_gpu() -> Tuple[int, int]:
    global _oldest_gpu
    if _oldest_gpu is None:
        _oldest_gpu = min(
            (
                torch.cuda.get_device_capability(f"cuda:{i}")
                for i in range(torch.cuda.device_count())
            ),
            default=(0, 0),
        )
    return _oldest_gpu


def gpu_capabilities_older_than_70() -> bool:
    """Return True if the GPU's compute capability is older than SM70."""
    return _get_oldest_gpu() < (7, 0)


def gpu_capabilities_older_than_80() -> bool:
    """Return True if the GPU's compute capability is older than SM80."""
    return _get_oldest_gpu() < (8, 0)
