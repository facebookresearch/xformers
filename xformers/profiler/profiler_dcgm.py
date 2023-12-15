# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import sys

from .profiler import _Profiler, logger

DCGM_PROFILER_AVAILABLE = False
try:
    DCGM_PYTHON_PATH: str = "/usr/local/dcgm/bindings/python3"
    sys.path.insert(0, DCGM_PYTHON_PATH)
    from .profiler_dcgm_impl import DCGMProfiler

    DCGM_PROFILER_AVAILABLE = True
except ModuleNotFoundError:
    logger.warning(
        f"Unable to find python bindings at {DCGM_PYTHON_PATH}. "
        "No data will be captured."
    )

    class DCGMProfiler:  # type: ignore
        """The dummy DCGM Profiler."""

        def __init__(
            self,
            main_profiler: "_Profiler",
            gpus_to_profile=None,
            field_ids_to_profile=None,
            updateFreq=None,
        ) -> None:
            pass

        def __enter__(self) -> None:
            pass

        def __exit__(self, exc_type, exc_val, exc_tb) -> None:
            pass

        def step(self) -> None:
            pass


del sys.path[0]
