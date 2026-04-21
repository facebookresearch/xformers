# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import sys

if sys.version_info >= (3, 9):
    from mslk.attention.fmha._triton.splitk_kernels import (  # noqa: E402, F401
        AUTOTUNER_KEY,
        get_autotuner_cache,
        set_autotuner_cache,
    )
