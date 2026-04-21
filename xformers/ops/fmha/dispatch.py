# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from mslk.attention.fmha.dispatch import (  # noqa: E402, F401
    _dispatch_fw_priority_list,
    _set_use_fa3,
    fa3_available,
)
