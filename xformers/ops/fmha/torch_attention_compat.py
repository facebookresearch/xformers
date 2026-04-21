# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from mslk.attention.fmha.torch_attention_compat import (  # noqa: E402, F401
    is_pt_cutlass_compatible,
    is_pt_flash_old,
)
