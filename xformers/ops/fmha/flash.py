# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import mslk.attention.fmha.flash as _flash
import torch
from mslk.attention.fmha.flash import BwOp, FwOp  # noqa: E402, F401

if hasattr(_flash, "_flash_bwd"):
    torch.library.custom_op(
        "xformers_flash::flash_fwd",
        _flash._flash_fwd._init_fn,
        mutates_args=(),
        device_types=["cuda"],
    )
    torch.library.register_fake("xformers_flash::flash_fwd", _flash._flash_fwd_abstract)
    torch.library.custom_op(
        "xformers_flash::flash_bwd",
        _flash._flash_bwd._init_fn,
        mutates_args=(),
        device_types=["cuda"],
    )
    torch.library.register_fake("xformers_flash::flash_bwd", _flash._flash_bwd_abstract)
