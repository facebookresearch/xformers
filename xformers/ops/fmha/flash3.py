# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import mslk.attention.fmha.flash3 as _flash3
import torch
from mslk.attention.fmha.flash3 import (  # noqa: E402, F401
    _C_flashattention3,
    BwOp,
    FwOp,
    FwOp_KVSplit,
    mask_non_zeros,
)
from torch.utils.flop_counter import register_flop_formula

if hasattr(_flash3, "mha_bwd"):
    torch.library.custom_op(
        "xformers_flash3::flash_fwd",
        _flash3.mha_fwd._init_fn,
        mutates_args=(),
        device_types=["cuda"],
    )
    torch.library.register_fake("xformers_flash3::flash_fwd", _flash3.mha_fwd_fake)
    register_flop_formula(torch.ops.xformers_flash3.flash_fwd, get_raw=True)(
        _flash3.mha_fwd_flops
    )
    torch.library.custom_op(
        "xformers_flash3::flash_bwd",
        _flash3.mha_bwd._init_fn,
        mutates_args=(),
        device_types=["cuda"],
    )
    torch.library.register_fake("xformers_flash3::flash_bwd", _flash3.mha_bwd_fake)
    register_flop_formula(torch.ops.xformers_flash3.flash_bwd, get_raw=True)(
        _flash3.mha_bwd_flops
    )
