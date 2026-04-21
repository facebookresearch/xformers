# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch
from mslk.attention.fmha._triton.available import is_triton_available
from mslk.attention.fmha.triton_splitk import (  # noqa: E402, F401
    _is_cuda,
    _is_cuda_at_least_sm80,
    _is_supported_causal_bias,
    _is_supported_gappy_bias,
    _is_supported_local_bias,
    _is_supported_paged_bias,
    _merge_attentions_backward,
    _prepare_reduce_kernel_params,
    _strides,
    FwOp,
    FwOp_Map,
    FwOp_S1,
    FwOp_S128,
    FwOp_S16,
    FwOp_S2,
    FwOp_S32,
    FwOp_S4,
    FwOp_S64,
    FwOp_S8,
    InputsFp8,
    merge_attentions,
    merge_attentions_varargs,
    merge_attentions_varargs_backward,
    merge_attentions_varargs_backward_fake,
    merge_attentions_varargs_fake,
)

if is_triton_available():
    # just to cause the import
    from ._triton.splitk_kernels import (  # noqa: E402, F401
        AUTOTUNER_KEY as _AUTOTUNER_KEY,
    )

    torch.library.custom_op(
        "xformers::fmha_merge_attentions_varargs",
        merge_attentions_varargs._init_fn,
        mutates_args=(),
        device_types=["cuda"],
    )

    torch.library.register_fake(
        "xformers::fmha_merge_attentions_varargs", merge_attentions_varargs_fake
    )

    merge_attentions_varargs.register_autograd(_merge_attentions_backward)

    torch.library.custom_op(
        "xformers::merge_attentions_varargs_backward",
        merge_attentions_varargs_backward._init_fn,
        mutates_args=(),
        device_types=["cuda"],
    )

    torch.library.register_fake(
        "xformers::merge_attentions_varargs_backward",
        merge_attentions_varargs_backward_fake,
    )
