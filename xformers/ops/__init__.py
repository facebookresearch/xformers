# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch

from .memory_efficient_attention import (  # noqa: F401
    AttentionMask,
    AttentionOpBase,
    AttentionOpDispatch,
    LowerTriangularMask,
    MemoryEfficientAttentionCutlassFwdFlashBwOp,
    MemoryEfficientAttentionCutlassOp,
    MemoryEfficientAttentionFlashAttentionOp,
    MemoryEfficientAttentionOp,
    memory_efficient_attention,
)
from .swiglu import (  # noqa: F401
    SwiGLUFusedOp,
    SwiGLUOp,
    SwiGLUOpDispatch,
    SwiGLUPackedFusedOp,
    _info,
    functional_swiglu,
)
from .unbind import get_stack_strides, stack_or_none, unbind  # noqa: F401


def masked_matmul(a, b, mask=None):
    if torch.overrides.has_torch_function((a, b, mask)):
        return torch.overrides.handle_torch_function(
            masked_matmul, (a, b, mask), a, b, mask
        )

    att = a @ b

    if mask is None:
        return att

    if mask.dtype == torch.bool:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).expand(att.shape[0], -1, -1)
        # mask is presumed false == ignore
        att[~mask] = float("-inf")
    else:
        # mask is presumed additive
        att += mask
    return att
