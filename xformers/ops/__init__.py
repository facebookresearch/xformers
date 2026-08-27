# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import importlib.util
import logging

import torch
import torch.distributed

if _HAS_MSLK := importlib.util.find_spec("mslk") is not None:
    from .fmha import (
        AttentionBias,
        AttentionBias as AttentionMask,
        AttentionOp,
        AttentionOpBase,
        LowerTriangularMask,
        memory_efficient_attention,
        memory_efficient_attention_backward,
        memory_efficient_attention_forward,
        memory_efficient_attention_forward_requires_grad,
        MemoryEfficientAttentionCkOp,
        MemoryEfficientAttentionCutlassFwdFlashBwOp,
        MemoryEfficientAttentionCutlassOp,
        MemoryEfficientAttentionFlashAttentionOp,
        MemoryEfficientAttentionSplitKCkOp,
    )
else:
    logging.getLogger("xformers").warning(
        "WARNING[XFORMERS]: the 'mslk' package is not installed, so the attention ops are "
        "unavailable.\n  xformers.ops.memory_efficient_attention and the related symbols "
        "will not exist.\n  mslk is a dependency of this part of xFormers."
    )
from .indexing import index_select_cat, scaled_index_add

# PyTorch can be built without distributed support, in which case
# torch.distributed.is_available() returns False and most of the
# torch.distributed namespace does not exist. That happens for instance in the
# NVIDIA PyTorch containers for Jetson iGPU devices. The model-parallel and
# sequence-parallel ops cannot work at all in that situation, so we skip them
# and keep the rest of xFormers usable, the same way we skip the ops that need
# mslk.
if _HAS_TORCH_DISTRIBUTED := torch.distributed.is_available():
    from .modpar_layers import ColumnParallelLinear, RowParallelLinear
from .rmsnorm import RMSNorm

if _HAS_MSLK:
    from .rope_padded import rope_padded

if _HAS_TORCH_DISTRIBUTED:
    from .seqpar import (
        sequence_parallel_leading_matmul,
        sequence_parallel_trailing_matmul,
    )
    from .sequence_parallel_fused_ops import (
        fused_allgather_and_anything,
        fused_allgather_and_linear,
        fused_anything_and_reducescatter,
        fused_linear_and_reducescatter,
    )
from .sp24 import Sparse24Tensor, sparsify24, sparsify24_like
from .swiglu_op import SwiGLU, swiglu, SwiGLUEagerOp, SwiGLUOp, SwiGLUOpDispatch
from .tiled_matmul import tiled_matmul
from .unbind import get_stack_strides, stack_or_none, unbind


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


__all__ = [
    # fmha
    "AttentionBias",
    "AttentionMask",
    "AttentionOp",
    "AttentionOpBase",
    "LowerTriangularMask",
    "MemoryEfficientAttentionCutlassFwdFlashBwOp",
    "MemoryEfficientAttentionCutlassOp",
    "MemoryEfficientAttentionFlashAttentionOp",
    "MemoryEfficientAttentionCkOp",
    "MemoryEfficientAttentionSplitKCkOp",
    "memory_efficient_attention",
    "memory_efficient_attention_backward",
    "memory_efficient_attention_forward",
    "memory_efficient_attention_forward_requires_grad",
    # indexing
    "index_select_cat",
    "scaled_index_add",
    # modpar_layers
    "ColumnParallelLinear",
    "RowParallelLinear",
    # rmsnorm
    "RMSNorm",
    # rope_padded
    "rope_padded",
    # seqpar
    "sequence_parallel_leading_matmul",
    "sequence_parallel_trailing_matmul",
    # sequence_parallel_fused_ops
    "fused_allgather_and_anything",
    "fused_allgather_and_linear",
    "fused_anything_and_reducescatter",
    "fused_linear_and_reducescatter",
    # swiglu_op
    "SwiGLU",
    "SwiGLUEagerOp",
    "SwiGLUOp",
    "SwiGLUOpDispatch",
    "swiglu",
    # tiled_matmul
    "tiled_matmul",
    # unbind
    "get_stack_strides",
    "stack_or_none",
    "unbind",
    # sp24
    "sparsify24",
    "sparsify24_like",
    "Sparse24Tensor",
    # .
    "masked_matmul",
]
