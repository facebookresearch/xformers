# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import torch

from .fmha import (
    AttentionBias,
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
from .indexing import index_select_cat, scaled_index_add
from .modpar_layers import ColumnParallelLinear, RowParallelLinear
from .rmsnorm import RMSNorm
from .rope_padded import rope_padded
from .seqpar import sequence_parallel_leading_matmul, sequence_parallel_trailing_matmul
from .sequence_parallel_fused_ops import (
    fused_allgather_and_anything,
    fused_allgather_and_linear,
    fused_anything_and_reducescatter,
    fused_linear_and_reducescatter,
)
from .sp24 import Sparse24Tensor, sparsify24, sparsify24_like
from .swiglu_op import (
    SwiGLU,
    swiglu,
    SwiGLUEagerOp,
    SwiGLUFusedOp,
    SwiGLUOp,
    SwiGLUOpDispatch,
    SwiGLUPackedFusedOp,
)
from .tiled_matmul import tiled_matmul
from .unbind import get_stack_strides, stack_or_none, unbind

# BW compatibility
AttentionMask = AttentionBias


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
    "SwiGLUFusedOp",
    "SwiGLUOp",
    "SwiGLUOpDispatch",
    "SwiGLUPackedFusedOp",
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
