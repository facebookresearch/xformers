# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# The fmha implementation has moved to the mslk package. This package and its
# submodules re-export mslk symbols to preserve the xformers.ops.fmha API.
import torch

try:
    # flake8: noqa
    from mslk.attention.fmha import (
        _deserialize_bias,
        _detect_lse_packed_or_raise,
        _fMHA,
        _memory_efficient_attention,
        _memory_efficient_attention_backward,
        _memory_efficient_attention_forward,
        _memory_efficient_attention_forward_requires_grad,
        _memory_efficient_attention_forward_torch_wrapper,
        _memory_efficient_attention_forward_torch_wrapper_meta,
        _memory_efficient_attention_forward_torch_wrapper_with_bias,
        _memory_efficient_attention_forward_torch_wrapper_with_bias_meta,
        _OPS_LOOKUP,
        _serialize_op,
        _unserialize_op,
        ALL_BW_OPS,
        ALL_FW_OPS,
        AttentionBias,
        AttentionBwOpBase,
        AttentionFwOpBase,
        AttentionOp,
        AttentionOpBase,
        BlockDiagonalMask,
        dispatch,
        Inputs,
        LowerTriangularMask,
        memory_efficient_attention,
        memory_efficient_attention_backward,
        memory_efficient_attention_forward,
        memory_efficient_attention_forward_requires_grad,
        memory_efficient_attention_partial,
        MemoryEfficientAttentionCkOp,
        MemoryEfficientAttentionCutlassBlackwellOp,
        MemoryEfficientAttentionCutlassFwdFlashBwOp,
        MemoryEfficientAttentionCutlassOp,
        MemoryEfficientAttentionFlashAttentionOp,
        MemoryEfficientAttentionSplitKCkOp,
        merge_attentions,
    )

    from mslk.attention.fmha.dispatch import (
        _dispatch_bw,
        _dispatch_fw,
        _ensure_op_supports_or_raise,
        _get_use_fa3,
        _set_use_fa3,
    )

    from . import (
        attn_bias,
        ck,
        ck_splitk,
        common,
        cutlass,
        cutlass_blackwell,
        flash,
        flash3,
        triton_splitk,
    )

    torch.library.define(
        "xformer::memory_efficient_attention_forward",
        "(Tensor q, Tensor k, Tensor v, Tensor? b = None, float? p = 0.0, float? scale = None) -> Tensor",
    )

    torch.library.impl(
        "xformer::memory_efficient_attention_forward",
        "Meta",
        _memory_efficient_attention_forward_torch_wrapper_meta,
    )
    torch.library.impl(
        "xformer::memory_efficient_attention_forward",
        "CUDA",
        _memory_efficient_attention_forward_torch_wrapper,
    )

    torch.library.define(
        "xformer::memory_efficient_attention_forward_with_bias",
        "(Tensor q, Tensor k, Tensor v, Tensor b, float? p = 0.0, float? scale = None) -> Tensor",
    )

    torch.library.impl(
        "xformer::memory_efficient_attention_forward_with_bias",
        "Meta",
        _memory_efficient_attention_forward_torch_wrapper_with_bias_meta,
    )

    torch.library.impl(
        "xformer::memory_efficient_attention_forward_with_bias",
        "CUDA",
        _memory_efficient_attention_forward_torch_wrapper_with_bias,
    )

except ModuleNotFoundError:
    pass
