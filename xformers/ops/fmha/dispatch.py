# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Type

from . import cutlass, flash, small_k, triton
from .common import AttentionBwOpBase, AttentionFwOpBase, Inputs


def _is_cutlass_fwd_faster_than_flash(inp: Inputs) -> bool:
    # Very small batch sizes - if batch size specified
    batch_size, q_len, num_heads, k = inp.query.shape
    if batch_size > 0:
        threads_flash = batch_size * num_heads
        threads_cutlass = threads_flash * (q_len // 64)
        if threads_flash < 60 and (threads_cutlass // 2) >= threads_flash:
            return True
    # Large values of K
    return max(k, inp.key.shape[-1]) == 128


def _is_triton_fwd_fastest(inp: Inputs) -> bool:
    # TODO: fill out
    return False


def _dispatch_fw(inp: Inputs) -> Type[AttentionFwOpBase]:
    """Computes the best operator for forward

    Raises:
        NotImplementedError: if not operator was found

    Returns:
        AttentionOp: The best operator for the configuration
    """

    priority_list_ops: List[Type[AttentionFwOpBase]] = [
        flash.FwOp,
        triton.FwOp,
        cutlass.FwOp,
        small_k.FwOp,
    ]
    if _is_cutlass_fwd_faster_than_flash(inp):
        priority_list_ops.insert(0, cutlass.FwOp)
    if _is_triton_fwd_fastest(inp):
        priority_list_ops.insert(0, triton.FwOp)
    for op in priority_list_ops:
        if op.supports(inp):
            return op
    raise NotImplementedError(f"No operator found for this attention: {inp}")


def _dispatch_bw(inp: Inputs) -> Type[AttentionBwOpBase]:
    priority_list_ops: List[Type[AttentionBwOpBase]] = [
        flash.BwOp,
        cutlass.BwOp,
        # CUDA illegal memory issues, race conditions etc..
        # triton.BwOp,
        # Deprecated
        small_k.BwOp,
    ]
    for op in priority_list_ops:
        if op.supports(inp):
            return op
    raise NotImplementedError(f"No operator found for this attention: {inp}")
