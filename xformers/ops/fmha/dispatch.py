# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import textwrap
from collections import deque
from typing import List, Sequence, Type, TypeVar

import torch

from . import (
    attn_bias,
    ck,
    ck_decoder,
    ck_splitk,
    cutlass,
    decoder,
    flash,
    small_k,
    triton_splitk,
)
from .common import AttentionBwOpBase, AttentionFwOpBase, Inputs

T = TypeVar("T", Type[AttentionFwOpBase], Type[AttentionBwOpBase])


def _format_inputs_description(inp: Inputs) -> str:
    return f"""query       : shape={tuple(inp.query.shape)} ({inp.query.dtype})
key         : shape={tuple(inp.key.shape)} ({inp.key.dtype})
value       : shape={tuple(inp.value.shape)} ({inp.value.dtype})
attn_bias   : {type(inp.attn_bias)}
p           : {inp.p}"""


def _ensure_op_supports_or_raise(exc_type, name: str, op, inp: Inputs) -> None:
    reasons = op.not_supported_reasons(inp)
    if not reasons:
        return
    raise exc_type(
        f"""Operator `{name}` does not support inputs:
{textwrap.indent(_format_inputs_description(inp), '     ')}
{_format_not_supported_reasons(op, reasons)}"""
    )


def _format_not_supported_reasons(op, reasons: List[str]) -> str:
    return f"`{op.NAME}` is not supported because:\n    " + "\n    ".join(reasons)


def _run_priority_list(name: str, priority_list: Sequence[T], inp: Inputs) -> T:
    not_supported_reasons: List[List[str]] = []
    for op in priority_list:
        not_supported = op.not_supported_reasons(inp)
        if not not_supported:
            return op
        not_supported_reasons.append(not_supported)

    # Let's write a nice message explaining what we tried and why it's not supported
    msg = f"""No operator found for `{name}` with inputs:
{textwrap.indent(_format_inputs_description(inp), '     ')}"""
    for op, not_supported in zip(priority_list, not_supported_reasons):
        msg += "\n" + _format_not_supported_reasons(op, not_supported)
    raise NotImplementedError(msg)


def _dispatch_fw_priority_list(
    inp: Inputs, needs_gradient: bool
) -> Sequence[Type[AttentionFwOpBase]]:
    if torch.version.cuda:
        priority_list_ops = deque(
            [
                flash.FwOp,
                cutlass.FwOp,
                small_k.FwOp,
            ]
        )
    else:
        priority_list_ops = deque(
            [
                ck.FwOp,
            ]
        )
    if not needs_gradient:
        mqa_or_gqa = (
            inp.key.ndim > 3 and inp.key.stride(-2) == 0 and inp.key.shape[-2] > 1
        )
        if not mqa_or_gqa:
            # With multiquery, cutlass is sometimes faster than decoder
            # but it's not currently clear when.
            priority_list_ops.appendleft(
                decoder.FwOp if torch.version.cuda else ck_decoder.FwOp
            )
        # Split-KV is useful with MQA
        # for short Q-seqlen / long K-seqlen
        if mqa_or_gqa and inp.query.shape[1] <= 32 and inp.key.shape[1] >= 256:
            parallelism_BH = 0  # BMK
            if inp.query.ndim == 3:
                parallelism_BH = inp.query.shape[0]
            elif inp.query.ndim == 4:  # BMHK
                parallelism_BH = inp.query.shape[0] * inp.query.shape[2]
            elif inp.query.ndim == 5:  # BMGHK
                parallelism_BH = inp.query.shape[0] * inp.query.shape[2]
            if parallelism_BH > 0 and parallelism_BH < 64:
                priority_list_ops.appendleft(ck_splitk.FwOp)
                priority_list_ops.appendleft(triton_splitk.FwOp)
                # Without variable seqlen flash is fastest
                if not isinstance(inp.attn_bias, attn_bias.BlockDiagonalMask):
                    priority_list_ops.remove(flash.FwOp)
                    priority_list_ops.appendleft(flash.FwOp)

    return priority_list_ops


def _dispatch_fw(inp: Inputs, needs_gradient: bool) -> Type[AttentionFwOpBase]:
    """Computes the best operator for forward

    Raises:
        NotImplementedError: if not operator was found

    Returns:
        AttentionOp: The best operator for the configuration
    """
    return _run_priority_list(
        "memory_efficient_attention_forward",
        _dispatch_fw_priority_list(inp, needs_gradient),
        inp,
    )


def _is_cutlassB_faster_than_flash(inp: Inputs) -> bool:
    return False


def _dispatch_bw(inp: Inputs) -> Type[AttentionBwOpBase]:
    priority_list_ops: List[Type[AttentionBwOpBase]] = [
        flash.BwOp,
        cutlass.BwOp,
        # CUDA illegal memory issues, race conditions etc..
        # triton.BwOp,
        # Deprecated
        small_k.BwOp,
    ]
    if _is_cutlassB_faster_than_flash(inp):
        priority_list_ops.remove(cutlass.BwOp)
        priority_list_ops.insert(0, cutlass.BwOp)
    return _run_priority_list(
        "memory_efficient_attention_backward", priority_list_ops, inp
    )
