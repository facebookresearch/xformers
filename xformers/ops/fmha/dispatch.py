# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import textwrap
from collections import deque
from typing import Any, List, Optional, Sequence, Tuple, Type, TypeVar

import torch

from . import attn_bias, ck, cutlass, flash, flash3, triton_splitk
from .common import AttentionBwOpBase, AttentionFwOpBase, Inputs

T = TypeVar("T", Type[AttentionFwOpBase], Type[AttentionBwOpBase])


_USE_FLASH_ATTENTION_3 = True


def _set_use_fa3(use_flash_attention3: bool) -> None:
    global _USE_FLASH_ATTENTION_3
    _USE_FLASH_ATTENTION_3 = use_flash_attention3


def _get_use_fa3() -> bool:
    global _USE_FLASH_ATTENTION_3
    return _USE_FLASH_ATTENTION_3


def fa3_available() -> bool:
    has_cuda = torch.version.cuda is not None
    is_90a = has_cuda and torch.cuda.get_device_capability() >= (9, 0)
    has_valid_flash3 = flash3._C_flashattention3 is not None  # pyre-ignore[16]
    return is_90a and has_valid_flash3


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


def _run_priority_list(
    name: str,
    priority_list: Sequence[T],
    inp: Inputs,
    extra_op_reasons: Optional[List[Tuple[Any, List[str]]]] = None,
) -> T:
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
    if extra_op_reasons is not None:
        for op, not_supported in extra_op_reasons:
            msg += "\n" + _format_not_supported_reasons(op, not_supported)
    raise NotImplementedError(msg)


def _dispatch_fw_priority_list(
    inp: Inputs, needs_gradient: bool
) -> Sequence[Type[AttentionFwOpBase]]:
    if torch.version.cuda:
        flash3_op = [flash3.FwOp] if _get_use_fa3() else []
        priority_list_ops = deque(
            flash3_op
            + [
                flash.FwOp,
                cutlass.FwOp,
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
                # priority_list_ops.appendleft(ck_splitk.FwOp)
                priority_list_ops.appendleft(triton_splitk.FwOp)
                # Without variable seqlen flash is fastest
                if torch.version.cuda and not isinstance(
                    inp.attn_bias, attn_bias.BlockDiagonalMask
                ):
                    if _get_use_fa3():
                        priority_list_ops.remove(flash3.FwOp)
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


def _dispatch_bw(
    inp: Inputs, varlen_lse_packed: Optional[bool]
) -> Type[AttentionBwOpBase]:
    if torch.version.cuda:
        priority_list_ops: List[Type[AttentionBwOpBase]] = [
            flash.BwOp,
            cutlass.BwOp,
        ]
        if _get_use_fa3():
            priority_list_ops = [flash3.BwOp] + priority_list_ops
    else:
        priority_list_ops = [
            ck.BwOp,
        ]

    # NOTE: If we have a variable seqlen `attn_bias`, we need to get a BW pass
    # that supports the LSE format
    # *unless* we are in the case where both formats are the same (bs=1)
    extra_op_reasons = []
    if (
        isinstance(inp.attn_bias, attn_bias.VARLEN_BIASES)
        and inp.attn_bias.q_seqinfo.seqstart.shape[0] > 2
    ):
        assert varlen_lse_packed is not None
        for op in priority_list_ops:
            if op.VARLEN_LSE_PACKED != varlen_lse_packed:
                extra_op_reasons.append(
                    (
                        op,
                        [
                            f"LSE is in {'packed' if varlen_lse_packed else 'padded'} format"
                        ],
                    )
                )
        priority_list_ops = [
            op for op in priority_list_ops if op.VARLEN_LSE_PACKED == varlen_lse_packed
        ]
    return _run_priority_list(
        "memory_efficient_attention_backward", priority_list_ops, inp
    )
