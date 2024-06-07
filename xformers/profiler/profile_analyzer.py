# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Sequence, Set, cast

import torch


class FakeKinetoEvent:
    def __init__(self, e: torch._C._autograd._KinetoEvent) -> None:
        for attr in dir(e):
            if attr.startswith("_"):
                continue
            setattr(self, attr, getattr(e, attr))
        self._kineto_event = e


def _attention_flops(queries, values, causal: bool) -> int:
    assert isinstance(causal, bool)
    *B, N, K = queries
    *B, Nv, Kv = values
    if causal:  # NOTE: Causal from bottom right
        # non-causal part
        flops = 2 * N * max(Nv - N, 0) * K + 2 * max(Nv - N, 0) * max(Nv - N, 0) * Kv
        # causal part
        flops += (
            2 * min(N, Nv) * min(N, Nv) * K + 2 * min(N, Nv) * min(N, Nv) * Kv
        ) // 2
    else:
        flops = 2 * N * Nv * K + 2 * N * Nv * Kv
    for b in B:
        flops *= b
    return int(flops)


def _get_arg_idx(op, *arg_names: str) -> int:
    for i, arg in enumerate(op.default._schema.arguments):
        if arg.name in arg_names:
            return i
    raise ValueError(f"No such argument {arg_names} found in {op.default._schema}")


def _replace_if_needed(
    e: torch._C._autograd._KinetoEvent,
) -> torch._C._autograd._KinetoEvent:
    """
    Adds a flops amount for operators that don't have this information in Kineto already
    This mostly applies for the attention for now, as GEMMs are already calculated by Kineto
    and other operations are negligible.
    """
    if e.device_type().name != "CPU":
        return e
    op_name = e.name()
    flops = None

    ATTN_OPS = {
        getattr(lib, op).default.name(): (getattr(lib, op), is_bwd)
        for lib, op, is_bwd in [
            (torch.ops.aten, "scaled_dot_product_attention", False),
            (torch.ops.xformers_flash, "flash_fwd", False),
            (torch.ops.xformers, "efficient_attention_forward_cutlass", False),
            (torch.ops.aten, "_efficient_attention_forward", False),
            (torch.ops.aten, "_scaled_dot_product_flash_attention_backward", True),
            (torch.ops.aten, "_scaled_dot_product_efficient_attention_backward", True),
            (torch.ops.xformers_flash, "flash_bwd", True),
            (torch.ops.xformers, "efficient_attention_backward_cutlass", True),
            (torch.ops.aten, "_efficient_attention_backward", True),
        ]
        if hasattr(lib, op)
    }
    if op_name in ATTN_OPS.keys():
        op, is_bwd = ATTN_OPS[op_name]
        shapes = e.shapes()
        concrete_inputs = e.concrete_inputs()
        try:
            is_causal = concrete_inputs[_get_arg_idx(op, "causal", "is_causal")]
        except ValueError:
            is_causal = concrete_inputs[_get_arg_idx(op, "custom_mask_type")] != 0
        flops = _attention_flops(
            shapes[_get_arg_idx(op, "query")],
            shapes[_get_arg_idx(op, "value")],
            is_causal,
        )
        if is_bwd:
            flops = flops * 5 // 2
    if flops is not None:
        new_e = FakeKinetoEvent(e)
        new_e.flops = lambda: flops  # type: ignore
        e = cast(torch._C._autograd._KinetoEvent, new_e)
    return e


# BW compat with older PT versions
if "start_ns" in dir(torch._C._autograd._KinetoEvent):

    def _start_ns(e) -> int:
        return e.start_ns()

    def _duration_ns(e) -> int:
        return e.duration_ns()

else:

    def _start_ns(e) -> int:
        return e.start_us() * 1000

    def _duration_ns(e) -> int:
        return e.duration_us() * 1000


@dataclass
class AnalyzedTrace:
    operations_per_dtype_fw: Dict[torch.dtype, float]
    operations_per_dtype_bw: Dict[torch.dtype, float]
    total_time_s: float

    def compute_num_ops(
        self, dtype: torch.dtype, fw: bool = True, bw: bool = True
    ) -> float:
        ops = 0.0
        if fw:
            ops += self.operations_per_dtype_fw.get(dtype, 0.0)
        if bw:
            ops += self.operations_per_dtype_bw.get(dtype, 0.0)
        return ops

    def compute_hfu(self, hardware_flops: Dict[torch.dtype, float]) -> float:
        hfu_seconds = 0.0
        for dtype, hw_flops in hardware_flops.items():
            hfu_seconds += self.compute_num_ops(dtype) / hw_flops
        return hfu_seconds / self.total_time_s

    def compute_mfu(self, hardware_flops: Dict[torch.dtype, float]) -> float:
        # Estimated by considering the bw flops should be exactly 2x the fw flops
        # The reason MFU!=HFU is because of recomputation in the BW pass
        hfu_seconds = 0.0
        for dtype, hw_flops in hardware_flops.items():
            hfu_seconds += (
                min(
                    3 * self.compute_num_ops(dtype, bw=False),
                    self.compute_num_ops(dtype),
                )
                / hw_flops
            )
        return hfu_seconds / self.total_time_s

    @staticmethod
    def from_profile(
        events: Sequence[torch._C._autograd._KinetoEvent],
    ) -> "AnalyzedTrace":
        events = [_replace_if_needed(e) for e in events]
        # All dispatcher ops
        all_ops = [
            e
            for e in events
            if (
                e.device_type().name == "CPU"
                and (e.dtypes() or e.shapes() or e.flops() > 0)
            )
        ]

        root_ops: Set[torch._C._autograd._KinetoEvent] = set()

        def _find_parent_op(
            e: torch._C._autograd._KinetoEvent,
        ) -> torch._C._autograd._KinetoEvent:
            e_range = [_start_ns(e), _start_ns(e) + _duration_ns(e)]
            candidate = e
            for parent in all_ops:
                if parent.device_type() != e.device_type():
                    continue
                if parent.start_thread_id() != e.start_thread_id():
                    continue
                p_range = [_start_ns(parent), _start_ns(parent) + _duration_ns(parent)]
                if not (p_range[0] < e_range[0] < e_range[1] < p_range[1]):
                    continue
                # We take the longest parent with flops
                if parent.flops() > 0 and _duration_ns(candidate) < _duration_ns(
                    parent
                ):
                    candidate = parent
            return candidate

        for op in all_ops:
            if op.flops() == 0:
                continue
            root_ops.add(_find_parent_op(op))

        operations_per_dtype_fw: Dict[torch.dtype, float] = defaultdict(float)
        operations_per_dtype_bw: Dict[torch.dtype, float] = defaultdict(float)
        # We detect BW pass ops based on their thread id
        all_bw_threads = {e.start_thread_id() for e in events if e.fwd_thread_id() > 0}
        # Find total dt
        ATEN_DTYPES = [
            # NOTE: A single torch.dtype per number of bits
            # (eg so we map bf16 --> b16)
            ("double", torch.float64),
            ("float", torch.float),
            ("c10::Half", torch.float16),
            ("c10::BFloat16", torch.float16),
            ("c10::Int8", torch.int8),
        ]
        begin_ns, end_ns = math.inf, 0
        for op in root_ops:
            dtype = None
            for aten_dtype, torch_dtype in ATEN_DTYPES:
                if aten_dtype in op.dtypes():
                    dtype = torch_dtype
                    break
            if dtype is None:  # ???
                continue
            if op.start_thread_id() in all_bw_threads:
                operations_per_dtype_bw[dtype] += op.flops()
            else:
                operations_per_dtype_fw[dtype] += op.flops()
        for op in events:
            if op.device_type().name != "CUDA":
                continue
            begin_ns = min(begin_ns, _start_ns(op))
            end_ns = max(end_ns, _start_ns(op) + _duration_ns(op))

        return AnalyzedTrace(
            operations_per_dtype_fw=operations_per_dtype_fw,
            operations_per_dtype_bw=operations_per_dtype_bw,
            total_time_s=(end_ns - begin_ns) / (10**9),
        )
