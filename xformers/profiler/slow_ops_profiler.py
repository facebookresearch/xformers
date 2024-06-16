# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Set, Tuple

import torch.cuda.memory
import torch.cuda.nvtx
import torch.profiler
import torch.utils.hooks
from torch.utils._python_dispatch import TorchDispatchMode, _pop_mode_temporarily
from torch.utils._pytree import tree_map

from ..ops.common import FUNC_TO_XFORMERS_OPERATOR
from .device_limits import get_device_limits
from .profiler import _Profiler


class TorchFuncMockNoDispatch:
    """
    Wraps a method to call it without the custom
    pytorch dispatcher
    """

    def __init__(self, pt_impl):
        self.pt_impl = pt_impl

    def __get__(self, obj, c):
        return partial(self, obj)

    def __call__(self, obj, *args, **kwargs):
        with _pop_mode_temporarily():
            return self.pt_impl(obj, *args, **kwargs)


class DispatcherWithoutBrokenFuncs(TorchDispatchMode):
    TENSOR_FUNCS_NO_DISPATCH = [
        # Can't convert Stream argument to Python object
        # https://github.com/pytorch/pytorch/issues/94403
        "record_stream"
    ]

    def __enter__(self) -> None:
        self._pt_impls = {}
        for k in self.TENSOR_FUNCS_NO_DISPATCH:
            impl = getattr(torch.Tensor, k)
            self._pt_impls[k] = impl
            setattr(torch.Tensor, k, TorchFuncMockNoDispatch(impl))
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for k in self.TENSOR_FUNCS_NO_DISPATCH:
            setattr(torch.Tensor, k, self._pt_impls[k])
        return super().__exit__(exc_type, exc_val, exc_tb)


def get_shape(i):
    return i.shape


def prod(x):
    res = 1
    for i in x:
        res *= i
    return res


class GemmOpComputeFlops:
    def _get_mnk(self, inputs: List[Any]) -> Tuple[int, int, int]:
        return (prod(inputs[0].shape[:-1]), inputs[1].shape[1], inputs[0].shape[-1])

    def __call__(self, inputs: List[Any], outputs: List[Any]) -> float:
        return 2 * prod(self._get_mnk(inputs))

    def op_suffix(self, inputs: List[Any]) -> str:
        m, n, k = self._get_mnk(inputs)
        return f"_{m}x{n}x{k}"


class GemmOpComputeFlopsLinear(GemmOpComputeFlops):
    def _get_mnk(self, inputs: List[Any]) -> Tuple[int, int, int]:
        return (prod(inputs[0].shape[:-1]), inputs[1].shape[0], inputs[0].shape[-1])


class GemmOpComputeFlopsMv(GemmOpComputeFlops):
    def _get_mnk(self, inputs: List[Any]) -> Tuple[int, int, int]:
        return (prod(inputs[0].shape[:-1]), 1, inputs[0].shape[-1])


class GemmOpComputeFlopsBmm(GemmOpComputeFlops):
    def _get_mnk(self, inputs: List[Any]) -> Tuple[int, int, int]:
        a, b = inputs[0], inputs[1]
        assert a.ndim == 3
        assert b.ndim == 3
        bs = max(inputs[0].shape[0], inputs[1].shape[0])
        return (bs * a.shape[1], b.shape[-1], b.shape[-2])


class GemmOpComputeFlopsAddmm(GemmOpComputeFlops):
    def _get_mnk(self, inputs: List[Any]) -> Tuple[int, int, int]:
        return super()._get_mnk(inputs[1:])


class GemmOpComputeFlopsAddbmm(GemmOpComputeFlopsBmm):
    def _get_mnk(self, inputs: List[Any]) -> Tuple[int, int, int]:
        return super()._get_mnk(inputs[1:])


def conv_flop_count(
    x_shape: List[int],
    w_shape: List[int],
    out_shape: List[int],
    transposed: bool = False,
) -> float:
    """
    Count flops for convolution. Note only multiplication is
    counted. Computation for addition and bias is ignored.
    Flops for a transposed convolution are calculated as
    flops = (x_shape[2:] * prod(w_shape) * batch_size).
    Args:
        x_shape (list(int)): The input shape before convolution.
        w_shape (list(int)): The filter shape.
        out_shape (list(int)): The output shape after convolution.
        transposed (bool): is the convolution transposed
    Returns:
        int: the number of flops
    """
    batch_size = x_shape[0]
    conv_shape = (x_shape if transposed else out_shape)[2:]
    flop = batch_size * prod(w_shape) * prod(conv_shape)
    return flop


def conv_flop(inputs: List[Any], outputs: List[Any]):
    """
    Count flops for convolution.
    """
    x, w = inputs[:2]
    x_shape, w_shape, out_shape = (get_shape(x), get_shape(w), get_shape(outputs[0]))
    transposed = inputs[6]

    return conv_flop_count(x_shape, w_shape, out_shape, transposed=transposed)


def transpose_shape(shape):
    return [shape[1], shape[0]] + list(shape[2:])


def conv_backward_flop(inputs: List[Any], outputs: List[Any]):
    grad_out_shape, x_shape, w_shape = [get_shape(i) for i in inputs[:3]]
    output_mask = inputs[-1]
    fwd_transposed = inputs[7]
    flop_count = 0.0

    if output_mask[0]:
        grad_input_shape = get_shape(outputs[0])
        flop_count += conv_flop_count(
            grad_out_shape, w_shape, grad_input_shape, not fwd_transposed
        )
    if output_mask[1]:
        grad_weight_shape = get_shape(outputs[1])
        flop_count += conv_flop_count(
            transpose_shape(x_shape), grad_out_shape, grad_weight_shape, fwd_transposed
        )

    return flop_count


def tensor_storage_size_in_mem(x: torch.Tensor):
    total = 1
    for dim_sz, stride in zip(x.shape, x.stride()):
        if stride >= 1:
            total *= dim_sz
    return total


def get_size(inputs: List[Any]):
    total_bytes = 0

    def process(x) -> None:
        nonlocal total_bytes
        if isinstance(x, torch.Tensor):
            total_bytes += tensor_storage_size_in_mem(x) * x.element_size()

    tree_map(process, inputs)
    return total_bytes


def operation_memory_rw_bytes(inputs: List[Any], outputs: List[Any]):
    size_input, size_output = get_size(inputs), get_size(outputs)
    return size_input + size_output


def output_read_from_input(inputs: List[Any], outputs: List[Any]):
    size_input, size_output = get_size(inputs), get_size(outputs)
    return size_output + min(size_input, size_output)


def output_total_size(inputs: List[Any], outputs: List[Any]):
    return get_size(outputs)


def input_total_size(inputs: List[Any], outputs: List[Any]):
    return get_size(inputs)


def guess_flops_unknown_op(inputs: List[Any], outputs: List[Any]):
    # Approximation that isn't too bad
    total_elements = 0

    def process(x) -> None:
        nonlocal total_elements
        if isinstance(x, torch.Tensor):
            total_elements += x.numel()

    tree_map(process, inputs)
    tree_map(process, outputs)
    return total_elements / 2


def no_flop(inputs: List[Any], outputs: List[Any]):
    return 0


def no_io(inputs: List[Any], outputs: List[Any]):
    return 0


aten = torch.ops.aten
NO_FLOPS_NO_IO_OPS = [
    aten.permute,
    aten.view,
    aten.view_as,
    aten.detach,
    aten.t,
    aten.transpose,
    aten.expand,
    aten._unsafe_view,
    aten.select,
    aten.split,
    aten.split_with_sizes,
    aten.empty,
    aten.empty_strided,
    aten.empty_like,
    aten.is_same_size,
]
NO_FLOPS_OPS = [
    aten._reshape_alias,
    aten.reshape,
    aten.clone,
    aten.cat,
    aten.select_backward,
    aten.slice,
    aten.slice_backward,
    aten.ones,
    aten.ones_like,
    aten.zeros_like,
    aten.zero_,
    aten.zeros,
    aten.masked_fill,
    aten.masked_fill_,
]

flop_mapping = {
    aten.mv: GemmOpComputeFlopsMv(),  # mat-vec
    aten.mm: GemmOpComputeFlops(),
    aten.matmul: GemmOpComputeFlops(),
    aten.addmm: GemmOpComputeFlopsAddmm(),
    aten.bmm: GemmOpComputeFlopsBmm(),
    aten.addbmm: GemmOpComputeFlopsAddbmm(),
    aten.linear: GemmOpComputeFlopsLinear(),
    aten.convolution: conv_flop,
    aten._convolution: conv_flop,
    aten.convolution_backward: conv_backward_flop,
    # Operations with 0 flop
    **{op: no_flop for op in NO_FLOPS_OPS},
    **{op: no_flop for op in NO_FLOPS_NO_IO_OPS},
}
io_mapping = {
    aten.clone: output_read_from_input,
    aten.cat: output_read_from_input,
    aten.slice: output_read_from_input,
    aten.ones_like: output_total_size,
    aten.zeros_like: output_total_size,
    aten.zero_: input_total_size,
    **{op: no_io for op in NO_FLOPS_NO_IO_OPS}
    # TODO: Check how this is implemented in PT
    # aten.slice_backward: no_flop,
    # aten.select_backward: no_flop,
}


@dataclass
class _OpInfo:
    flop_count: float = 0.0
    time_ms: float = 0.0
    io_bytes: int = 0
    is_exact_flop: bool = True
    op_name: str = ""
    op_suffix: str = ""
    stacktrace: Tuple[str, ...] = field(default_factory=tuple)
    ev_start: torch.cuda.Event = field(
        default_factory=lambda: torch.cuda.Event(enable_timing=True)
    )
    ev_end: torch.cuda.Event = field(
        default_factory=lambda: torch.cuda.Event(enable_timing=True)
    )

    # Hardware limits for this operation (inf if unknown)
    hardware_tflops_limit: float = math.inf
    hardware_membw_limit: float = math.inf

    @property
    def time_membound_ms(self) -> float:
        assert self.time_ms > 0.0
        if self.io_bytes == 0:
            return 0.0
        return min(self.time_ms, 1000 * self.io_bytes / self.hardware_membw_limit)

    @property
    def time_computebound_ms(self) -> float:
        assert self.time_ms > 0.0
        tflop = self.flop_count / (1000**4)
        if tflop == 0.0:
            return 0.0
        return min(self.time_ms, 1000 * tflop / self.hardware_tflops_limit)

    def finalize(self) -> None:
        self.time_ms = self.ev_start.elapsed_time(self.ev_end)


@dataclass
class _OpInfoAggregated:
    is_exact_flop: bool = True
    total_flop_count: float = 0.0
    total_io_bytes: int = 0
    total_time_ms: float = 0.0
    total_time_membound_ms: float = 0.0
    total_time_computebound_ms: float = 0.0
    num: int = 0
    stacktraces: List[Tuple[str, ...]] = field(default_factory=list)

    def add(self, op: _OpInfo) -> None:
        self.total_flop_count += op.flop_count
        self.total_time_ms += op.time_ms
        self.total_io_bytes += op.io_bytes
        self.total_time_membound_ms += op.time_membound_ms
        self.total_time_computebound_ms += op.time_computebound_ms
        self.num += 1
        self.is_exact_flop = op.is_exact_flop
        self.stacktraces.append(op.stacktrace)

    def as_dict(self, **kwargs) -> Dict[str, Any]:
        mem_bound = min(1, self.total_time_membound_ms / self.total_time_ms)
        tflops = self.total_flop_count / (self.total_time_ms / 1000) / (1000**4)
        compute_bound = min(1, self.total_time_computebound_ms / self.total_time_ms)
        return {
            "is_exact_flop": self.is_exact_flop,
            "total_flop_count": self.total_flop_count,
            "total_time_ms": self.total_time_ms,
            "total_io_bytes": self.total_io_bytes,
            "num": self.num,
            "Tflops": tflops,
            "mem_bound": mem_bound,
            "compute_bound": compute_bound,
            **kwargs,
        }


class DetectSlowOpsProfiler(DispatcherWithoutBrokenFuncs):
    """
    Inspired from https://fb.workplace.com/groups/pytorch.dev/permalink/1054537595124720/
    """

    def __init__(self, main_profiler: _Profiler) -> None:
        self.main_profiler = main_profiler
        self.trace: List[_OpInfo] = []
        self.temp_disabled = False

    def _hardware_tflops_membw_limit(
        self, args: Tuple[Any, ...], outputs: Tuple[Any, ...]
    ) -> Tuple[float, float]:
        device = None
        dtypes: List[torch.dtype] = []
        for a in itertools.chain(outputs, args):
            if isinstance(a, torch.Tensor):
                if device is None:
                    device = a.device
                dtypes.append(a.dtype)
        limits = get_device_limits(device)
        if not limits:
            return (math.inf, math.inf)
        dtypes = [dt for dt in dtypes if dt in limits.gemm_tflops]
        if not dtypes or device is None:
            return (math.inf, math.inf)
        dtype = dtypes[0]
        if torch.is_autocast_enabled() and dtype is torch.float32:
            dtype = torch.get_autocast_gpu_dtype()
        return limits.gemm_tflops[dtype], limits.gmem_bandwidth

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        func_packet = func._overloadpacket
        if self.temp_disabled or func_packet.__name__ in [
            "_record_function_exit",
            "_record_function_enter_new",
        ]:
            return func(*args, **kwargs)

        op = _OpInfo()
        op.ev_start.record()
        out = func(*args, **kwargs)
        op.ev_end.record()

        (
            op.hardware_tflops_limit,
            op.hardware_membw_limit,
        ) = self._hardware_tflops_membw_limit(
            args, out if isinstance(out, tuple) else (out,)
        )
        op.op_name = func_packet.__name__
        # Prevent functions called by flop counting ops to be recorded
        self.temp_disabled = True
        flop_count = -1
        compute_flops = None
        if func_packet in FUNC_TO_XFORMERS_OPERATOR:
            flop_count = FUNC_TO_XFORMERS_OPERATOR[func_packet].operator_flop(
                *args, **kwargs
            )
        if flop_count == -1:
            compute_flops = flop_mapping.get(func_packet, guess_flops_unknown_op)
            flop_count = compute_flops(args, out if isinstance(out, tuple) else (out,))
            if isinstance(compute_flops, GemmOpComputeFlops):
                op.op_name += compute_flops.op_suffix(args)

        compute_io = io_mapping.get(func_packet, operation_memory_rw_bytes)
        op.io_bytes = compute_io(args, out if isinstance(out, tuple) else (out,))
        self.temp_disabled = False

        op.stacktrace = tuple(self.main_profiler.parents)
        op.flop_count = flop_count
        op.is_exact_flop = compute_flops is not guess_flops_unknown_op
        self.trace.append(op)

        return out

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        torch.cuda.synchronize()  # Wait for the events to be recorded
        for op in self.trace:
            op.finalize()
        self.save_json()

    def step(self) -> None:
        pass

    def save_json(self) -> None:
        # Aggregate data at the module + op level
        all_paths: Set[Tuple[str, ...]] = set()
        per_module_data: Dict[Tuple[str, ...], _OpInfoAggregated] = defaultdict(
            _OpInfoAggregated
        )
        per_op_data: Dict[str, _OpInfoAggregated] = defaultdict(_OpInfoAggregated)
        for op in self.trace:
            all_paths.add(op.stacktrace)
        for op in self.trace:
            for i in range(len(op.stacktrace)):
                if op.stacktrace[: i + 1] in all_paths:
                    per_module_data[op.stacktrace[: i + 1]].add(op)
            per_op_data[op.op_name].add(op)

        # Generate JSON
        all_data = []
        for stacktrace, agg_info in per_module_data.items():
            all_data.append(
                agg_info.as_dict(
                    agg="module", path=stacktrace, name=stacktrace[-1], op=""
                )
            )
        for op_name, agg_info in per_op_data.items():
            # Find the most common path
            paths_count: Dict[Tuple[str, ...], int] = defaultdict(int)
            agg_info.stacktraces.sort()  # In case of a draw, let's always return the same
            for p in agg_info.stacktraces:
                paths_count[p] += 1
            maxp = agg_info.stacktraces[0]
            for p, count in paths_count.items():
                if count > paths_count[maxp]:
                    maxp = p
            all_data.append(
                agg_info.as_dict(
                    agg="opname",
                    path=f"{'.'.join(maxp)} (x{paths_count[maxp]})",
                    name="",
                    op=op_name,
                )
            )

        filename = self.main_profiler._create_output_filename("ops.json")
        self.main_profiler.summary.append(("OpsSummary", str(filename)))
        with open(filename, "w+") as f:
            json.dump(all_data, f)
