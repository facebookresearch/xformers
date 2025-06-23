# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Callable, Dict, List, Optional, overload, Sequence, Union

import torch
import torch.distributed as dist
import torch.multiprocessing.reductions
from torch.distributed._symmetric_memory import get_symm_mem_workspace

OP_FINISHED_CHANNEL = 0
COMMS_READY_CHANNEL = 1

MS_IN_S = 1_000


def _is_fp8_dtype(dt: torch.dtype):
    # Detect if it's float8_e4m3fn or float8_e5m2 without mentioning them in
    # order to support old versions of PyTorch that don't define them.
    return dt.is_floating_point and torch.finfo(dt).bits == 8


class _FusedSequenceParallel:
    """Set up a communication ring and perform fused ops on it

    Stores the persistent state needed to support a ring of connections between
    processes, and the logic that can do fused comms + matmuls on it.

    We want to achieve overlap between:
    - a computation which reads from the data we received from a remote GPU
    - and the communication where we send some data to another GPU
    And in order to do that we need some staging buffers and a way to
    synchronize access to them across processes.

    To perform the communication over NVLink we make the processes exchange
    their staging buffers using IPC (Inter-Process Communication) handles, which
    "mounts"/"mmaps" an allocation on one GPU into the virtual address space of
    another GPU: the memory remains backed by the original GPU but the other GPU
    can access it as if it were local. We exchange these IPC handles using
    multiprocessing Connections (and the "reductions" provided by PyTorch),
    which we establish over UNIX domain sockets, whose addresses we exchange by
    using a ProcessGroup.

    To synchronize accesses we use a set of counters/sequence numbers that are
    also allocated in memory shared over IPC handles. Processes signal that they
    completed an operation by launching a kernel that increases that value, and
    they wait for anoher process to complete an operation by launching a kernel
    that busy-waits for that value to increase. Currently we implement these
    kernels manually, but on recent CUDA drivers (515.43.04+, corresponding to
    CUDA 11.7) we could use standard stream memory operations (see
    https://docs.nvidia.com/cuda/archive/11.7.0/cuda-driver-api/group__CUDA__MEMOP.html).

    We prefer to use these kernels (or the stream memory ops) over IPC events
    because IPC events require signaling between processes at launch time to
    ensure that the wait on one process occurs after the record on another
    process. This signaling means that _launching_ our fused operation becomes a
    synchronization barrier, which can increase the launch overhead. It would
    also behave differently from NCCL, where launching is async and all the
    synchronization happens on device in the kernels. A previous version of this
    code which uses IPC events can be found here:
    https://github.com/fairinternal/xformers/pull/504.

    """

    def __init__(
        self,
        device: torch.device,
        group: dist.ProcessGroup,
    ):
        self.my_device = device
        self.my_rank = group.rank()
        self.world_size = group.size()
        self.group = group

        self.second_stream = torch.cuda.Stream()
        # CUDA can schedule the matmul and the memcpy at the same time, but it
        # tends to run the matmul first and delay the memcpy, which causes a
        # domino effect. We thus "encourage" it to prioritize the memcpy.
        self.memcpy_stream = torch.cuda.Stream(priority=-1)
        # Use dedicated streams to run the wait kernels in the background.
        self.compute_wait_stream = torch.cuda.Stream(priority=-1)
        self.memcpy_wait_stream = torch.cuda.Stream(priority=-1)

        self.next_stream_idx = 0

    def make_stream_factory(
        self, current_stream: torch.cuda.Stream
    ) -> Callable[[], torch.cuda.Stream]:
        def result():
            stream = [current_stream, self.second_stream][self.next_stream_idx]
            self.next_stream_idx += 1
            self.next_stream_idx %= 2
            return stream

        return result

    def allgather_and_linear(
        self,
        scattered_inputs: List[torch.Tensor],
        my_matmul: Callable[
            [List[torch.Tensor], int, Callable[[], torch.cuda.Stream]], None
        ],
        timeout_s: int,
        _wait: bool = True,
        _memcpy: bool = True,
    ):
        """Perform a fused all-gather followed by a linear layer"""

        dtype = scattered_inputs[0].dtype
        assert all(si.device == self.my_device for si in scattered_inputs)
        assert all(si.dtype == dtype for si in scattered_inputs)

        scattered_input_numels = [si.numel() for si in scattered_inputs]
        total_scattered_input_numel = sum(scattered_input_numels)

        with torch.cuda.device(self.my_device):
            symm_mem = get_symm_mem_workspace(
                self.group.group_name,
                self.world_size * total_scattered_input_numel * dtype.itemsize,
            )
            # FIXME Do something about random_init if _memcpy is True.
            buffers = [
                [
                    s.view((self.world_size,) + si.shape)
                    for s, si in zip(
                        symm_mem.get_buffer(
                            rank, [self.world_size, total_scattered_input_numel], dtype
                        ).split(scattered_input_numels, dim=-1),
                        scattered_inputs,
                    )
                ]
                for rank in range(self.world_size)
            ]

        current_stream = torch.cuda.current_stream()

        # Signal to buddy that we have read from the data (in previous iter) so
        # it can overwrite it (this write matches up with wait [B] below).
        for iter_ in range(1, self.world_size):
            src_rank = (self.my_rank - iter_) % self.world_size
            if _wait:
                with torch.cuda.stream(current_stream):
                    symm_mem.put_signal(src_rank, OP_FINISHED_CHANNEL)

        self.second_stream.wait_stream(current_stream)
        self.compute_wait_stream.wait_stream(current_stream)
        self.memcpy_wait_stream.wait_stream(current_stream)
        stream_factory = self.make_stream_factory(current_stream)

        for iter_ in range(1, self.world_size):
            dst_rank = (self.my_rank + iter_) % self.world_size

            # Wait for buddy to signal that it read from the data before we
            # overwrite it (this wait matches up with write [B] above).
            if _wait:
                with torch.cuda.stream(self.memcpy_wait_stream):
                    symm_mem.wait_signal(
                        dst_rank,
                        OP_FINISHED_CHANNEL,
                        timeout_ms=timeout_s * MS_IN_S,  # type: ignore[call-arg]
                    )

            self.memcpy_stream.wait_stream(self.memcpy_wait_stream)

            if _memcpy:
                with torch.cuda.stream(self.memcpy_stream):
                    for bs, si in zip(buffers[dst_rank], scattered_inputs):
                        bs[self.my_rank].copy_(si)

            # Signal to buddy that we have written into the data so it can
            # read from it (this write matches up with wait [A] below).
            if _wait:
                with torch.cuda.stream(self.memcpy_stream):
                    symm_mem.memset32(
                        symm_mem.get_signal_pad(dst_rank),  # type: ignore[attr-defined]
                        self.world_size * COMMS_READY_CHANNEL + self.my_rank,
                        val=1,
                        count=1,
                    )

        my_matmul(scattered_inputs, self.my_rank, stream_factory)

        for iter_ in range(1, self.world_size):
            src_rank = (self.my_rank - iter_) % self.world_size

            # Wait for buddy to signal that it wrote into the data before we
            # read from it (this wait matches up with write [A] above).
            if _wait:
                with torch.cuda.stream(self.compute_wait_stream):
                    symm_mem.wait_signal(
                        src_rank,
                        COMMS_READY_CHANNEL,
                        timeout_ms=timeout_s * MS_IN_S,  # type: ignore[call-arg]
                    )

            current_stream.wait_stream(self.compute_wait_stream)
            self.second_stream.wait_stream(self.compute_wait_stream)

            my_matmul(
                [s[src_rank] for s in buffers[self.my_rank]], src_rank, stream_factory
            )

        current_stream.wait_stream(self.second_stream)
        current_stream.wait_stream(self.memcpy_stream)

    def linear_and_reducescatter(
        self,
        my_matmul: Callable[
            [List[torch.Tensor], int, Callable[[], torch.cuda.Stream]], None
        ],
        gathered_outputs: List[torch.Tensor],
        scattered_outputs: List[torch.Tensor],
        timeout_s: int,
        _wait: bool = True,
        _memcpy: bool = True,
    ):
        """Perform a fused linear layer followed by a reduce-scatter"""

        dtype = gathered_outputs[0].dtype
        assert all(go.device == self.my_device for go in gathered_outputs)
        assert all(go.dtype == dtype for go in gathered_outputs)
        assert all(so.device == self.my_device for so in scattered_outputs)
        assert all(so.dtype == dtype for so in scattered_outputs)

        scattered_output_numels = [so.numel() for so in scattered_outputs]
        total_scattered_output_numel = sum(scattered_output_numels)

        with torch.cuda.device(self.my_device):
            symm_mem = get_symm_mem_workspace(
                self.group.group_name,
                self.world_size * total_scattered_output_numel * dtype.itemsize,
            )
            # FIXME Do something about random_init if _memcpy is True.
            buffers = [
                [
                    s.view((self.world_size,) + so.shape)
                    for s, so in zip(
                        symm_mem.get_buffer(
                            rank, [self.world_size, total_scattered_output_numel], dtype
                        ).split(scattered_output_numels, dim=-1),
                        scattered_outputs,
                    )
                ]
                for rank in range(self.world_size)
            ]

        current_stream = torch.cuda.current_stream()

        # Signal to buddy that we have read from the data (in previous iter)
        # so it can overwrite it (this write matches up with wait [2] below).
        for iter_ in range(1, self.world_size):
            src_rank = (self.my_rank - iter_) % self.world_size
            if _wait:
                with torch.cuda.stream(current_stream):
                    symm_mem.put_signal(src_rank, OP_FINISHED_CHANNEL)

        self.second_stream.wait_stream(current_stream)
        self.compute_wait_stream.wait_stream(current_stream)
        self.memcpy_wait_stream.wait_stream(current_stream)
        stream_factory = self.make_stream_factory(current_stream)

        for iter_ in range(1, self.world_size):
            dst_rank = (self.my_rank + iter_) % self.world_size

            # Wait for buddy to signal that it read from the data before we
            # overwrite it (this wait matches up with write [2] above).
            if _wait:
                with torch.cuda.stream(self.compute_wait_stream):
                    symm_mem.wait_signal(
                        dst_rank,
                        OP_FINISHED_CHANNEL,
                        timeout_ms=timeout_s * MS_IN_S,  # type: ignore[call-arg]
                    )

            current_stream.wait_stream(self.compute_wait_stream)
            self.second_stream.wait_stream(self.compute_wait_stream)

            my_matmul(
                [s[dst_rank] for s in buffers[self.my_rank]], dst_rank, stream_factory
            )

            # Deduce which stream contains the last kernel launched.
            final_stream = [current_stream, self.second_stream][
                (self.next_stream_idx - 1) % 2
            ]
            final_stream.wait_stream(current_stream)
            final_stream.wait_stream(self.second_stream)

            # Signal to buddy that we have written into the data so it can
            # read from it (this write matches up with wait [1] below).
            if _wait:
                with torch.cuda.stream(final_stream):
                    symm_mem.memset32(
                        symm_mem.get_signal_pad(dst_rank),  # type: ignore[attr-defined]
                        self.world_size * COMMS_READY_CHANNEL + self.my_rank,
                        val=1,
                        count=1,
                    )

        my_matmul(
            [o[self.my_rank] for o in gathered_outputs],
            self.my_rank,
            stream_factory,
        )

        for iter_ in range(1, self.world_size):
            src_rank = (self.my_rank - iter_) % self.world_size

            # Wait for buddy to signal that it wrote into the data before we
            # read from it (this wait matches up with write [1] above).
            if _wait:
                with torch.cuda.stream(self.memcpy_wait_stream):
                    symm_mem.wait_signal(
                        src_rank,
                        COMMS_READY_CHANNEL,
                        timeout_ms=timeout_s * MS_IN_S,  # type: ignore[call-arg]
                    )

            self.memcpy_stream.wait_stream(self.memcpy_wait_stream)

            if _memcpy:
                with torch.cuda.stream(self.memcpy_stream):
                    for go, bs in zip(gathered_outputs, buffers[src_rank]):
                        go[src_rank].copy_(bs[self.my_rank])

        current_stream.wait_stream(self.second_stream)
        current_stream.wait_stream(self.memcpy_stream)

        for go, so in zip(gathered_outputs, scattered_outputs):
            torch.sum(go, dim=0, out=so)


# We'd store this as an attribute on the PG object itself, but some PGs are
# pybind-bound classes and thus don't support it, so we simulate this as an
# external cache.
CACHE: Dict[int, Optional[_FusedSequenceParallel]] = {}


def _can_ranks_communicate_all_to_all_over_nvlink(group: dist.ProcessGroup) -> bool:
    # FIXME This is currently overly simplistic, must be improved. The following
    # should be enough:
    # - ensure that all ranks are running on the same machine (by exchanging
    #   their /proc/sys/kernel/random/boot_id value)
    # - ensure there's P2P between all pairs of ranks (can_device_access_peer
    #   could help here but it's unclear what happens if target devices aren't
    #   visible? maybe just trying to exchange IPC handles and catching errors
    #   would work? note that in any case some ranks might succeed while some
    #   might fail so we need a barrier to have them all make the same decision)
    return group.size() <= 8


def _lazy_init(
    device: torch.device, group: dist.ProcessGroup
) -> Optional[_FusedSequenceParallel]:
    world_size = group.size()
    try:
        obj = CACHE[id(group)]
    except KeyError:
        if int(os.environ.get("DISABLE_FUSED_SEQUENCE_PARALLEL", "0")):
            obj = None
        elif world_size == 1:
            obj = None
        elif not _can_ranks_communicate_all_to_all_over_nvlink(group):
            obj = None
        else:
            obj = _FusedSequenceParallel(device, group)
        CACHE[id(group)] = obj
    return obj


def _default_stream_factory() -> torch.cuda.Stream:
    return torch.cuda.current_stream()


@overload
def fused_allgather_and_linear(
    scattered_input: torch.Tensor,
    weight: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    out: Optional[torch.Tensor] = None,
    timeout_s: int = 60 * 60,
    scale_scattered_input: Optional[torch.Tensor] = None,
    scale_weight: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    out_dtype: Optional[torch.dtype] = None,
    **private_args_DO_NOT_USE,
) -> torch.Tensor: ...


@overload
def fused_allgather_and_linear(
    scattered_input: torch.Tensor,
    weight: List[torch.Tensor],
    *,
    group: dist.ProcessGroup,
    out: Optional[List[torch.Tensor]] = None,
    timeout_s: int = 60 * 60,
    scale_scattered_input: Optional[torch.Tensor] = None,
    scale_weight: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    out_dtype: Optional[torch.dtype] = None,
    **private_args_DO_NOT_USE,
) -> List[torch.Tensor]: ...


def fused_allgather_and_linear(
    scattered_input: torch.Tensor,
    weight: Union[torch.Tensor, List[torch.Tensor]],
    *,
    group: dist.ProcessGroup,
    out: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    timeout_s: int = 60 * 60,
    scale_scattered_input: Optional[torch.Tensor] = None,
    scale_weight: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    out_dtype: Optional[torch.dtype] = None,
    **private_args_DO_NOT_USE,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Performs a fused all-gather followed by a linear op

    It is equivalent to the following plain PyTorch code:

    # like scattered_input but with first dim multiplied by group's world size
    gathered_input = scattered_input.new_empty(...)
    dist.all_gather_into_tensor(gathered_input, scattered_input, group=group)
    return torch.nn.functional.linear(gathered_input, weight)

    It achieves this by breaking down the matmul into smaller partial ops (as
    many as the world size), each needing as input a different "contribution"
    to the all-gather (by a different rank), and writing to a different chunk of
    the output. Then, on one stream, it sends the local contribution to all
    other ranks (first one rank over, then two, ...) while, on another stream,
    it launches the sub-matmuls in the order in which the remote contributions
    (which are the sub-matmuls' inputs) are supposed to arrive, so that ideally
    none of the sub-matmuls will ever have to wait.

    The idea comes from this paper: https://arxiv.org/abs/2302.05442

    This method uses a staging buffer, which persists across calls, of the same
    size as the all-gathered input tensor (i.e., the input's size times the
    world size). If multiple inputs of multiple sizes are used, the staging
    buffer will be the maximum needed by any of them. Each call, when it starts,
    must first wait for the previous call to finish using the staging buffer. In
    normal conditions, where there's some other operation between two calls,
    this isn't an issue.

    Supports FP8 gemm for tensor-wise quantized weight and input tensors.
    To enable FP8 gemm:
    1. pass scattered_input and weight as quantized FP8 datatype
    2. pass scale_scattered_input and scale_weight, the scales used to
    quantize input and weight, respectively.
    3. set out_dtype, if not specified, will be inferred from scattered_input type.

    """
    world_size = group.size()
    weights = weight if isinstance(weight, list) else [weight]
    assert (scale_scattered_input is None) == (scale_weight is None)
    if scale_weight is not None:
        assert isinstance(weight, list) == isinstance(scale_weight, list)
        scales_weights: Sequence[Optional[torch.Tensor]] = (
            scale_weight if isinstance(scale_weight, list) else [scale_weight]
        )
        assert len(weights) == len(scales_weights)
        assert _is_fp8_dtype(scattered_input.dtype)
        assert all(_is_fp8_dtype(w.dtype) for w in weights)
        assert out_dtype is not None, "output_dtype is required with FP8"
    else:
        scales_weights = [None] * len(weights)
    assert all(w.ndim == 2 for w in weights)
    assert scattered_input.ndim >= 2
    assert all(scattered_input.shape[-1] == w.shape[-1] for w in weights)
    assert scattered_input.is_contiguous()
    gathered_input_shape = (world_size,) + scattered_input.shape
    gathered_output_shapes = [gathered_input_shape[:-1] + w.shape[:-1] for w in weights]
    if out is not None:
        assert isinstance(out, list) == isinstance(weight, list)
        gathered_outputs = out if isinstance(out, list) else [out]
        assert len(gathered_outputs) == len(gathered_output_shapes)
        assert all(
            go.shape == gos for go, gos in zip(gathered_outputs, gathered_output_shapes)
        )
        assert all(go.is_contiguous() for go in gathered_outputs)
        if out_dtype is not None:
            if isinstance(out, list):
                for o in out:
                    assert o.dtype == out_dtype
            else:
                assert out.dtype == out_dtype
    else:
        gathered_outputs = [
            scattered_input.new_empty(
                gos,
                dtype=out_dtype if out_dtype is not None else scattered_input.dtype,
            )
            for gos in gathered_output_shapes
        ]

    torch.ops.xformers_python._fused_allgather_and_linear_impl(
        scattered_input,
        weights,
        group.group_name,
        gathered_outputs,
        timeout_s=timeout_s,
        _wait=private_args_DO_NOT_USE.get("_wait", True),
        _memcpy=private_args_DO_NOT_USE.get("_memcpy", True),
        scale_scattered_input=scale_scattered_input,
        scales_weights=scales_weights,
    )

    if isinstance(weight, list):
        return [go.flatten(0, 1) for go in gathered_outputs]
    else:
        return gathered_outputs[0].flatten(0, 1)


@torch.library.custom_op(
    "xformers_python::_fused_allgather_and_linear_impl",
    mutates_args={"gathered_outputs"},
    device_types="cuda",
)
def _fused_allgather_and_linear_custom_op(
    scattered_input: torch.Tensor,
    weights: List[torch.Tensor],
    process_group_name: str,
    gathered_outputs: List[torch.Tensor],
    timeout_s: int,
    _wait: bool,
    _memcpy: bool,
    scale_scattered_input: torch.Tensor,
    scales_weights: Sequence[Optional[torch.Tensor]],
) -> None:
    process_group = dist.distributed_c10d._resolve_process_group(process_group_name)

    def my_matmul(
        inputs: List[torch.Tensor],
        src_rank: int,
        stream_factory: Callable[[], torch.cuda.Stream],
    ) -> None:
        for w, scale_weight, go in zip(weights, scales_weights, gathered_outputs):
            with torch.cuda.stream(stream_factory()):
                if scale_scattered_input is not None and scale_weight is not None:
                    torch._scaled_mm(
                        inputs[0],
                        w.t(),
                        out_dtype=go[src_rank].dtype,
                        scale_a=scale_scattered_input,
                        scale_b=scale_weight,
                        out=go[src_rank],
                    )
                else:
                    torch.matmul(inputs[0], w.t(), out=go[src_rank])

    fused_allgather_and_anything(
        [scattered_input],
        my_matmul,
        group=process_group,
        timeout_s=timeout_s,
        _wait=_wait,
        _memcpy=_memcpy,
    )


def fused_allgather_and_anything(
    scattered_inputs: List[torch.Tensor],
    my_matmul: Callable[
        [List[torch.Tensor], int, Callable[[], torch.cuda.Stream]], None
    ],
    *,
    group: dist.ProcessGroup,
    timeout_s: int = 60 * 60,
    **private_args_DO_NOT_USE,
) -> None:
    world_size = group.size()

    if len(scattered_inputs) == 0:
        for src_rank in range(world_size):
            my_matmul([], src_rank, _default_stream_factory)
        return

    assert all(si.is_contiguous() for si in scattered_inputs)
    assert all(si.device == scattered_inputs[0].device for si in scattered_inputs)
    assert all(si.dtype == scattered_inputs[0].dtype for si in scattered_inputs)

    gathered_input_shapes = [(world_size,) + si.shape for si in scattered_inputs]

    obj = _lazy_init(scattered_inputs[0].device, group)

    if world_size == 1:
        my_matmul(scattered_inputs, 0, _default_stream_factory)

    # Fallback
    elif obj is None:
        gathered_inputs = [
            si.new_empty(gis)
            for si, gis in zip(scattered_inputs, gathered_input_shapes)
        ]
        for si, gi in zip(scattered_inputs, gathered_inputs):
            dist.all_gather_into_tensor(output_tensor=gi, input_tensor=si, group=group)
        for src_rank in range(world_size):
            my_matmul(
                [gi[src_rank] for gi in gathered_inputs],
                src_rank,
                _default_stream_factory,
            )

    # Fast path
    else:
        assert scattered_inputs[0].device == obj.my_device
        obj.allgather_and_linear(
            scattered_inputs,
            my_matmul,
            timeout_s=timeout_s,
            _wait=private_args_DO_NOT_USE.get("_wait", True),
            _memcpy=private_args_DO_NOT_USE.get("_memcpy", True),
        )


@overload
def fused_linear_and_reducescatter(
    gathered_input: torch.Tensor,
    weight: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    out: Optional[torch.Tensor] = None,
    timeout_s: int = 60 * 60,
    scale_gathered_input: Optional[torch.Tensor] = None,
    scale_weight: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    out_dtype: Optional[torch.dtype] = None,
    **private_args_DO_NOT_USE,
) -> torch.Tensor: ...


@overload
def fused_linear_and_reducescatter(
    gathered_input: torch.Tensor,
    weight: List[torch.Tensor],
    *,
    group: dist.ProcessGroup,
    out: Optional[List[torch.Tensor]] = None,
    timeout_s: int = 60 * 60,
    scale_gathered_input: Optional[torch.Tensor] = None,
    scale_weight: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    out_dtype: Optional[torch.dtype] = None,
    **private_args_DO_NOT_USE,
) -> List[torch.Tensor]: ...


def fused_linear_and_reducescatter(
    gathered_input: torch.Tensor,
    weight: Union[torch.Tensor, List[torch.Tensor]],
    *,
    group: dist.ProcessGroup,
    out: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    timeout_s: int = 60 * 60,
    scale_gathered_input: Optional[torch.Tensor] = None,
    scale_weight: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    out_dtype: Optional[torch.dtype] = None,
    **private_args_DO_NOT_USE,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Performs a fused linear op followed by a reduce-scatter

    It is equivalent to the following plain PyTorch code:

    gathered_output = torch.nn.functional.linear(gathered_input, weight)
    # like gathered_output but with first dim divided by group's world size
    scattered_output = gathered_output.new_empty(...)
    dist.reduce_scatter_tensor(scattered_output, gathered_output, group=group)

    Supports FP8 gemm with tensor-wise quantized weights. To enable FP8 gemm:
    1. pass weight and gathered_input as FP8 tensors
    2. Set `scale_gathered_input` and `scale_weight` to the scales used to quantize
    inputs and weight, respectively.
    3. Set out_dtype to the desired output dtype. If not specified, it will be inferred from
    gathered_input datatype.
    """
    world_size = group.size()
    weights = weight if isinstance(weight, list) else [weight]
    assert (scale_gathered_input is None) == (scale_weight is None)
    if scale_weight is not None:
        assert isinstance(weight, list) == isinstance(scale_weight, list)
        scales_weights: Sequence[Optional[torch.Tensor]] = (
            scale_weight if isinstance(scale_weight, list) else [scale_weight]
        )
        assert len(weights) == len(scales_weights)
        assert _is_fp8_dtype(gathered_input.dtype)
        assert all(_is_fp8_dtype(w.dtype) for w in weights)
        assert out_dtype is not None, "output_dtype is required with FP8"
    else:
        scales_weights = [None] * len(weights)
    assert all(w.ndim == 2 for w in weights)
    assert gathered_input.ndim >= 2
    assert all(gathered_input.shape[-1] == w.shape[-1] for w in weights)
    assert gathered_input.is_contiguous()
    assert gathered_input.shape[0] % world_size == 0
    gathered_input = gathered_input.view(
        (world_size, gathered_input.shape[0] // world_size) + gathered_input.shape[1:]
    )
    gathered_output_shapes = [gathered_input.shape[:-1] + w.shape[:-1] for w in weights]
    scattered_output_shapes = [gos[1:] for gos in gathered_output_shapes]
    if out is not None:
        assert isinstance(out, list) == isinstance(weight, list)
        scattered_outputs = out if isinstance(out, list) else [out]
        assert len(scattered_outputs) == scattered_output_shapes
        assert all(so.device == gathered_input.device for so in scattered_outputs)
        assert all(so.dtype == gathered_input.dtype for so in scattered_outputs)
        assert all(
            so.shape == sos
            for so, sos in zip(scattered_outputs, scattered_output_shapes)
        )
        if out_dtype is not None:
            if isinstance(out, list):
                for o in out:
                    assert o.dtype == out_dtype
            else:
                assert out.dtype == out_dtype
    else:
        scattered_outputs = [
            gathered_input.new_empty(
                sos,
                dtype=out_dtype if out_dtype is not None else gathered_input.dtype,
            )
            for sos in scattered_output_shapes
        ]

    torch.ops.xformers_python._fused_linear_and_reducescatter_impl(
        gathered_input,
        weights,
        group.group_name,
        scattered_outputs,
        timeout_s=timeout_s,
        _wait=private_args_DO_NOT_USE.get("_wait", True),
        _memcpy=private_args_DO_NOT_USE.get("_memcpy", True),
        scale_gathered_input=scale_gathered_input,
        scales_weights=scales_weights,
    )

    if isinstance(weight, list):
        return scattered_outputs
    else:
        return scattered_outputs[0]


@torch.library.custom_op(
    "xformers_python::_fused_linear_and_reducescatter_impl",
    mutates_args={"scattered_outputs"},
    device_types="cuda",
)
def _fused_linear_and_reducescatter_custom_op(
    gathered_input: torch.Tensor,
    weights: List[torch.Tensor],
    process_group_name: str,
    scattered_outputs: List[torch.Tensor],
    timeout_s: int,
    _wait: bool,
    _memcpy: bool,
    scale_gathered_input: torch.Tensor,
    scales_weights: Sequence[Optional[torch.Tensor]],
) -> None:
    process_group = dist.distributed_c10d._resolve_process_group(process_group_name)

    def my_matmul(
        outputs: List[torch.Tensor],
        dst_rank: int,
        stream_factory: Callable[[], torch.cuda.Stream],
    ) -> None:
        for w, scale_weight, o in zip(weights, scales_weights, outputs):
            with torch.cuda.stream(stream_factory()):
                if scale_gathered_input is not None and scale_weight is not None:
                    torch._scaled_mm(
                        gathered_input[dst_rank],
                        w.t(),
                        out_dtype=o.dtype,
                        scale_a=scale_gathered_input,
                        scale_b=scale_weight,
                        out=o,
                    )
                else:
                    torch.matmul(gathered_input[dst_rank], w.t(), out=o)

    fused_anything_and_reducescatter(
        my_matmul,
        scattered_outputs,
        group=process_group,
        timeout_s=timeout_s,
        _wait=_wait,
        _memcpy=_memcpy,
    )


def fused_anything_and_reducescatter(
    my_matmul: Callable[
        [List[torch.Tensor], int, Callable[[], torch.cuda.Stream]], None
    ],
    scattered_outputs: List[torch.Tensor],
    *,
    group: dist.ProcessGroup,
    timeout_s: int = 60 * 60,
    **private_args_DO_NOT_USE,
) -> None:
    world_size = group.size()

    if len(scattered_outputs) == 0:
        for dst_rank in range(world_size):
            my_matmul([], dst_rank, _default_stream_factory)
        return

    assert all(so.is_contiguous() for so in scattered_outputs)
    assert all(so.device == scattered_outputs[0].device for so in scattered_outputs)
    assert all(so.dtype == scattered_outputs[0].dtype for so in scattered_outputs)

    gathered_output_shapes = [(world_size,) + so.shape for so in scattered_outputs]

    obj = _lazy_init(scattered_outputs[0].device, group)

    if world_size == 1:
        my_matmul(scattered_outputs, 0, _default_stream_factory)

    # Fallback
    elif obj is None:
        gathered_outputs = [
            so.new_empty(gos)
            for so, gos in zip(scattered_outputs, gathered_output_shapes)
        ]
        for dst_rank in range(world_size):
            my_matmul(
                [go[dst_rank] for go in gathered_outputs],
                dst_rank,
                _default_stream_factory,
            )
        for go, so in zip(gathered_outputs, scattered_outputs):
            dist.reduce_scatter_tensor(output=so, input=go, group=group)

    # Fast path
    else:
        assert scattered_outputs[0].device == obj.my_device
        gathered_outputs = [
            scattered_outputs[0].new_empty(gos) for gos in gathered_output_shapes
        ]
        obj.linear_and_reducescatter(
            my_matmul,
            gathered_outputs,
            scattered_outputs,
            timeout_s=timeout_s,
            _wait=private_args_DO_NOT_USE.get("_wait", True),
            _memcpy=private_args_DO_NOT_USE.get("_memcpy", True),
        )
