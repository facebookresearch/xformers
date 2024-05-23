# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.distributed as dist

from xformers.ops import init_ipc

from .multiprocessing_utils import launch_subprocesses

compute_capability = (0, 0)
if torch.cuda.is_available():
    compute_capability = torch.cuda.get_device_capability("cuda")
cuda_sm70_only = pytest.mark.skipif(
    compute_capability < (7, 0), reason="requires sm70+"
)


def inner_test_ipc() -> None:
    my_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    subgroup = torch.distributed.new_group()

    ipcs = init_ipc(subgroup)

    send_bufs = [
        torch.full([1], my_rank, device="cuda", dtype=torch.int32)
        for _ in range(world_size)
    ]
    recv_bufs = send_bufs.copy()
    for other_rank, conn in enumerate(ipcs):
        if conn is None:
            continue
        conn.send(send_bufs[other_rank])
    for other_rank, conn in enumerate(ipcs):
        if conn is None:
            continue
        recv_bufs[other_rank] = conn.recv()

    torch.cuda.synchronize()
    dist.barrier(subgroup)

    # Use the buffer to send data
    for other_rank, buf in enumerate(recv_bufs):
        assert buf[0].item() == other_rank
        buf.fill_(my_rank)

    torch.cuda.synchronize()
    dist.barrier(subgroup)

    # Verify we've received the data correctly
    for other_rank, buf in enumerate(send_bufs):
        assert (
            buf[0].item() == other_rank
        ), f"[#{my_rank}] {other_rank=} != {buf[0].item()=}"


@cuda_sm70_only
def test_ipc() -> None:
    world_size = 2
    launch_subprocesses(
        world_size=world_size,
        fn=inner_test_ipc,
    )


# We had an issue where the second rendezvous in a single process would use the
# same store keys as the first one, thus retrieve a stale address to connect to,
# and fail.
def inner_test_ipc_twice() -> None:
    subgroup = torch.distributed.new_group()

    init_ipc(subgroup)
    init_ipc(subgroup)


@cuda_sm70_only
def test_ipc_twice() -> None:
    world_size = 2
    launch_subprocesses(
        world_size=world_size,
        fn=inner_test_ipc_twice,
    )
