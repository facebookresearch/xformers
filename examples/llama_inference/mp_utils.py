# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import functools
import os
from typing import Optional

import torch
from torch.distributed import ProcessGroup

_GROUP: Optional[ProcessGroup] = None
_WORLD_SIZE: Optional[int] = None
_LOCAL_RANK: int = 0


def initialize(
    world_size: int,
    local_rank: int,
    group: Optional[ProcessGroup] = None,
    use_gpu: bool = True,
    seed: int = 80486,
) -> str:
    """
    Initialize model parallelism support.

    Args:
        world_size (int): the number of processes running on
            the current node available for model parallelism.
        local_rank (int): the present process' rank.
        group (torch.distributed.ProcessGroup, optional): the
            process group to use for model parallel communications.
        use_gpu (bool, optional): whether computations are
            happening on a GPU or not (defaults to True).
        seed (int, optional): the seed used to seed the prng
            on all model parallel processes

    Returns
        The pytorch device to use in the present process.

    Note:
        If ``group`` is not specified, the default process group is
        used for model parallelism. This means that the present
        module may be incompatible with other forms of parallelism
        such as data parallelism.
    """
    global _GROUP
    global _WORLD_SIZE
    global _LOCAL_RANK

    assert local_rank < world_size

    if use_gpu:
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
    else:
        device = "cpu"

    if group is None:
        if "MASTER_ADDR" not in os.environ:
            assert world_size == 1
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = "1234"

        torch.distributed.init_process_group(
            backend="nccl" if use_gpu else "gloo",
            init_method="env://",
            world_size=world_size,
            rank=local_rank,
        )

    _GROUP = group
    _WORLD_SIZE = world_size
    _LOCAL_RANK = local_rank

    torch.manual_seed(seed)

    return device


@functools.cache
def get_world_size() -> int:
    if _WORLD_SIZE is None:
        raise RuntimeError("model parallelism was not initialized")
    return _WORLD_SIZE


@functools.cache
def get_rank() -> int:
    if _WORLD_SIZE is None:
        raise RuntimeError("model parallelism was not initialized")
    return _LOCAL_RANK


def all_gather(x: torch.Tensor) -> torch.Tensor:
    """
    Gather a tensor of shape (n, m) into a tensor of shape (n, mp_size * m).
    """

    mp_size = get_world_size()
    if mp_size == 1:
        return x

    gather = [torch.empty_like(x) for _ in range(mp_size)]
    torch.distributed.all_gather(gather, x, group=_GROUP)
    return torch.cat(gather, dim=-1)


def all_reduce(x: torch.Tensor):
    if get_world_size() > 1:
        # reduce with a sum
        torch.distributed.all_reduce(x, group=_GROUP)
