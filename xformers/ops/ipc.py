# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import concurrent.futures
import json
import multiprocessing.connection
from typing import Any, List, Optional, Union

import torch
import torch.distributed as dist
import torch.multiprocessing.reductions

# We could just send tensors directly on mp.Connections, since PyTorch installs
# the necessary reductions to make it work. However, in the receiving process,
# PyTorch "mounts" the tensor in the CUDA context for the GPU with the **SAME
# INDEX** as on the sender. This works if all processes use CUDA_VISIBLE_DEVICES
# to limit themselves to a single GPU (which thus has index 0 everywhere) but in
# all other cases it's a mess. Hence we use our own reductions (which wrap the
# ones from PyTorch) to use the right devices.


def _serialize_cuda_tensor(tensor: torch.Tensor):
    assert tensor.device.type == "cuda"
    func, args = torch.multiprocessing.reductions.reduce_tensor(tensor)
    assert func is torch.multiprocessing.reductions.rebuild_cuda_tensor
    assert args[6] == tensor.device.index
    return args


def _deserialize_cuda_tensor(args, device: torch.device) -> torch.Tensor:
    args = list(args)
    args[6] = device.index
    return torch.multiprocessing.reductions.rebuild_cuda_tensor(*args)


# We need all processes to exchange a few strings with their addresses (in order
# to be able to connect to each other). The solution for this kind of things in
# PyTorch is a Store (TCPStore or FileStore) but we cannot create one ourselves
# (we don't know which addr/port/file to use, since the default one is already
# being used by PyTorch's global store) nor can we extract one from the
# ProcessGroup (since there's no API to do so). We thus resort to using the PG
# itself to exchange data, which is overkill (we need to store the pickled data
# into tensors and send it to the GPU). On top of that, it introduces one more
# catch: it doesn't work in inference mode because of something about modifying
# tensors inplace. I couldn't find a way to temporarily disable inference mode
# (although it's supposed to be possible) however inference mode is thread-local
# so we can dodge it by offloading the collective call to another thread. I hate
# all this so much.


def _exchange_addresses(
    listeners: List[multiprocessing.connection.Listener],
    group: dist.ProcessGroup,
    device: torch.device,
) -> List[List[str]]:
    rank = group.rank()
    world_size = group.size()
    my_addresses: List[str] = []
    for listener in listeners:
        addr = listener.address
        # The address could be a tuple if the listener weren't a UNIX socket
        if isinstance(addr, bytes):
            # Shouldn't be bytes, according to docs and typeshed, but...
            # https://github.com/python/typeshed/issues/10054
            addr = addr.decode("utf-8")
        assert isinstance(addr, str)
        my_addresses.append(addr)
    if world_size == 1:
        return [my_addresses]
    # In fact, we can retrieve the store from the ProcessGroup, but only using
    # a private API. Hence we catch whatever exception and fall back in case.
    try:
        _, store = torch.distributed.distributed_c10d._world.pg_map.get(
            group, (None, None)
        )
        assert store is not None
        store.set(f"xformers_exchange_addresses_{rank}", json.dumps(my_addresses))
        all_addresses = [
            json.loads(store.get(f"xformers_exchange_addresses_{i}"))
            for i in range(world_size)
        ]
    except Exception:
        all_addresses = [[""] * (world_size - 1)] * world_size
        with concurrent.futures.ThreadPoolExecutor(
            initializer=torch.cuda.set_device, initargs=(device,)
        ) as e:
            e.submit(
                dist.all_gather_object,
                object_list=all_addresses,
                obj=my_addresses,
                group=group,
            ).result()
    return all_addresses


class IPCPipe:
    def __init__(self, connection, my_device) -> None:
        self.connection = connection
        self.my_device = my_device

    def send(self, tensor: torch.Tensor) -> None:
        assert self.connection is not None, "Sending to myself!"
        assert tensor.device == self.my_device, f"{tensor.device=} != {self.my_device=}"
        self.connection.send(_serialize_cuda_tensor(tensor))

    def recv(self) -> torch.Tensor:
        assert self.connection is not None, "Receiving from myself!"
        return _deserialize_cuda_tensor(self.connection.recv(), self.my_device)


def init_ipc(
    group: dist.ProcessGroup,
    device: Union[torch.device, str] = "cuda",
) -> List[Optional[IPCPipe]]:
    """
    Initializes pipes between processes of a `ProcessGroup`, that can be used
    to exchange `torch.Tensor` later
    """
    if isinstance(device, str):
        device = torch.device(device)
    if device.index is None:
        device = torch.device(device.type, index=torch.cuda.current_device())
    world_size = group.size()
    my_rank = group.rank()
    # Open connections to all other processes. We exchange addresses via
    # NCCL since we don't have access to a Store.
    listeners = [
        multiprocessing.connection.Listener(family="AF_UNIX", address="", backlog=1)
        for _ in range(world_size)
    ]
    # If any process is late, all other ones will block here
    all_addresses = _exchange_addresses(listeners, group, device)
    connections: Any = []
    for other_rank in range(world_size):
        # For p2p connection between ranks i<->j
        # if `i<j`, `i` listens, and `j` connects
        if my_rank < other_rank:  # `other` connects to me
            connections.append(listeners[other_rank].accept())
        elif other_rank == my_rank:
            connections.append(None)
        else:
            connections.append(
                multiprocessing.connection.Client(
                    family="AF_UNIX",
                    # Mypy wants it to be str, but it actually can also be bytes
                    # https://github.com/python/typeshed/issues/10054
                    address=all_addresses[other_rank][my_rank],
                )
            )
    return [
        IPCPipe(connection, my_device=device) if connection is not None else None
        for connection in connections
    ]
