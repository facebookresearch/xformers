# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import List

import torch
import torch.nn as nn
from torch.autograd.function import Function
from torch.utils.checkpoint import get_device_states, set_device_states

from xformers.components import RequiresWrappedInputs

# CREDITS: Code adapted from
# https://github.com/lucidrains/reformer-pytorch/blob/master/reformer_pytorch/reversible.py
# https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py,
# https://pytorch.org/docs/stable/_modules/torch/utils/checkpoint.html


# pyre-fixme[13]: `cpu_state` is not initialized in the constructor.
class Deterministic(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.cpu_state: torch.Tensor = torch.get_rng_state()
        self.cuda_in_fwd: bool = False
        self.gpu_devices: List[int] = []
        self.gpu_states: List[torch.Tensor] = []
        self.wrap_inputs = isinstance(net, RequiresWrappedInputs)

    def record_rng(self, *args):
        self.cpu_state = torch.get_rng_state()
        if torch.cuda._initialized:
            self.cuda_in_fwd = True
            self.gpu_devices, self.gpu_states = get_device_states(*args)

    def forward(self, *args, record_rng: bool = False, set_rng: bool = False, **kwargs):
        if record_rng:
            self.record_rng(*args)

        if not set_rng:
            # Normal FW run
            if self.wrap_inputs:
                return self.net(inputs=args, **kwargs)
            else:
                return self.net(*args, **kwargs)

        else:  # pragma: no cover  # this is called in the backward pass, not picked up
            # This is analogous to checkpointing, reset the original random state
            rng_devices: List[int] = []
            if self.cuda_in_fwd:
                rng_devices = self.gpu_devices

            with torch.random.fork_rng(devices=rng_devices, enabled=True):
                torch.set_rng_state(self.cpu_state)
                if self.cuda_in_fwd:
                    set_device_states(self.gpu_devices, self.gpu_states)

                if self.wrap_inputs:
                    return self.net(inputs=args, **kwargs)
                else:
                    return self.net(*args, **kwargs)


class ReversibleBlock(nn.Module):
    def __init__(self, f: nn.Module, g: nn.Module, split_dim: int = -1):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)
        self.split_dim = split_dim

    def forward(self, x: torch.Tensor, f_args={}, g_args={}):
        x1, x2 = torch.chunk(x, 2, dim=-1)
        y1, y2 = None, None

        with torch.no_grad():
            y1 = x1 + self.f(x2, record_rng=self.training, **f_args)
            y2 = x2 + self.g(y1, record_rng=self.training, **g_args)

        return torch.cat([y1, y2], dim=self.split_dim)

    def backward_pass(
        self, y: torch.Tensor, dy: torch.Tensor, f_args={}, g_args={}
    ):  # pragma: no cover  # this is covered, but called directly from C++
        y1, y2 = torch.chunk(y, 2, dim=self.split_dim)
        del y

        dy1, dy2 = torch.chunk(dy, 2, dim=self.split_dim)
        del dy

        with torch.enable_grad():
            y1.requires_grad = True
            gy1 = self.g(y1, set_rng=True, **g_args)
            torch.autograd.backward(gy1, dy2)

        with torch.no_grad():
            x2 = y2 - gy1
            del y2, gy1

            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None

        with torch.enable_grad():
            x2.requires_grad = True
            fx2 = self.f(x2, set_rng=True, **f_args)
            torch.autograd.backward(fx2, dx1)

        with torch.no_grad():
            x1 = y1 - fx2
            del y1, fx2

            dx2 = dy2 + x2.grad
            del dy2
            x2.grad = None

            x = torch.cat([x1, x2.detach()], dim=self.split_dim)
            dx = torch.cat([dx1, dx2], dim=self.split_dim)

        return x, dx


class _ReversibleFunction(Function):
    @staticmethod
    def forward(ctx, x, blocks, kwargs):
        ctx.kwargs = kwargs
        for block in blocks:
            x = block(x, **kwargs)
        ctx.y = x.detach()
        ctx.blocks = blocks
        return x

    @staticmethod
    def backward(
        ctx, dy
    ):  # pragma: no cover # this is covered, but called directly from C++
        y = ctx.y
        kwargs = ctx.kwargs
        for block in ctx.blocks[::-1]:
            y, dy = block.backward_pass(y, dy, **kwargs)
        return dy, None, None


class ReversibleSequence(nn.Module):
    def __init__(self, blocks: nn.ModuleList):
        super().__init__()

        # pyre-fixme[23]: Unable to unpack `torch.nn.Module` into 2 values.
        self.blocks = nn.ModuleList([ReversibleBlock(f, g) for f, g in blocks])

    def forward(self, x, arg_route=(True, False), **kwargs):
        f_args, g_args = map(lambda route: kwargs if route else {}, arg_route)
        block_kwargs = {"f_args": f_args, "g_args": g_args}

        return _ReversibleFunction.apply(x, self.blocks, block_kwargs)
