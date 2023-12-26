# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import weakref
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, ContextManager, Dict, List

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map
from torch.utils.checkpoint import get_device_states, set_device_states


def _detach(x):
    if isinstance(x, torch.Tensor):
        return x.detach()
    return x


class CachingTorchDispatchMode(TorchDispatchMode):
    def __init__(self, policy_fn, storage):
        self.policy_fn = policy_fn
        self.storage = storage

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if self.policy_fn(func, *args, **kwargs):
            out = func(*args, **kwargs)
            out_detached = tree_map(_detach, out)
            self.storage[func].append(out_detached)
            return out
        return func(*args, **kwargs)


class CachedTorchDispatchMode(TorchDispatchMode):
    def __init__(self, policy_fn, storage):
        self.policy_fn = policy_fn
        self.storage = storage

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if self.policy_fn(func, *args, **kwargs):
            # this is a basic guard if there are additional ops
            # that were present in the second forward. An example
            # is for ops like `detach`, which can be added if our
            # policy is too loose
            if self.storage[func]:
                out = self.storage[func].pop(0)
                return out
        return func(*args, **kwargs)


def _get_default_policy(allow_list=None):
    _default_allow_list = [
        "xformers.efficient_attention_forward_cutlass.default",
        "xformers_flash.flash_fwd.default",
        "aten.addmm.default",
        "aten.mm.default",
    ]
    if allow_list is None:
        allow_list = _default_allow_list

    def _default_policy(func, *args, **kwargs):
        return str(func) in allow_list

    return _default_policy


class VerboseTorchDispatchMode(TorchDispatchMode):
    def __init__(self):
        self.operators = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        self.operators.append(func)
        return func(*args, **kwargs)


def list_operators(function, *args, **kwargs):
    """
    Returns the list of operators used inside `function` with
    *args and **kwargs
    """
    verbose_mode = VerboseTorchDispatchMode()
    with verbose_mode:
        function(*args, **kwargs)
    return verbose_mode.operators


def checkpoint(
    function, *args, preserve_rng_state=True, policy_fn=None, **kwargs
) -> Any:
    """Checkpointining with custom policy function for selectively deciding
    what to store and what to recompute
    Args:
        function: describes what to run in the forward pass of the model or
            part of the model. It should also know how to handle the inputs
            passed as the tuple. For example, in LSTM, if user passes
            ``(activation, hidden)``, :attr:`function` should correctly use the
            first input as ``activation`` and the second input as ``hidden``
        preserve_rng_state(bool, optional):  Omit stashing and restoring
            the RNG state during each checkpoint.
            Default: ``True``
        policy_fn(Union[List[Op], callable]): policy for deciding what to
            store (instead of recompute). If it's a function, it should
            be of form (func, *args, **kwargs) -> bool which indicates
            if func outputs with *args and **kwargs should be stored or not.
            Additionally, a list[Op] is also supported for easier cases.
            The op should be in the format `torch.ops.***`, where the `***`
            names of operators can be obtained with `list_operators`.
        *args: Arguments to pass in to the given ``function``.
        **kwargs: Keyword arguments to pass into the given ``function``.
    """

    # Requires PyTorch 1.13 at least
    from torch.utils.checkpoint import _get_autocast_kwargs

    # Accommodates the (remote) possibility that autocast is enabled for cpu AND gpu.
    gpu_autocast_kwargs, cpu_autocast_kwargs = _get_autocast_kwargs()

    if preserve_rng_state:
        fwd_cpu_state = torch.get_rng_state()
        # Don't eagerly initialize the cuda context by accident.
        # (If the user intends that the context is initialized later, within their
        # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
        # we have no way to anticipate this will happen before we run the function.
        # If they do so, we raise an error.)
        had_cuda_in_fwd = False
        if torch.cuda._initialized:
            had_cuda_in_fwd = True
            fwd_gpu_devices, fwd_gpu_states = get_device_states(*args)

    # Custom class to be able to take weak references
    class Holder:
        pass

    # The Holder object for each of the saved object is saved directly on the
    # SavedVariable and is cleared when reset_data() is called on it. We MUST make
    # sure that this is the only object having an owning reference to ensure that
    # the Tensor stored in storage is deleted as soon as the corresponding SavedVariable
    # data is cleared.
    storage: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
    weak_holder_list = []

    if policy_fn is None:
        policy_fn = _get_default_policy()
    elif isinstance(policy_fn, list):
        policy_fn = _get_default_policy(policy_fn)
    else:
        assert callable(policy_fn), "policy_fn should be None, list or a callable"

    temp_storage: Dict[Any, List[Any]] = defaultdict(list)
    # assumption: grad_mode doesn't change inside function
    caching_mode: ContextManager[None]
    if torch.is_grad_enabled():
        caching_mode = CachingTorchDispatchMode(policy_fn, temp_storage)
    else:
        caching_mode = nullcontext()
    cached_mode = CachedTorchDispatchMode(policy_fn, temp_storage)

    def pack(x):
        # TODO(varal7): Instead of returning abstract object, we can return things metadata (such as
        # size, device, ...) to catch certain cases of undeterministic behavior of the forward
        res = Holder()
        weak_holder_list.append(weakref.ref(res))
        return res

    def unpack(x):
        unpack_counter = 0
        if len(storage) == 0:

            def inner_pack(inner):
                nonlocal unpack_counter
                unpack_counter += 1
                # If the holder went out of scope, the SavedVariable is dead and so
                # the value will never be read from the storage. Skip filling it.
                if weak_holder_list[unpack_counter - 1]() is None:
                    return
                # Use detach here to ensure we don't keep the temporary autograd
                # graph created during the second forward
                storage[weak_holder_list[unpack_counter - 1]()] = inner.detach()
                return

            def inner_unpack(packed):
                raise RuntimeError(
                    "You are calling backwards on a tensor that is never exposed. Please open an issue."
                )

            # Stash the surrounding rng state, and mimic the state that was
            # present at this time during forward.  Restore the surrounding state
            # when we're done.
            rng_devices = []
            if preserve_rng_state and had_cuda_in_fwd:
                rng_devices = fwd_gpu_devices
            with torch.random.fork_rng(devices=rng_devices, enabled=preserve_rng_state):
                if preserve_rng_state:
                    torch.set_rng_state(fwd_cpu_state)
                    if had_cuda_in_fwd:
                        set_device_states(fwd_gpu_devices, fwd_gpu_states)

                with cached_mode, torch.enable_grad(), torch.cuda.amp.autocast(
                    **gpu_autocast_kwargs
                ), torch.cpu.amp.autocast(
                    **cpu_autocast_kwargs
                ), torch.autograd.graph.saved_tensors_hooks(
                    inner_pack, inner_unpack
                ):
                    _unused = function(*args, **kwargs)  # noqa: F841

        if x not in storage:
            raise RuntimeError(
                "Attempt to retrieve a tensor saved by autograd multiple times without checkpoint"
                " recomputation being triggered in between, this is not currently supported. Please"
                " open an issue with details on your use case so that we can prioritize adding this."
            )

        return storage[x]

    with caching_mode, torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        output = function(*args, **kwargs)
        if torch.cuda._initialized and preserve_rng_state and not had_cuda_in_fwd:
            # Cuda was not initialized before running the forward, so we didn't
            # stash the CUDA state.
            raise RuntimeError(
                "PyTorch's CUDA state was initialized in the forward pass "
                "of a Checkpoint, which is not allowed. Please open an issue "
                "if you need this feature."
            )

    return output
