# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import functools
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import astuple, dataclass
from typing import Any, Callable, ContextManager, Dict, List, Optional, Tuple

import torch
from torch.testing._internal.composite_compliance import (
    is_inplace,
    is_inplace_view_fn,
    is_view_fn,
)
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map

_scipy_is_available = False
try:
    from scipy.optimize import Bounds, LinearConstraint, milp

    _scipy_is_available = True
except ImportError:
    _scipy_is_available = False

try:
    # let's keep BC for older PyTorch for now
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        ActivationWrapper,
    )
    from torch.utils.checkpoint import (  # type: ignore
        _CachedTorchDispatchMode,
        _CachingTorchDispatchMode,
        _ignored_ops,
    )
except ImportError:
    ActivationWrapper = torch.nn.Module  # type: ignore

    class _NotAvailable:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Need PyTorch >= 2.2")

    _CachedTorchDispatchMode = _NotAvailable  # type: ignore
    _CachingTorchDispatchMode = _NotAvailable  # type: ignore
    _ignored_ops = set()  # type: ignore


_additional_ignored_ops = {
    torch.ops.aten.lift_fresh.default,
    torch.ops.profiler._record_function_exit._RecordFunction,
    torch.ops.aten.clone.default,  # seems needed for torch.compile
}
OPS_TO_ALWAYS_SKIP = _ignored_ops | _additional_ignored_ops


@dataclass
class ProfileMetadata:
    name: str
    time_taken: float
    memory_used: float
    curr_idx: int
    output_ids: Any
    inplace_info: Tuple[int, int]
    is_view_like: bool
    is_rand_op: bool


def _detach(x):
    if isinstance(x, torch.Tensor):
        return x.detach()
    return x


def _get_default_policy(allow_list=None):
    _default_allow_list = [
        "xformers.efficient_attention_forward_cutlass.default",
        "xformers_flash.flash_fwd.default",
        "aten.addmm.default",
        "aten.mm.default",
    ]
    if allow_list is None:
        allow_list = _default_allow_list

    def _default_policy(mode, func, *args, **kwargs):
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


class CachedTorchDispatchMode(_CachedTorchDispatchMode):
    def pop_from_storage(self, func, args, kwargs):
        # the autograd engine might add spurious views. This is a basic
        # guard and should be improved
        if self.storage[func]:
            return self.storage[func].pop(0)
        return func(*args, **kwargs)


class NullTorchDispatchMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)


def selective_checkpoint_context_fn(policy_fn=None):
    """An activation checkpoint context_fn for selectively deciding what to
    store and what to recompute. Accepts a custom policy.
    Args:
        policy_fn(Union[List[Op], callable]): policy for deciding what to
            store (instead of recompute). If it's a function, it should
            be of form (func, *args, **kwargs) -> bool which indicates
            if func outputs with *args and **kwargs should be stored or not.
            Additionally, a list[Op] is also supported for easier cases.
            The op should be in the format `torch.ops.***`, where the `***`
            names of operators can be obtained with `list_operators`.
    """
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
        caching_mode = _CachingTorchDispatchMode(deepcopy(policy_fn), temp_storage)
    else:
        caching_mode = NullTorchDispatchMode()
    cached_mode = CachedTorchDispatchMode(deepcopy(policy_fn), temp_storage)

    return caching_mode, cached_mode


def checkpoint(
    function, *args, preserve_rng_state=True, policy_fn=None, **kwargs
) -> Any:
    """Wrapper around torch.utils.checkpoint that accepts a custom policy
    function for selectively deciding what to store and what to recompute
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
    return torch.utils.checkpoint.checkpoint(
        function,
        *args,
        use_reentrant=False,
        preserve_rng_state=preserve_rng_state,
        context_fn=functools.partial(selective_checkpoint_context_fn, policy_fn),
        **kwargs,
    )


class ProfileOperatorsTorchDispatchMode(TorchDispatchMode):
    def __init__(self, num_runs: int = 10) -> None:
        self.data: List[ProfileMetadata] = []
        self.num_runs: int = num_runs

    def _get_inplace_metadata(self, func, out) -> Tuple[int, int, Tuple[int, ...]]:
        curr_idx = len(self.data)

        def get_tensor_id(e):
            return (
                e.untyped_storage().data_ptr() if isinstance(e, torch.Tensor) else None
            )

        output_ids = tree_map(get_tensor_id, out)
        if not is_inplace(func):
            return curr_idx, output_ids, ()

        op_id = curr_idx
        op_parent_id = -1
        for i, d in enumerate(self.data):
            # find the first occurence of a tensor that
            # shares the same storage as the current tensor
            past_output_ids = d.output_ids
            past_output_ids = (
                [past_output_ids]
                if not isinstance(past_output_ids, (list, tuple, dict))
                else past_output_ids
            )
            if output_ids in past_output_ids:
                op_parent_id = i
                break
        if op_parent_id < 0:
            op_parent_id = op_id
        inplace_info = (op_id, op_parent_id)
        return curr_idx, output_ids, inplace_info

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        out = func(*args, **kwargs)

        curr_idx, output_ids, inplace_info = self._get_inplace_metadata(func, out)
        is_view_like = is_view_fn(func) or is_inplace_view_fn(func)
        is_rand_op = torch.Tag.nondeterministic_seeded in func.tags
        # sdpa has non-deterministic seed, but might be deterministic
        # if no dropout is applied
        if func.overloadpacket.__name__ == "_scaled_dot_product_flash_attention":
            is_rand_op = kwargs.get("dropout_p", 0) != 0

        # get runtime info of func
        torch.cuda.synchronize()
        t = time.time()
        for i in range(self.num_runs):
            func(*args, **kwargs)
        torch.cuda.synchronize()
        time_taken = (time.time() - t) / self.num_runs

        # get memory usage of func
        torch.cuda.reset_peak_memory_stats()
        mem1 = torch.cuda.max_memory_allocated() / 2**20
        func(*args, **kwargs)
        mem2 = torch.cuda.max_memory_allocated() / 2**20

        self.data.append(
            ProfileMetadata(
                func,
                time_taken,
                mem2 - mem1,
                curr_idx,
                output_ids,
                inplace_info,
                is_view_like,
                is_rand_op,
            )
        )
        return out


def _analyze_operators(function, *args) -> List[ProfileMetadata]:
    """
    Use ProfileOperatorsTorchDispatchMode to get runtime and memory info.

    Args:
        function: The function to optimize which will be selectively checkpointed. Usually the forward pass
            of the model.
        *args: Arguments to pass in to the given ``function``.

    Returns:
        A list of tuples, where each tuples contains the name of the operator, the runtime of the operator,
            and the memory usage of the operator.

    """
    profile_ops = ProfileOperatorsTorchDispatchMode()
    with profile_ops:
        function(*args)

    data = profile_ops.data
    return data


def get_optimal_checkpoint_policy(function, *args, memory_budget: float) -> Callable:
    """
    Given a function, its arguments, and the maximum amount of memory available,
    find the subset of operators that can be optimized to reduce runtime while still fitting within the memory budget.

    Args:
        function: The function to optimize which will be selectively checkpointed. Usually the forward pass
            of the model.
        *args: Arguments to pass in to the given ``function``.
        memory_budget (float): A float between 0 and 1 which describes what percentage of the total memory to use.

    Returns:
        A callable policy which can be passed to xformers.checkpoint()

    Raises:
        RuntimeError: If `scipy` is not available.
        ValueError: If `memory_budget` is not a float between 0 and 1.

    """
    if not _scipy_is_available:
        raise RuntimeError(
            "Please install scipy 1.9.0+ to use `get_optimal_checkpoint_policy`. You can do so using "
            "`pip install scipy`."
        )
    if memory_budget < 0 or memory_budget > 1:
        raise ValueError(
            f"`memory_budget` must be a float between 0 and 1. Got {memory_budget}."
        )

    data = _analyze_operators(function, *args)
    # remove aten.detach.default from the list of ops because autograd
    # inserts those during backward and it breaks the fwd-bwd alignment
    data = [x for x in data if x.name not in OPS_TO_ALWAYS_SKIP]

    ops, runtimes_, memory_, new_ids, _, inplace_ops_, view_like_ops_, rand_ops_ = zip(
        *[astuple(x) for x in data]
    )
    runtimes = torch.tensor(runtimes_, dtype=torch.float64)
    memory = torch.tensor(memory_, dtype=torch.float64)
    view_like_ops = [i for i, x in enumerate(view_like_ops_) if x]
    rand_ops = [i for i, x in enumerate(rand_ops_) if x]

    # remap the inplace indices as we have removed OPS_TO_ALWAYS_SKIP
    inplace_ops = [tuple(map(new_ids.index, x)) for x in inplace_ops_ if x]

    # the last operation is always stored as the output of the checkpoint
    # block, so we can avoid recomputing it. We set the memory to zero
    # instead of adding a new constraint because we want both the 0 and 1
    # endpoints for memory_budget to be valid
    # FIXME: this heuristic for finding the last non-view non-inplace op
    # might not always be correct, which would yield suboptimal policies
    last_op = len(ops) - 1
    skip_ops_ = set(view_like_ops) | set([x[0] for x in inplace_ops])
    skip_ops = sorted(list(skip_ops_))
    for op in reversed(skip_ops):
        if op == last_op:
            last_op -= 1

    memory[last_op] = 0

    max_memory = memory_budget * memory.sum().item()

    # workaround to fix https://github.com/pytorch/pytorch/issues/121212
    force_store_random = all([not isinstance(x, torch.Tensor) for x in args])

    optim_output = _optimize_runtime_with_given_memory(
        memory=memory,
        runtimes=runtimes,
        max_memory=max_memory,
        view_like_ops=view_like_ops,
        inplace_ops=inplace_ops,
        random_ops=rand_ops,
        force_store_random=force_store_random,
    )
    return _OptimalPolicy(optim_output=optim_output)


def _optimize_runtime_with_given_memory(
    memory: torch.Tensor,
    runtimes: torch.Tensor,
    max_memory: float,
    view_like_ops: List[int],
    inplace_ops: List[Tuple[int, ...]],
    random_ops: List[int],
    force_store_random: bool,
) -> torch.Tensor:
    """
    Given a list of operator names, their corresponding runtimes, and the maximum amount of memory available,
    find the subset of operators that can be optimized to reduce runtime while still fitting within the memory budget.
    Uses https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html

    Args:
        memory (torch.Tensor): Tensor containing the memory usage of each operator.
        runtimes (torch.Tensor): Tensor containing the runtime of each operator.
        max_memory (float): Maximum amount of memory to use.
        view_like_ops ([List[int]): Indices of the view-like ops.
        inplace_ops (List[Tuple[int, int]]): Tuple with the pair of inplace op -> parent of inplace op.
            This will be used to add the constraint that in-place ops need to either be
            stored in memory with the previous op, or recomputed with the previous op.
        random_ops ([List[int]): Indices of the random ops, which will always be recomputed.
        force_store_random (bool): force random ops to always be stored (instead of recomputed)
    """
    c = -runtimes  # type: ignore[operator]

    memory_constraint = LinearConstraint(A=memory, ub=max_memory)
    constraints = [memory_constraint]

    # view-like ops should always be recomputed
    for i in view_like_ops:
        A = torch.zeros_like(c)
        A[i] = 1
        constraints.append(LinearConstraint(A=A, lb=0, ub=0))

    # inplace ops should always be done in conjuction with its parent op
    # i.e., if we recompute the parent op the inplace should also be
    # recomputed, and vice versa
    for op, op_parent in inplace_ops:
        A = torch.zeros_like(c)
        if op != op_parent:
            A[op_parent] = 1
            A[op] = -1
            constraints.append(LinearConstraint(A=A, lb=0, ub=0))
        else:
            # if op == op_parent, it's because it's the first op
            # that is inplace. Thus never recompute it
            A[op] = 1
            constraints.append(LinearConstraint(A=A, lb=1, ub=1))

    # ideally, always recompute random ops
    # in practice, due to a bug in https://github.com/pytorch/pytorch/issues/121212
    # sometimes we need to store them to avoid correctness issues
    for i in random_ops:
        A = torch.zeros_like(c)
        A[i] = 1
        val = int(force_store_random)
        constraints.append(LinearConstraint(A=A, lb=val, ub=val))

    integrality = torch.ones_like(c)
    res = milp(
        c=c, constraints=constraints, integrality=integrality, bounds=Bounds(0, 1)
    )
    if not res.success:
        raise ValueError(
            "The problem is infeasible, and probably due to a change in xformers "
            "that makes random ops always be stored. Try passing a larger memory_budget. "
            "This will be fixed once https://github.com/pytorch/pytorch/issues/121212 "
            "is solved"
        )
    x = torch.from_numpy(res.x)
    return x


class _OptimalPolicy:
    def __init__(self, optim_output: torch.Tensor):
        self.counter = 0
        self.optim_output = optim_output.tolist()

    def __call__(self, mode, func, *args, **kwargs) -> bool:
        # returning False means recompute, True means store in memory
        if func in OPS_TO_ALWAYS_SKIP:
            return False
        count = self.counter
        self.counter += 1
        return self.optim_output[count] == 1


class SelectiveCheckpointWrapper(ActivationWrapper):
    def __init__(self, mod, memory_budget=None, policy_fn=None):
        if torch.__version__ < (2, 1):
            raise RuntimeError(
                "SelectiveCheckpointWrapper only supported for torch >- 2.1"
            )
        super().__init__(mod)
        if not ((memory_budget is None) ^ (policy_fn is None)):
            raise ValueError("Need to specify either policy_fn or memory_budget")
        self.memory_budget = memory_budget
        self.policy_fn = policy_fn

        # TODO: this should be enabled by default in PyTorch
        torch._dynamo.config._experimental_support_context_fn_in_torch_utils_checkpoint = (
            True
        )

    @torch.compiler.disable
    def _get_policy_fn(self, *args, **kwargs):
        if not torch.is_grad_enabled():
            # no need to compute a policy as it won't be used
            return []
        # if policy is not specified, initialize policy for a given memory budget
        with torch.random.fork_rng():
            policy_fn = get_optimal_checkpoint_policy(
                self._checkpoint_wrapped_module,
                *args,
                **kwargs,
                memory_budget=self.memory_budget,
            )
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            # use the same policy across different GPUs
            objects = [policy_fn]
            torch.distributed.broadcast_object_list(objects, src=0)
            policy_fn = objects[0]
        return policy_fn

    def get_policy_fn(self, *args, **kwargs):
        if self.policy_fn is None:
            self.policy_fn = self._get_policy_fn(*args, **kwargs)
        return self.policy_fn

    def forward(self, *args, **kwargs):
        policy_fn = self.get_policy_fn(*args, **kwargs)
        return checkpoint(
            self._checkpoint_wrapped_module, *args, **kwargs, policy_fn=policy_fn
        )


def selective_checkpoint_wrapper(
    module: torch.nn.Module,
    memory_budget: Optional[float] = None,
    policy_fn: Optional[Callable] = None,
):
    """
    Wrap a module with selective activation checkpointing.

    It behaves similarly to PyTorch's checkpoint_wrapper, but gives the possibility
    to the user to either specify a handcrafted policy_fn, or to let an optimization
    algorithm to select the policy given a user-specified memory_budget.

    The user should either specify the memory_budget argument or the policy_fn.

    The memory_budget is a float value between 0 (recompute everything in the backward) or 1
    (store everything for backward). Using a value of 0 should be similar to PyTorch's
    activation checkpoint, while 1 should be similar to the behavior of not using any
    activation checkpointing.
    """
    return SelectiveCheckpointWrapper(module, memory_budget, policy_fn)
