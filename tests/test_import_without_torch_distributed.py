# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import os
import pathlib
import subprocess
import sys

import pytest

import xformers

# Exit code the child process uses when it cannot set up the simulation.
_CANNOT_SIMULATE = 77

# Run in a child process so that we can pretend that PyTorch was built without
# distributed support before xFormers gets imported for the first time. This is
# the situation on devices such as the NVIDIA Jetson, whose official PyTorch
# containers ship a build where torch.distributed.is_available() is False.
_SIMULATE_AND_IMPORT = """
import importlib
import sys

import torch
import torch.distributed


class _Stub:
    pass


# When torch.distributed.is_available() is False, torch/distributed/__init__.py
# only defines a couple of stubs, and every submodule wrapping the C++ bindings
# is unimportable. Reproduce both of these.
_MISSING_ATTRS = [
    "Work",
    "ReduceOp",
    "all_gather_into_tensor",
    "all_reduce",
    "broadcast_object_list",
    "get_world_size",
    "init_process_group",
    "is_initialized",
    "new_group",
    "reduce_scatter_tensor",
    "_symmetric_memory",
    "distributed_c10d",
]
_MISSING_MODULES = [
    "torch._C._distributed_c10d",
    "torch.distributed._symmetric_memory",
    "torch.distributed.distributed_c10d",
]

try:
    torch.distributed.is_available = lambda: False
    for _attr in _MISSING_ATTRS:
        if hasattr(torch.distributed, _attr):
            delattr(torch.distributed, _attr)
    torch.distributed.GroupName = _Stub
    torch.distributed.ProcessGroup = _Stub
    for _name in _MISSING_MODULES:
        # A None entry in sys.modules makes any later import of that name fail.
        sys.modules[_name] = None
    assert not torch.distributed.is_available()
except Exception as exc:
    print("cannot simulate a PyTorch without distributed support: %r" % (exc,))
    sys.exit(77)

importlib.import_module(sys.argv[1])
print("imported %s" % (sys.argv[1],))
"""


@pytest.mark.parametrize(
    "module",
    ["xformers", "xformers.ops", "xformers.ops.differentiable_collectives"],
)
def test_import_without_torch_distributed(tmp_path: pathlib.Path, module: str) -> None:
    script = tmp_path / "simulate_and_import.py"
    script.write_text(_SIMULATE_AND_IMPORT)

    # Make sure the child picks up the same xFormers as the parent, whether it
    # is installed or used straight from a source checkout.
    xformers_parent = str(pathlib.Path(xformers.__file__).resolve().parent.parent)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [xformers_parent] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )

    proc = subprocess.run(
        [sys.executable, str(script), module],
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode == _CANNOT_SIMULATE:
        pytest.skip(proc.stdout.strip())
    assert proc.returncode == 0, (
        f"`import {module}` failed on a PyTorch without distributed support\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
