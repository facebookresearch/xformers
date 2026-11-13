# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import pkgutil

import pytest

import xformers.ops as xops

if xops._HAS_MSLK:
    import xformers.ops.fmha

    _REEXPORTING_MODULES = [
        m.name
        for m in pkgutil.walk_packages(xformers.ops.fmha.__path__, "xformers.ops.fmha.")
    ] + ["xformers.ops.tree_attention", "xformers.attn_bias_utils"]
else:
    _REEXPORTING_MODULES = []


@pytest.mark.parametrize("module", _REEXPORTING_MODULES)
def test_reexports_exist_in_mslk(module: str) -> None:
    # These modules re-export names from mslk: catch them going missing there
    importlib.import_module(module)
