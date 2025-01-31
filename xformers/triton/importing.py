# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util


def libdevice_find(name):
    """
    rsqrt = libdevice_find("rsqrt")

    is a triton-version-friendly way to say

    from triton.language.extra.libdevice import rsqrt
    """
    locs = (
        "triton.language.extra.libdevice",
        "triton.language.extra.cuda.libdevice",
        "triton.language.math",
        "triton.language.libdevice",
    )

    for loc in locs:
        if spec := importlib.util.find_spec(loc):
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            return getattr(module, name)

    raise ImportError(f"Could not find a library to import {name}")
