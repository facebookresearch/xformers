# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict

import torch

from . import (
    __version__,
    _cpp_lib,
    _is_functorch_available,
    _is_opensource,
    _is_triton_available,
    ops,
)
from .ops.common import OPERATORS_REGISTRY


def get_features_status() -> Dict[str, str]:
    features = {}
    for op in OPERATORS_REGISTRY:
        features[f"{op.OPERATOR_CATEGORY}.{op.NAME}"] = op.info()
    for k, v in ops.swiglu_op._info().items():
        features[f"swiglu.{k}"] = v
    features["is_triton_available"] = str(_is_triton_available())
    features["is_functorch_available"] = str(_is_functorch_available)
    return features


def print_info():
    features = get_features_status()
    print(f"xFormers {__version__}")
    features["pytorch.version"] = torch.__version__
    if torch.cuda.is_available():
        features["pytorch.cuda"] = "available"
        device = torch.cuda.current_device()
        cap = torch.cuda.get_device_capability(device)
        features["gpu.compute_capability"] = ".".join(str(ver) for ver in cap)
        features["gpu.name"] = torch.cuda.get_device_name(device)
    else:
        features["pytorch.cuda"] = "not available"

    build_info = _cpp_lib._build_metadata
    if build_info is None and isinstance(
        _cpp_lib._cpp_library_load_exception, _cpp_lib.xFormersInvalidLibException
    ):
        build_info = _cpp_lib._cpp_library_load_exception.build_info
    if build_info is not None:
        features["build.info"] = "available"
        features["build.cuda_version"] = build_info.cuda_version
        features["build.python_version"] = build_info.python_version
        features["build.torch_version"] = build_info.torch_version
        for k, v in build_info.build_env.items():
            features[f"build.env.{k}"] = v
    else:
        features["build.info"] = "none"

    if _is_opensource:
        features["source.privacy"] = "open source"
    else:
        features["source.privacy"] = "fairinternal"

    for name, status in features.items():
        print("{:<50} {}".format(f"{name}:", status))


if __name__ == "__main__":
    print_info()
