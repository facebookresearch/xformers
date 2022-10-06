# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict, List, Type

import torch

from . import __version__, _is_functorch_available, _is_triton_available, ops


def get_features_status() -> Dict[str, str]:
    ALL_OPS: List[Type[ops.AttentionOpBase]] = [
        ops.MemoryEfficientAttentionFlashAttentionOp,
        ops.MemoryEfficientAttentionCutlassOp,
        ops.MemoryEfficientAttentionOp,
    ]
    features = {}
    for op in ALL_OPS:
        features[f"memory_efficient_attention.{op.NAME}"] = op.info()
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
    for name, status in features.items():
        print("{:<40} {}".format(f"{name}:", status))


if __name__ == "__main__":
    print_info()
