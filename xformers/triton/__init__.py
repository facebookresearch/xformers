# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import torch

_triton_available = torch.cuda.is_available()
if _triton_available:
    try:
        from .fused_linear_layer import FusedLinear  # noqa
        from .layer_norm import FusedLayerNorm, layer_norm  # noqa
        from .softmax import log_softmax, softmax  # noqa

        __all__ = [
            "softmax",
            "log_softmax",
            "FusedLinear",
            "FusedLayerNorm",
            "layer_norm",
        ]
    except ImportError as e:
        import logging

        logging.warning(f"Triton components import fails with {e}")
        __all__ = []
