# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from xformers.components import Activation, LayerNormStyle
from xformers.components.nvfuser import (
    NVFusedBiasActivationDropout,
    NVFusedBiasDropoutRes,
    NVFusedBiasDropoutResLayerNorm,
)


def build_nvfused(
    fused_pattern: nn.Module,
    shape: tuple,
    bias: bool,
    activation: Activation,
    p: float,
    layer_norm_style: LayerNormStyle,
):
    bias_shape = shape[-1] if bias else None
    d_model = shape[-1]
    init_args = {
        NVFusedBiasActivationDropout: [p, activation, bias_shape],
        NVFusedBiasDropoutRes: [p, bias_shape],
        NVFusedBiasDropoutResLayerNorm: [p, d_model, bias_shape, layer_norm_style],
    }
    return fused_pattern(*init_args[fused_pattern])
