# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from enum import Enum
from typing import Union

import torch.nn as nn

from xformers import _is_triton_available

if _is_triton_available:
    from xformers.triton.layer_norm import FusedLayerNorm


class LayerNormStyle(str, Enum):
    """Support different layer norm styles.
    See "On Layer Normalization in the Transformer Architecture",
    Xiong et al., https://arxiv.org/pdf/2002.04745v1.pdf
    """

    Pre = "pre"
    Post = "post"


# CREDITS: the following is inspired by FastAI's Transformer implementation
class Residual(nn.Module):
    """Object-oriented handling of the residual path

    .. warning: by convention, if multiple tensors are being passed in,
        the first one is used for the residual path
    """

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer

    def forward(
        self,
        *args,
        **kwargs,
    ):
        residual = args[0]
        return residual + self.layer(*args, **kwargs)


class PreNorm(nn.Module):
    """Adds LayerNorm before computing attention

    ..Note: If a list of inputs is passed, all of them get normalized"""

    def __init__(self, d_model: int, sublayer: nn.Module, use_triton: bool = True):
        super().__init__()
        if _is_triton_available and use_triton:
            self.norm: Union[nn.LayerNorm, FusedLayerNorm] = FusedLayerNorm(d_model)
        else:
            self.norm = nn.LayerNorm(d_model)

        self.sublayer = sublayer

    def forward(self, *args, **kwargs):
        # Could be that the same tensor has been passed multiple times
        # in that case we'll just normalize once
        list_ids = [id(inp) for inp in args]
        if list_ids.count(list_ids[0]) == len(list_ids):
            normalized_input = self.norm(args[0])
            sublayer_inputs = [normalized_input for _ in args]
        else:
            sublayer_inputs = [self.norm(x_) for x_ in args]
        return self.sublayer(*sublayer_inputs, **kwargs)


class PostNorm(nn.Module):
    """Adds LayerNorm after computing attention"""

    def __init__(self, d_model: int, sublayer: nn.Module, use_triton: bool = True):
        super().__init__()
        if _is_triton_available and use_triton:
            self.norm: Union[nn.LayerNorm, FusedLayerNorm] = FusedLayerNorm(d_model)
        else:
            self.norm = nn.LayerNorm(d_model)

        self.sublayer = sublayer

    def forward(self, *args, **kwargs):
        x = self.sublayer(*args, **kwargs)
        return self.norm(x)
