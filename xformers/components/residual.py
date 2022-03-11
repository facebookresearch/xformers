# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from enum import Enum
from typing import List, Union

import torch
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
    """
    Object-oriented handling of the residual path

    .. Note: the wrapped layers must accept all the inputs as a single list
    """

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer

        # PreNorm and PostNorm require all the tensors to be passed as a list
        self.wrap_inputs = isinstance(layer, PreNorm) or isinstance(layer, PostNorm)

    def forward(self, inputs: List[torch.Tensor], **kwargs):
        if self.wrap_inputs:
            return inputs[0] + self.layer(inputs=inputs, **kwargs)

        else:
            return inputs[0] + self.layer(*inputs, **kwargs)


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
        self.wrap_inputs = isinstance(sublayer, PostNorm) or isinstance(
            sublayer, Residual
        )

    def forward(self, inputs: List[torch.Tensor], **kwargs):
        assert len(inputs) > 0

        # Perf improvement: if the inputs are all the same, only norm once
        ids = [id(x) for x in inputs]
        if ids.count(ids[0]) == len(ids):
            # The same tensor is passed multiple times
            x_norm = self.norm(inputs[0])
            inputs_normed = [x_norm for _ in inputs]
        else:
            # The inputs differ, norm them all
            inputs_normed = [self.norm(x_) for x_ in inputs]

        if self.wrap_inputs:
            return self.sublayer(inputs=inputs_normed, **kwargs)
        else:
            return self.sublayer(*inputs_normed, **kwargs)


class PostNorm(nn.Module):
    """Adds LayerNorm after computing attention"""

    def __init__(self, d_model: int, sublayer: nn.Module, use_triton: bool = True):
        super().__init__()
        if _is_triton_available and use_triton:
            self.norm: Union[nn.LayerNorm, FusedLayerNorm] = FusedLayerNorm(d_model)
        else:
            self.norm = nn.LayerNorm(d_model)

        self.sublayer = sublayer
        self.wrap_inputs = isinstance(sublayer, PreNorm) or isinstance(
            sublayer, Residual
        )

    def forward(self, inputs: List[torch.Tensor], **kwargs):
        if self.wrap_inputs:
            x = self.sublayer(inputs=inputs, **kwargs)
        else:
            x = self.sublayer(*inputs, **kwargs)
        return self.norm(x)
