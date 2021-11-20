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


def _to_tensor_list(
    inputs: Union[torch.Tensor, List[torch.Tensor]]
) -> List[torch.Tensor]:
    if not isinstance(inputs, list):
        inputs = [inputs]
    return inputs


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
        inputs: Union[torch.Tensor, List[torch.Tensor]],
        *args,
        **kwargs,
    ):
        inputs = _to_tensor_list(inputs)
        residual = inputs[0]
        return residual + self.layer(inputs, *args, **kwargs)


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

    def forward(self, inputs: Union[torch.Tensor, List[torch.Tensor]], *args, **kwargs):
        inputs = _to_tensor_list(inputs)

        # Could be that the same tensor has been passed multiple times
        # in that case we'll just normalize once
        list_ids = [id(inp) for inp in inputs]
        if list_ids.count(list_ids[0]) == len(list_ids):
            normalized_input = self.norm(inputs[0])
            sublayer_inputs = [normalized_input for _ in inputs]
        else:
            sublayer_inputs = [self.norm(x_) for x_ in inputs]
        return self.sublayer(*sublayer_inputs, *args, **kwargs)


class PostNorm(nn.Module):
    """Adds LayerNorm after computing attention"""

    def __init__(self, d_model: int, sublayer: nn.Module, use_triton: bool = True):
        super().__init__()
        if _is_triton_available and use_triton:
            self.norm: Union[nn.LayerNorm, FusedLayerNorm] = FusedLayerNorm(d_model)
        else:
            self.norm = nn.LayerNorm(d_model)

        self.sublayer = sublayer

    def forward(self, inputs: Union[torch.Tensor, List[torch.Tensor]], *args, **kwargs):
        inputs = _to_tensor_list(inputs)

        x = self.sublayer(*inputs, *args, **kwargs)
        return self.norm(x)
