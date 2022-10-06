# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from enum import Enum
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from xformers import _is_triton_available

if _is_triton_available():
    from xformers.triton.layer_norm import FusedLayerNorm

from collections import namedtuple


class ResidualNormStyle(str, Enum):
    """Support different residual path and norm styles.
    See "On Layer Normalization in the Transformer Architecture",
    Xiong et al., https://arxiv.org/pdf/2002.04745v1.pdf
    """

    Pre = "pre"
    Post = "post"
    DeepNorm = "deepnorm"


class NormalizationType(str, Enum):
    LayerNorm = "layernorm"
    Skip = "skip"
    # TODO: BatchNorm = "batchnorm"
    # TODO: GroupNorm = "groupnorm"


def get_normalization_layer(normalization_type: NormalizationType):
    class Skip(nn.Module):
        def __init__(self, *_, **__) -> None:
            super().__init__()

        def forward(self, x: torch.Tensor, **_):
            return x

    return {
        NormalizationType.LayerNorm: nn.LayerNorm,
        NormalizationType.Skip: Skip,
    }[normalization_type]


class RequiresWrappedInputs:
    """Used to mark, through inheritance,
    the fact that this class will require inputs to be passed as a single list"""

    pass


# CREDITS: the following is inspired by FastAI's Transformer implementation
class Residual(nn.Module, RequiresWrappedInputs):
    """
    Object-oriented handling of the residual path

    This supports scaling of the residual path, as proposed by DeepNet_
    .. _DeepNet: https://arxiv.org/pdf/2203.00555v1.pdf

    .. Note: the wrapped layers must accept all the inputs as a single list
    """

    def __init__(self, layer: nn.Module, scale: Optional[float] = None):
        super().__init__()
        self.layer = layer
        self.scale = scale

        # PreNorm and PostNorm require all the tensors to be passed as a list
        self.wrap_inputs = isinstance(layer, RequiresWrappedInputs)

    def forward(self, inputs: List[torch.Tensor], **kwargs):
        if self.scale is not None:
            residue = inputs[0] * self.scale
        else:
            residue = inputs[0]

        if self.wrap_inputs:
            return residue + self.layer(inputs=inputs, **kwargs)

        else:
            return residue + self.layer(*inputs, **kwargs)


class PreNorm(nn.Module, RequiresWrappedInputs):
    """Adds a normalization before computing attention

    ..Note: If a list of inputs is passed, all of them get normalized"""

    def __init__(
        self,
        d_norm: int,
        sublayer: nn.Module,
        normalization: NormalizationType,
        use_triton: bool = True,
    ):

        super().__init__()
        if (
            _is_triton_available()
            and use_triton
            and normalization == NormalizationType.LayerNorm
        ):
            self.norm: Union[nn.LayerNorm, FusedLayerNorm] = FusedLayerNorm(d_norm)
        else:
            self.norm = get_normalization_layer(normalization)(d_norm)

        self.sublayer = sublayer
        self.wrap_inputs = isinstance(sublayer, RequiresWrappedInputs)

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


class PostNorm(nn.Module, RequiresWrappedInputs):
    """Adds LayerNorm after computing attention"""

    def __init__(
        self,
        d_norm: int,
        sublayer: nn.Module,
        normalization: NormalizationType,
        use_triton: bool = True,
    ):
        super().__init__()
        if (
            _is_triton_available()
            and use_triton
            and normalization == NormalizationType.LayerNorm
        ):
            self.norm: Union[nn.LayerNorm, FusedLayerNorm] = FusedLayerNorm(d_norm)
        else:
            self.norm = get_normalization_layer(normalization)(d_norm)

        self.sublayer = sublayer
        self.wrap_inputs = isinstance(sublayer, RequiresWrappedInputs)

    def forward(self, inputs: List[torch.Tensor], **kwargs):
        if self.wrap_inputs:
            x = self.sublayer(inputs=inputs, **kwargs)
        else:
            x = self.sublayer(*inputs, **kwargs)
        return self.norm(x)


DeepNormCoefficients = namedtuple("DeepNormCoefficients", ["alpha", "beta"])


def get_deepnorm_coefficients(
    encoder_layers: int, decoder_layers: int
) -> Tuple[Optional[DeepNormCoefficients], Optional[DeepNormCoefficients]]:
    """
    See DeepNet_.

    Returns alpha and beta depending on the number of encoder and decoder layers,
    first tuple is for the encoder and second for the decoder

    .. _DeepNet: https://arxiv.org/pdf/2203.00555v1.pdf
    """

    N = encoder_layers
    M = decoder_layers

    if decoder_layers == 0:
        # Encoder only
        return (
            DeepNormCoefficients(alpha=(2 * N) ** 0.25, beta=(8 * N) ** -0.25),
            None,
        )

    elif encoder_layers == 0:
        # Decoder only
        return None, DeepNormCoefficients(alpha=(2 * M) ** 0.25, beta=(8 * M) ** -0.25)
    else:
        # Encoder/decoder
        encoder_coeffs = DeepNormCoefficients(
            alpha=0.81 * ((N**4) * M) ** 0.0625, beta=0.87 * ((N**4) * M) ** -0.0625
        )

        decoder_coeffs = DeepNormCoefficients(
            alpha=(3 * M) ** 0.25, beta=(12 * M) ** -0.25
        )

        return (encoder_coeffs, decoder_coeffs)
