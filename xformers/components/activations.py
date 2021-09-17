# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from enum import Enum
from typing import Optional

from torch import nn


class Activation(str, Enum):
    GeLU = "gelu"
    LeakyReLU = "leaky_relu"
    ReLU = "relu"


# The following activations require their inputs to be saved
# to be able to compute their gradients
requires_bwd_inputs = [Activation.GeLU]


def build_activation(activation: Optional[Activation]):
    if not activation:
        return lambda x: x

    return {
        Activation.ReLU: nn.ReLU,
        Activation.GeLU: nn.GELU,
        Activation.LeakyReLU: nn.LeakyReLU,
    }[activation]()
