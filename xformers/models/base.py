# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from typing import Optional

from xformers.components import Activation


@dataclass
class ModelConfig:
    num_heads: int
    dim_sequence: int
    dim_embedding: int
    dim_feedforward: int
    num_encoder_layers: int
    num_decoder_layers: int
    activation: Activation
    attention_dropout: Optional[float]
    relu_dropout: Optional[float]
    dropout: Optional[float]
