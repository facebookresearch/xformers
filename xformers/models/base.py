from dataclasses import dataclass
from typing import Optional

from xformers.components import Activation
from xformers.utils import ExtensibleConfig


@dataclass
class ModelConfig(ExtensibleConfig):
    n_heads: int
    dim_sequence: int
    dim_embedding: int
    dim_feedforward: int
    num_encoder_layers: int
    num_decoder_layers: int
    activation: Activation
    attention_dropout: Optional[float]
    relu_dropout: Optional[float]
    dropout: Optional[float]
