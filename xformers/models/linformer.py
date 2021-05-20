from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from xformers.components import Activation, MultiHeadDispatch
from xformers.components.attention import LinformerAttention
from xformers.components.feedforward import MLP
from xformers.models.base import ModelConfig


@dataclass(init=False)
class LinformerConfig(ModelConfig):
    k: int  # The dimension of the space on which the key and values are projected


class LinformerEncoderLayer(torch.nn.Module):
    """
    An implementation of a Linformer_ encoder layer using the xFormers components

    .. _Linformer: `Wang, S., Li, B. Z., Khabsa, M., Fang, H., & Ma, H. (2020).
    Linformer: Self-Attention with Linear Complexity. ArXiv, 2048(2019).`
    """

    def __init__(
        self,
        n_heads: int,
        dim_sequence: int,  # FIXME: should not be needed, make this dynamic
        dim_embedding: int,
        dim_feedforward: int,
        activation: Activation,
        attention_dropout: Optional[float],
        ff_dropout: Optional[float],
        final_dropout: Optional[float],
        k: int,
        *args,
        **kwargs,
    ):
        super().__init__()

        if not attention_dropout:
            attention_dropout = 0.0

        if not ff_dropout:
            ff_dropout = 0.0

        if not final_dropout:
            final_dropout = 0.0

        self.attention = LinformerAttention(
            dropout=attention_dropout, causal=False, max_seq_len=dim_sequence, k=k
        )
        self.multihead = MultiHeadDispatch(
            dim_model=dim_embedding,
            n_heads=n_heads,
            residual_dropout=attention_dropout,
            attention=self.attention,
        )

        self.norm1 = torch.nn.LayerNorm(dim_embedding)

        self.feedforward = MLP(
            dim_latent=dim_feedforward,
            dropout=ff_dropout,
            activation=activation,
            hidden_layer_multiplier=4,
        )
        self.norm2 = torch.nn.LayerNorm(dim_embedding)

        self.dropout = nn.Dropout(p=final_dropout)

    def forward(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # FIXME: Handle automatically different layer norm positions
        tensor = self.norm1(tensor + self.multihead(tensor, tensor, tensor))
        tensor = self.norm2(tensor + self.feedforward(tensor))
        tensor *= mask.unsqueeze(-1).type_as(tensor)
        return tensor

    @classmethod
    def from_config(cls, config: LinformerConfig):
        return cls(**config.as_patchy_dict())


class LinformerDecoderLayer(torch.nn.Module):
    """
    An implementation of a Linformer_ decoder layer using the xFormers components

    .. _Linformer: `Wang, S., Li, B. Z., Khabsa, M., Fang, H., & Ma, H. (2020).
    Linformer: Self-Attention with Linear Complexity. ArXiv, 2048(2019).`
    """

    def __init__(
        self,
        n_heads: int,
        dim_sequence: int,  # FIXME: should not be needed, make this dynamic
        dim_embedding: int,
        dim_feedforward: int,
        activation: Activation,
        attention_dropout: Optional[float],
        ff_dropout: Optional[float],
        final_dropout: Optional[float],
        k: int,
        *args,
        **kwargs,
    ):
        super().__init__()

        if not attention_dropout:
            attention_dropout = 0.0

        if not ff_dropout:
            ff_dropout = 0.0

        if not final_dropout:
            final_dropout = 0.0

        self.multihead1 = MultiHeadDispatch(
            dim_model=dim_embedding,
            n_heads=n_heads,
            residual_dropout=attention_dropout,
            attention=LinformerAttention(
                dropout=attention_dropout, causal=True, max_seq_len=dim_sequence, k=k
            ),
        )

        self.multihead2 = MultiHeadDispatch(
            dim_model=dim_embedding,
            n_heads=n_heads,
            residual_dropout=attention_dropout,
            attention=LinformerAttention(
                dropout=attention_dropout, causal=False, max_seq_len=dim_sequence, k=k
            ),
        )

        self.feedforward = MLP(
            dim_latent=dim_feedforward,
            dropout=ff_dropout,
            activation=activation,
            hidden_layer_multiplier=4,
        )
        self.norm1 = torch.nn.LayerNorm(dim_embedding)
        self.norm2 = torch.nn.LayerNorm(dim_embedding)
        self.norm3 = torch.nn.LayerNorm(dim_embedding)

        self.dropout = nn.Dropout(p=final_dropout)

    def forward(self, target: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # FIXME: Handle automatically different layer norm positions
        # Masked multi head attention
        x = self.norm1(target + self.multihead1(target, target, target))

        # Include the memory/Encoder results
        x = self.norm2(x + self.multihead2(key=memory, value=memory, query=x))

        # FF
        x = self.norm3(x + self.feedforward(x))
        return x

    @classmethod
    def from_config(cls, config: LinformerConfig):
        return cls(**config.as_patchy_dict())


class LinFormer(torch.nn.Module):
    def __init__(
        self,
        n_heads: int,
        dim_sequence: int,  # FIXME: should not be needed, make this dynamic
        dim_embedding: int,
        dim_feedforward: int,
        activation: Activation,
        num_encoder_layers: int,
        num_decoder_layers: int,
        attention_dropout: Optional[float],
        ff_dropout: Optional[float],
        final_dropout: Optional[float],
        k: int,
    ):
        super().__init__()

        encoders = [
            LinformerEncoderLayer(
                n_heads,
                dim_sequence,
                dim_embedding,
                dim_feedforward,
                activation,
                attention_dropout,
                ff_dropout,
                final_dropout,
                k,
            )
            for _ in range(num_encoder_layers)
        ]

        self.encoders = torch.nn.Sequential(*encoders) if encoders else None

        self.decoders = nn.ModuleList(
            [
                LinformerDecoderLayer(
                    n_heads,
                    dim_sequence,
                    dim_embedding,
                    dim_feedforward,
                    activation,
                    attention_dropout,
                    ff_dropout,
                    final_dropout,
                    k,
                )
                for _ in range(num_decoder_layers)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Encode to latent space if encoder is present
        latent = self.encoders(inputs) if self.encoders else None

        # If decoder: either use the encoder ouput, or just decode, both options are possible
        if self.decoders:
            for decoder in self.decoders:
                inputs = decoder(
                    target=inputs, memory=latent if latent is not None else inputs
                )

            return inputs

        # There was no decoder, we're looking for encoded values
        return latent

    @classmethod
    def from_config(cls, config: LinformerConfig):
        return cls(**config.as_patchy_dict())
