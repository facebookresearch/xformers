from dataclasses import dataclass
from typing import List, Optional, Union, cast

import torch

from xformers.factory.block_factory import (
    BlockType,
    xFormerDecoderBlock,
    xFormerDecoderConfig,
    xFormerEncoderBlock,
    xFormerEncoderConfig,
)


@dataclass(init=False)
class xFormerConfig:
    block_configs: List[Union[xFormerEncoderConfig, xFormerDecoderConfig]]

    def __init__(self, block_configs):
        typed_configs = []

        for config in block_configs:
            if config["block_type"] == BlockType.Encoder:
                typed_configs.append(xFormerEncoderConfig(**config))
            else:
                typed_configs.append(xFormerDecoderConfig(**config))

        self.block_configs = typed_configs


class xFormer(torch.nn.Module):
    def __init__(
        self, block_configs: List[Union[xFormerEncoderConfig, xFormerDecoderConfig]]
    ):
        """
        Given a serialized configuration, generate the corresponding model.
        This is only a helper and can easily be bypassed
        """

        super().__init__()

        encoders: List[torch.nn.Module] = []
        decoders: List[torch.nn.Module] = []

        for config in block_configs:
            if type(config) is xFormerEncoderConfig:
                config = cast(xFormerEncoderConfig, config)
                for i in range(config.num_layers):
                    if i > 0:
                        config.position_encoding_config = None
                    encoders.append(xFormerEncoderBlock.from_config(config))

            elif type(config) is xFormerDecoderConfig:
                config = cast(xFormerDecoderConfig, config)
                for i in range(config.num_layers):
                    if i > 0:
                        config.position_encoding_config = None
                    decoders.append(xFormerDecoderBlock.from_config(config))

            else:
                raise NotImplementedError(f"{config} is not supported")

        self.encoders = torch.nn.ModuleList(encoders)
        self.decoders = torch.nn.ModuleList(decoders)

    @classmethod
    def from_config(cls, config: xFormerConfig):
        return cls(config.block_configs)

    def forward(
        self,
        inputs: torch.Tensor,
        encoder_input_mask: Optional[torch.Tensor] = None,
        decoder_input_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        # Encode to latent space if encoder is present
        latent = inputs

        if self.encoders:
            for encoder in self.encoders:
                latent = encoder(latent, input_mask=encoder_input_mask)

        # If decoder: either use the encoder ouput, or just decode, both options are possible
        if self.decoders:
            for decoder in self.decoders:
                inputs = decoder(
                    target=inputs,
                    memory=latent,
                    input_mask=decoder_input_mask,
                )

            return inputs

        # There was no decoder, we're looking for encoded values
        return latent
