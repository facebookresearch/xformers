from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch

from xformers.factory.block_factory import (
    BlockType,
    xFormerDecoderBlock,
    xFormerDecoderConfig,
    xFormerEncoderBlock,
    xFormerEncoderConfig,
)


@dataclass(init=False)
class xFormerStackConfig:
    """
    A stack is defined by the definition of a given block, and an optional repetition factor
    """

    block_config: Union[xFormerEncoderConfig, xFormerDecoderConfig]
    num_layers: int

    def __init__(self, block_config: Dict[str, Any]):

        if block_config["block_type"] == BlockType.Encoder:
            self.block_config = xFormerEncoderConfig(**block_config)
        else:
            self.block_config = xFormerDecoderConfig(**block_config)

        # Convenience: make num_layers optional, so that a stack at that point could
        # only be defined by a given block, and no repetition
        if "num_layers" in block_config.keys():
            self.num_layers = block_config["num_layers"]
        else:
            self.num_layers = 1


@dataclass(init=False)
class xFormerConfig:
    stack_configs: List[Union[xFormerStackConfig, xFormerStackConfig]]

    def __init__(self, block_configs: List[Dict[str, Any]]):
        # Type all the configurations. Possible typos are caught here
        self.stack_configs = [xFormerStackConfig(config) for config in block_configs]


class xFormer(torch.nn.Module):
    def __init__(
        self, stack_configs: List[Union[xFormerStackConfig, xFormerStackConfig]]
    ):
        """
        Given a serialized configuration, generate the corresponding model.
        This is only a helper and can easily be bypassed
        """

        super().__init__()

        encoders: List[torch.nn.Module] = []
        decoders: List[torch.nn.Module] = []

        for stack in stack_configs:
            config = stack.block_config

            # Handle either Encoder or Decoder stacks
            builder = (
                xFormerEncoderBlock.from_config
                if isinstance(config, xFormerEncoderConfig)
                else xFormerDecoderBlock.from_config
            )
            recipient = (
                encoders if isinstance(config, xFormerEncoderConfig) else decoders
            )

            # Build up the stack
            for i in range(stack.num_layers):
                if i > 0:
                    config.layer_position.mark_not_first()
                if i < stack.num_layers - 1:
                    config.layer_position.mark_not_last()

                recipient.append(builder(config))  # type: ignore

        self.encoders = torch.nn.ModuleList(encoders)
        self.decoders = torch.nn.ModuleList(decoders)

    @classmethod
    def from_config(cls, config: xFormerConfig):
        return cls(config.stack_configs)

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
