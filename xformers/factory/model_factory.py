from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch

from xformers.components import reversible as rv
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
    reversible: bool

    def __init__(
        self, block_config: Dict[str, Any], reversible: bool = False, *args, **kwargs
    ):
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

        self.reversible = reversible


@dataclass(init=False)
class xFormerConfig:
    stack_configs: List[xFormerStackConfig]

    def __init__(self, stack_configs: List[Dict[str, Any]]):
        # Type all the configurations. Possible typos are caught here
        self.stack_configs = [xFormerStackConfig(**config) for config in stack_configs]


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

        self.reversible_encoder = False
        self.enc_pose_encoding = None
        self.dec_pose_encoding = None

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
                # Label where this layer is in the stack
                # (for instance useful for the positional encoding, or late layer norm)
                if i > 0:
                    config.layer_position.mark_not_first()
                if i < stack.num_layers - 1:
                    config.layer_position.mark_not_last()
                block = builder(config)  # type: ignore

                # If reversible: extract the reversible sub-parts, else append the block as-is
                if stack.reversible:
                    # WARNING: only one pose encoding is saved here (not Focal Transformer compatible for instance)
                    assert isinstance(config, xFormerEncoderConfig)
                    if block.pose_encoding is not None:
                        self.enc_pose_encoding = block.pose_encoding
                    self.reversible_encoder = True

                    f, g = xFormerEncoderBlock.get_reversible_layer(config)
                    recipient.append(torch.nn.ModuleList([f, g]))
                else:
                    recipient.append(block)  # type: ignore

        self.encoders = (
            rv.ReversibleSequence(torch.nn.ModuleList(encoders))
            if self.reversible_encoder
            else torch.nn.ModuleList(encoders)
        )
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

        latent = inputs.clone()  # Make sure that we don't modify the inputs in place

        # Encode to latent space if encoder is present
        if self.encoders:
            if isinstance(self.encoders, torch.nn.ModuleList):
                for encoder in self.encoders:
                    latent = encoder(latent, input_mask=encoder_input_mask)
            else:
                if self.enc_pose_encoding:
                    latent = self.enc_pose_encoding(inputs)

                # Reversible Encoder
                x = torch.cat([latent, latent], dim=-1)

                # TODO: pass in key and value independently.
                kwargs = {"att_mask": encoder_input_mask}
                x = self.encoders(x, **kwargs)
                latent = torch.stack(x.chunk(2, dim=-1)).mean(dim=0)

        # If decoder: either use the encoder ouput, or just decode, both options are possible
        if self.decoders:
            x = inputs.clone()

            for decoder in self.decoders:
                x = decoder(
                    target=x,
                    memory=latent,
                    input_mask=decoder_input_mask,
                )

            return x

        # There was no decoder, we're looking for encoded values
        return latent
