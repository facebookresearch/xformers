# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch

from xformers._deprecation_warning import deprecated_function
from xformers.components import reversible as rv
from xformers.components.residual import ResidualNormStyle, get_deepnorm_coefficients
from xformers.factory.block_configs import (
    xFormerBlockConfig,
    xFormerDecoderConfig,
    xFormerEncoderConfig,
)
from xformers.factory.block_factory import xFormerDecoderBlock, xFormerEncoderBlock
from xformers.factory.weight_init import get_weight_init_fn, xFormerWeightInit

logger = logging.getLogger("xformers")


@dataclass(init=False)
class xFormerConfig:
    """
    The configuration structure to define a full Transformer.
    This can include a stack of encoder layers, and a stack of decoder layers.

    It is optionally possible to share the embedding weights in between
    the encoder and decoder positional encoding, as proposed for instance by
    `Using the Output Embedding to Improve Language Models`, Press et al.

    A full config example is for instance as follows:

    ::

        xformer_config = [
            {
                "reversible": False,  # Turn on to test the effect of using reversible layers
                "block_type": "encoder",
                "num_layers": LAYERS,
                "dim_model": EMB,
                "residual_norm_style": "pre",
                "position_encoding_config": {
                    "name": "vocab",
                    "seq_len": CONTEXT,
                    "vocab_size": VOCAB_SIZE,
                },
                "multi_head_config": {
                    "num_heads": NUM_HEADS,
                    "residual_dropout": RES_DROP,
                    "use_rotary_embeddings": True,
                    "attention": {
                        "name": ATTENTION_MECHANISM_STR,
                        "dropout": ATTN_DROP,
                        "causal": True,
                        "seq_len": CONTEXT,
                    },
                },
                "feedforward_config": {
                    "name": "MLP",
                    "dropout": MLP_DROP,
                    "activation": "gelu",
                    "hidden_layer_multiplier": MLP_MULTIPLIER,
                },
            }
        ]


    .. _`Using the Output Embedding to Improve Language Models`: https://arxiv.org/pdf/1608.05859.pdf
    """

    stack_configs: Union[List[xFormerBlockConfig], Dict[str, xFormerBlockConfig]]
    tie_embedding_weights: bool = False
    weight_init: xFormerWeightInit = xFormerWeightInit.ViT

    def __init__(
        self,
        stack_configs: Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]],
        tie_embedding_weights: bool = False,
        weight_init: xFormerWeightInit = xFormerWeightInit.ViT,
    ):
        # Type all the configurations. Possible typos are caught here
        if isinstance(stack_configs, dict):
            self.stack_configs = {}
            for k, config in stack_configs.items():
                if config["block_type"] == "encoder":
                    self.stack_configs[k] = xFormerEncoderConfig(**config)
                else:
                    self.stack_configs[k] = xFormerDecoderConfig(**config)
        else:
            self.stack_configs = []
            for config in stack_configs:
                if config["block_type"] == "encoder":
                    self.stack_configs.append(xFormerEncoderConfig(**config))
                else:
                    self.stack_configs.append(xFormerDecoderConfig(**config))

        self.tie_embedding_weights = tie_embedding_weights
        self.weight_init = weight_init
        deprecated_function(self)


class xFormer(torch.nn.Module):
    def __init__(
        self,
        stack_configs: Union[
            xFormerBlockConfig, List[xFormerBlockConfig], Dict[str, xFormerBlockConfig]
        ],
        tie_embedding_weights: bool = False,
        weight_init: xFormerWeightInit = xFormerWeightInit.ViT,
    ):
        """
        Given a serialized configuration, generate the corresponding model.
        This is only a helper and can easily be bypassed
        """
        super().__init__()
        deprecated_function(self)

        if isinstance(stack_configs, Dict):
            stack_configs = list(stack_configs.values())

        # Convenience, users can pass either a list of configs or a single one
        if not isinstance(stack_configs, List):
            stack_configs = [stack_configs]

        # Sanity checks, some config combinations do not make sense
        self._verify_reversible(stack_configs)
        self._verify_deepnorm(stack_configs)

        encoders: List[torch.nn.Module] = []
        decoders: List[torch.nn.Module] = []

        self.reversible_encoder = False
        self.rev_enc_pose_encoding = None

        # Unroll the configs and build the model
        for config in stack_configs:
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
            for i in range(config.num_layers):
                # Label where this layer is in the stack
                # (for instance useful for the positional encoding, or late layer norm)
                if len(recipient) > 0:
                    config.layer_position.mark_not_first()

                if config != stack_configs[-1] or i < config.num_layers - 1:
                    config.layer_position.mark_not_last()

                block = builder(config)  # type: ignore

                # If reversible: extract the reversible sub-parts, else append the block as-is
                if config.reversible:
                    # WARNING: only one pose encoding is saved here (not Focal Transformer compatible for instance)
                    assert isinstance(config, xFormerEncoderConfig)
                    if block.pose_encoding is not None:
                        self.rev_enc_pose_encoding = block.pose_encoding
                    self.reversible_encoder = True

                    f, g = xFormerEncoderBlock.get_reversible_layer(config)
                    recipient.append(torch.nn.ModuleList([f, g]))
                else:
                    recipient.append(block)  # type: ignore

        # Tie embedding weights, if requested and possible
        assert (
            not tie_embedding_weights or not self.reversible_encoder
        ), "Reversible layers and  tied embeddings is not supported for now"

        if (
            tie_embedding_weights
            and encoders
            and encoders[0].pose_encoding
            and decoders
            and decoders[0].pose_encoding
            and not config.reversible
        ):
            logger.info("Tying encoder and decoder embeddings, as requested")
            encoders[0].pose_encoding = decoders[0].pose_encoding

        self.encoders: torch.nn.Module = (
            rv.ReversibleSequence(torch.nn.ModuleList(encoders))
            if self.reversible_encoder
            else torch.nn.ModuleList(encoders)
        )
        self.decoders = torch.nn.ModuleList(decoders)

        use_deepnorm = (
            stack_configs[0].residual_norm_style == ResidualNormStyle.DeepNorm
        )

        assert (
            not use_deepnorm or not self.reversible_encoder
        ), "Reversible layers and deepnorm is not supported for now"

        self.init_weights(weight_init=weight_init, use_deep_norm=use_deepnorm)

    @classmethod
    def from_config(cls, config: xFormerConfig):
        return cls(
            config.stack_configs, config.tie_embedding_weights, config.weight_init
        )

    def _verify_reversible(self, stack_configs: List[xFormerBlockConfig]):
        reversible = [
            c.reversible
            for c in filter(lambda x: x.block_type == "encoder", stack_configs)
        ]

        assert all(reversible) or not any(reversible), (
            "All layers need to have the same reversibility setting. "
            + f"Currently {reversible}"
        )

    def _verify_deepnorm(self, stack_configs: List[xFormerBlockConfig]):
        deepnorm = [
            c.residual_norm_style == ResidualNormStyle.DeepNorm for c in stack_configs
        ]

        assert all(deepnorm) or not any(deepnorm), (
            "All layers need to have the same deepnorm setting. "
            + f"Currently {deepnorm}"
        )

    def init_weights(self, weight_init: xFormerWeightInit, use_deep_norm: bool):
        # The deepnorm weight initialization method requires different gain factors for the encoder
        # and decoder, depending on the general model structure (number of respective layers)
        if use_deep_norm:
            encoder_coefficients, decoder_coefficients = get_deepnorm_coefficients(
                encoder_layers=len(self.encoders), decoder_layers=len(self.decoders)  # type: ignore
            )
        else:
            encoder_coefficients, decoder_coefficients = None, None

        encoder_gain = (
            encoder_coefficients.beta if encoder_coefficients is not None else 1.0
        )
        decoder_gain = (
            decoder_coefficients.beta if decoder_coefficients is not None else 1.0
        )

        # Pick the desired init function
        init_fn = get_weight_init_fn(weight_init)

        # Initialize all the encoder weights
        for name, module in self.encoders.named_children():
            init_fn(module=module, name=name, gain=encoder_gain)

        for name, module in self.decoders.named_children():
            init_fn(module=module, name=name, gain=decoder_gain)

    def forward(
        self,
        src: torch.Tensor,
        tgt: Optional[torch.Tensor] = None,
        encoder_input_mask: Optional[torch.Tensor] = None,
        decoder_input_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:

        # Encode to latent space if encoder is present
        if len(list(self.encoders.parameters())) > 0:
            encoders = self.encoders
            memory = src.clone()
            if isinstance(encoders, torch.nn.ModuleList):
                for encoder in encoders:
                    memory = encoder(memory, input_mask=encoder_input_mask)
            else:
                if self.rev_enc_pose_encoding:
                    memory = self.rev_enc_pose_encoding(src)

                # Reversible Encoder
                x = torch.cat([memory, memory], dim=-1)

                # Apply the optional input masking
                if encoder_input_mask is not None:
                    if x.dim() - encoder_input_mask.dim() > 1:
                        encoder_input_mask.unsqueeze(0)
                    x += encoder_input_mask.unsqueeze(-1)

                x = encoders(x)
                memory = torch.stack(x.chunk(2, dim=-1)).mean(dim=0)

            if not self.decoders:
                return memory

        # If decoder: either use the encoder output, or just decode, both options are possible
        if len(self.decoders) > 0:
            tgt = src.clone() if tgt is None else tgt

            for decoder in self.decoders:
                tgt = decoder(
                    target=tgt,
                    # pyre-fixme[61]: `memory` is not always initialized here.
                    memory=memory,
                    input_mask=decoder_input_mask,
                )

            return tgt

        return None
