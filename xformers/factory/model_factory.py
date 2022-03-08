# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch

from xformers.components import reversible as rv
from xformers.components.residual import LayerNormStyle, get_deepnorm_coefficients
from xformers.factory.block_factory import (
    xFormerBlockConfig,
    xFormerDecoderBlock,
    xFormerDecoderConfig,
    xFormerEncoderBlock,
    xFormerEncoderConfig,
)


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
                "layer_norm_style": "pre",
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
                    "name": "FusedMLP",  # Use MLP if Triton is not available
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

    def __init__(
        self,
        stack_configs: Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]],
        tie_embedding_weights: bool = False,
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


class xFormer(torch.nn.Module):
    def __init__(
        self,
        stack_configs: Union[
            xFormerBlockConfig, List[xFormerBlockConfig], Dict[str, xFormerBlockConfig]
        ],
        tie_embedding_weights: bool = False,
    ):
        """
        Given a serialized configuration, generate the corresponding model.
        This is only a helper and can easily be bypassed
        """
        super().__init__()

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
                if i > 0:
                    config.layer_position.mark_not_first()
                if i < config.num_layers - 1:
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
            logging.info("Tying encoder and decoder embeddings, as requested")
            encoders[0].pose_encoding = decoders[0].pose_encoding

        self.encoders: torch.nn.Module = (
            rv.ReversibleSequence(torch.nn.ModuleList(encoders))
            if self.reversible_encoder
            else torch.nn.ModuleList(encoders)
        )
        self.decoders = torch.nn.ModuleList(decoders)

        use_deepnorm_init = stack_configs[0].layer_norm_style == LayerNormStyle.DeepNorm

        assert (
            not use_deepnorm_init or not self.reversible_encoder
        ), "Reversible layers and deepnorm is not supported for now"

        if use_deepnorm_init:
            self._deepnorm_weight_init()

    @classmethod
    def from_config(cls, config: xFormerConfig):
        return cls(config.stack_configs, config.tie_embedding_weights)

    def _deepnorm_weight_init(self):
        r"""Initiate parameters in the transformer model, using the DeepNorm_ method."""

        def is_ffn_w(n):
            return "feedforward" in n and "weight" in n

        def is_out_mha_proj_w(n):
            return "mha.proj" in n and "weight" in n

        def is_v_proj_weight(n):
            return "v_proj_weight" in n

        encoder_coefficients, decoder_coefficients = get_deepnorm_coefficients(
            encoder_layers=len(self.encoders), decoder_layers=len(self.decoders)  # type: ignore
        )

        encoder_gain = (
            encoder_coefficients.beta if encoder_coefficients is not None else 1.0
        )
        decoder_gain = (
            decoder_coefficients.beta if decoder_coefficients is not None else 1.0
        )

        # Initialize all the encoder weights
        for n, p in self.encoders.named_parameters():
            if is_ffn_w(n) or is_out_mha_proj_w(n) or is_v_proj_weight(n):
                torch.nn.init.xavier_normal_(p, gain=encoder_gain)
            elif "weight" in n and p.ndim > 1:
                torch.nn.init.xavier_normal_(p, gain=1)

        # Initialize all the decoder weights
        for n, p in self.decoders.named_parameters():
            if is_ffn_w(n) or is_out_mha_proj_w(n) or is_v_proj_weight(n):
                torch.nn.init.xavier_normal_(p, gain=decoder_gain)
            elif "weight" in n and p.ndim > 1:
                torch.nn.init.xavier_normal_(p, gain=1)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model
        following the Xavier distribution."""

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

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
            c.layer_norm_style == LayerNormStyle.DeepNorm for c in stack_configs
        ]

        assert all(deepnorm) or not any(deepnorm), (
            "All layers need to have the same deepnorm setting. "
            + f"Currently {deepnorm}"
        )

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

                # TODO: pass in key and value independently.
                kwargs = {"att_mask": encoder_input_mask}
                x = encoders(x, **kwargs)
                memory = torch.stack(x.chunk(2, dim=-1)).mean(dim=0)

            if not self.decoders:
                return memory

        # If decoder: either use the encoder ouput, or just decode, both options are possible
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
