# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import logging
from dataclasses import asdict
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from xformers.components import (
    PatchEmbeddingConfig,
    PostNorm,
    PreNorm,
    Residual,
    ResidualNormStyle,
    build_multi_head_attention,
    build_patch_embedding,
)
from xformers.components.attention import AttentionMask
from xformers.components.feedforward import build_feedforward
from xformers.components.positional_embedding import build_positional_embedding
from xformers.components.residual import get_deepnorm_coefficients
from xformers.components.simplicial_embedding import SimplicialEmbedding
from xformers.factory.block_configs import (
    NormalizationType,
    xFormerDecoderConfig,
    xFormerEncoderConfig,
)

logger = logging.getLogger("xformers")


def _get_ln_factory(
    d_model: int,
    residual_norm_style: Optional[ResidualNormStyle],
    use_triton: bool,
    residual: bool,
    normalization: NormalizationType = NormalizationType.LayerNorm,
    residual_scale: float = 1.0,
):
    """
    Handle all the supported residual path configurations.

    ..Note: we return the appropriate constructor, not an actual layer
    """

    def get_layer_wrapper(
        d_model: int,
        sublayer: nn.Module,
        residual_norm_style: Optional[ResidualNormStyle],
        residual: bool,
        residual_scale: float,
    ):
        if residual:
            if residual_norm_style == ResidualNormStyle.Pre:
                return Residual(
                    layer=PreNorm(d_model, sublayer, normalization, use_triton),
                    scale=None,
                )
            elif residual_norm_style == ResidualNormStyle.Post:
                return PostNorm(
                    d_model,
                    Residual(layer=sublayer, scale=None),
                    normalization,
                    use_triton,
                )
            elif residual_norm_style == ResidualNormStyle.DeepNorm:
                return PostNorm(
                    d_model,
                    Residual(layer=sublayer, scale=residual_scale),
                    normalization,
                    use_triton=use_triton,
                )
            else:
                raise ValueError

        return (
            PreNorm(d_model, sublayer, normalization, use_triton)
            if residual_norm_style == ResidualNormStyle.Pre
            else PostNorm(d_model, sublayer, normalization, use_triton)
        )

    def ln_factory(sublayer: nn.Module):
        return get_layer_wrapper(
            d_model, sublayer, residual_norm_style, residual, residual_scale
        )

    return ln_factory


class xFormerEncoderBlock(torch.nn.Module):
    r"""A vanilla Transformer Encoder block"""

    def __init__(self, config: xFormerEncoderConfig, **kwargs):
        super().__init__()

        self.reversible_f = None
        self.reversible_g = None
        self.residual_norm_style = config.residual_norm_style
        self.dim_model = config.dim_model

        # If this layer is the first one, and a pose encoding has been requested
        if (
            config.position_encoding_config is not None
            and config.layer_position.is_first()
        ):
            self.pose_encoding = build_positional_embedding(
                asdict(config.position_encoding_config)
            )

            pos_encoding_dim = config.position_encoding_config.dim_model
            mha_dim = config.multi_head_config["dim_model"]

            if pos_encoding_dim != mha_dim:
                logger.warning(
                    f"The embedding dim and model dim do not match ({pos_encoding_dim} vs {mha_dim}), adding a projector layer."  # noqa
                )
                self.embedding_projector = nn.Linear(pos_encoding_dim, mha_dim)
        else:
            self.pose_encoding = None

        if config.residual_norm_style == ResidualNormStyle.DeepNorm:
            # Just use the layer norm coefficient here,
            # the init will be handled at the xformers level (knows about encoder and decoder blocks)
            deep_norm_coefficients, _ = get_deepnorm_coefficients(
                encoder_layers=config.num_layers, decoder_layers=0
            )
            assert deep_norm_coefficients is not None
            residual_scale = deep_norm_coefficients.alpha
        else:
            residual_scale = 1.0

        # mini helper, builds a normalization layer with the right Pre/Post config, residuals, and the right dimensions
        ln_factory = _get_ln_factory(
            config.dim_model,
            config.residual_norm_style,
            use_triton=config.use_triton,
            residual=True,
            residual_scale=residual_scale,
            normalization=config.normalization,
        )

        mha = build_multi_head_attention(config.multi_head_config)
        feedforward = build_feedforward(asdict(config.feedforward_config))

        # Expose attention specific capabilities
        self.supports_attention_mask = mha.attention.supports_attention_mask
        self.requires_same_k_q_dimensions = mha.attention.requires_same_k_q_dimensions
        self.causal = (
            mha.attention.causal if hasattr(mha.attention, "causal") else False
        )

        # Wrappers handle the different layer norm styles (pre- and post-) and the residual path
        self.wrap_att = ln_factory(mha)
        self.wrap_ff: Union[Residual, PostNorm] = ln_factory(feedforward)
        if (
            config.residual_norm_style == ResidualNormStyle.Pre
            and config.layer_position.is_last()
        ):
            self.wrap_ff = PostNorm(
                config.dim_model,
                self.wrap_ff,
                normalization=config.normalization,
                use_triton=config.use_triton,
            )

        # Simplicial embeddings are only used if specified, and on the last layer
        self.simplicial_embedding: Optional[SimplicialEmbedding] = None
        if config.simplicial_embeddings is not None and config.layer_position.is_last():
            self.simplicial_embedding = SimplicialEmbedding(
                **config.simplicial_embeddings
            )

        # Optional patch embedding
        self.patch_emb: Optional[nn.Module] = None

        if config.patch_embedding_config is not None:
            self.patch_emb = build_patch_embedding(
                PatchEmbeddingConfig(**config.patch_embedding_config)
            )

    @classmethod
    def from_config(cls, config: xFormerEncoderConfig):
        return cls(config)

    @staticmethod
    def get_reversible_layer(config) -> Tuple[nn.Module, nn.Module]:
        ln_factory = _get_ln_factory(
            config.dim_model,
            config.residual_norm_style,
            residual=False,
            use_triton=config.use_triton,
            normalization=config.normalization,
        )

        mha = build_multi_head_attention(config.multi_head_config)
        feedforward = build_feedforward(asdict(config.feedforward_config))

        reversible_f = ln_factory(mha)
        reversible_g = ln_factory(feedforward)
        return reversible_f, reversible_g

    def forward(
        self,
        x: torch.Tensor,
        att_mask: Optional[Union[torch.Tensor, AttentionMask]] = None,
        input_mask: Optional[torch.Tensor] = None,
    ):
        if self.patch_emb is not None:
            x = self.patch_emb(x)

        if self.pose_encoding is not None:
            x = self.pose_encoding(x)

            if hasattr(self, "embedding_projector"):
                x = self.embedding_projector(x)

        # Handle the optional input masking, differs on Q, K, V
        if input_mask is not None:
            q = x
            k = x * input_mask.unsqueeze(-1)
            v = k
        else:
            q, k, v = x, x, x

        # Pre/Post norms and residual paths are already handled
        x = self.wrap_att(inputs=[q, k, v], att_mask=att_mask)
        x = self.wrap_ff(inputs=[x])

        # Optional simplicial embeddings
        if self.simplicial_embedding is not None:
            x = self.simplicial_embedding(x)

        return x


class xFormerDecoderBlock(torch.nn.Module):
    r"""A vanilla Transformer Decoder block

    ... note: this implementation is not (yet ?) reversible"""

    def __init__(self, config: xFormerDecoderConfig, **kwargs):
        super().__init__()

        # If this layer is the first one, and a pose encoding as been requested
        if (
            config.position_encoding_config is not None
            and config.layer_position.is_first()
        ):
            self.pose_encoding = build_positional_embedding(
                config.position_encoding_config
            )

            pos_encoding_dim = config.position_encoding_config.dim_model
            mha_dim = config.multi_head_config_masked["dim_model"]

            if pos_encoding_dim != mha_dim:

                logger.warning(
                    f"The embedding dim and model dim do not match ({pos_encoding_dim} vs {mha_dim}), adding a projector layer."  # noqa
                )

                self.embedding_projector = nn.Linear(pos_encoding_dim, mha_dim)
        else:
            self.pose_encoding = None

        if config.residual_norm_style == ResidualNormStyle.DeepNorm:
            # Just use the layer norm coefficient here,
            # the init will be handled at the xformers level (knows about encoder and decoder blocks)
            _, deep_norm_coefficients = get_deepnorm_coefficients(
                encoder_layers=0, decoder_layers=config.num_layers
            )
            assert deep_norm_coefficients is not None
            residual_scale = deep_norm_coefficients.alpha
        else:
            residual_scale = 1.0

        # mini helper, builds a LayerNorm with the right Pre/Post config and the right dimensions
        ln_factory = _get_ln_factory(
            config.dim_model,
            config.residual_norm_style,
            use_triton=config.use_triton,
            residual=True,
            residual_scale=residual_scale,
            normalization=config.normalization,
        )

        mha = build_multi_head_attention(config.multi_head_config_masked)
        cross_mha = build_multi_head_attention(config.multi_head_config_cross)
        feedforward = build_feedforward(config.feedforward_config)

        # Expose attention or feedforward specific capabilities
        self.supports_attention_mask = mha.attention.supports_attention_mask
        self.requires_same_k_q_dimensions = mha.attention.requires_same_k_q_dimensions
        self.requires_squared_context_length = (
            feedforward.requires_squared_context
            or mha.attention.requires_squared_context
        )

        self.causal_attention = (
            mha.attention.causal if hasattr(mha.attention, "causal") else False
        )

        # Wrappers handle the different layer norm styles (pre- and post-) and the residual path
        self.wrap_att = ln_factory(mha)
        self.wrap_cross = ln_factory(cross_mha)
        self.wrap_ff: Union[Residual, PostNorm] = ln_factory(feedforward)

        if (
            config.residual_norm_style == ResidualNormStyle.Pre
            and config.layer_position.is_last()
        ):
            self.wrap_ff = PostNorm(
                config.dim_model,
                self.wrap_ff,
                normalization=NormalizationType.LayerNorm,
            )

    @classmethod
    def from_config(cls, config: xFormerDecoderConfig):
        return cls(config)

    def forward(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        encoder_att_mask: Optional[Union[torch.Tensor, AttentionMask]] = None,
        decoder_att_mask: Optional[Union[torch.Tensor, AttentionMask]] = None,
        input_mask: Optional[torch.Tensor] = None,
    ):
        if self.pose_encoding is not None:
            target = self.pose_encoding(target)

            if hasattr(self, "embedding_projector"):
                target = self.embedding_projector(target)

        # Handle the optional input masking, differs on Q, K, V
        if input_mask is not None:
            target_q = target
            target_k = target * input_mask.unsqueeze(-1)
            target_v = target_k
        else:
            target_q, target_k, target_v = target, target, target

        x = self.wrap_att(
            inputs=[target_q, target_k, target_v], att_mask=decoder_att_mask
        )
        x = self.wrap_cross(inputs=[x, memory, memory], att_mask=encoder_att_mask)
        x = self.wrap_ff(inputs=[x])

        return x
