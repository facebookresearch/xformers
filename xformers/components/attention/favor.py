# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from xformers.components.attention import Attention, AttentionConfig, register_attention
from xformers.components.attention.feature_maps import (
    FeatureMap,
    FeatureMapType,
    SMHyperbolic,
    SMOrf,
    SMReg,
)

logger = logging.getLogger("xformers")


@dataclass
class FavorAttentionConfig(AttentionConfig):
    causal: Optional[bool]
    dim_features: Optional[int] = None  # The dimensions of the random features
    dim_head: Optional[
        int
    ] = None  # The embedding dimension of the inputs. Only useful to get a dim_features estimate
    iter_before_redraw: Optional[
        int
    ] = None  # The number of iterations before the random features are re-drawn from scratch
    feature_map: Optional[FeatureMapType] = None


@register_attention("favor", FavorAttentionConfig)
class FavorAttention(Attention):
    def __init__(
        self,
        causal: bool = False,
        dropout: float = 0.0,
        dim_features: Optional[int] = None,
        dim_head: Optional[int] = None,
        iter_before_redraw: Optional[int] = None,
        feature_map_type: FeatureMapType = FeatureMapType.SMReg,
        normalize_inputs: bool = False,
        *_,
        **__,
    ):
        r"""
        Kernelized attention, as proposed in Performers_
        ("Rethinking attention with performers." K. Choromanski et al. (2020).).

        FAVOR stands for "Fast Attention Via positive Orthogonal Random features"

        Args:
            dropout (float): the probability of an output to be randomly dropped at training time
            dim_features (int): the dimension of the random features space
            iter_before_redraw (int): the number of steps (forward calls) before a redraw of the features
            feature_map_type (FeatureMapType): the type of feature map being used,
            for instance orthogonal random features.

        .. _Performers: https://arxiv.org/pdf/2009.14794v1.pdf
        """
        super().__init__()

        self.causal = causal
        self.iter_before_redraw = (
            (2 * iter_before_redraw)
            if iter_before_redraw is not None
            else iter_before_redraw
        )  # This will be used for both key and query
        self.normalize_inputs = normalize_inputs
        self.feature_map_type = feature_map_type
        self.attn_drop = nn.Dropout(dropout, inplace=True)

        # Setup dimension-dependent variables
        # Reasonable dimension default
        if dim_features is None:
            assert dim_head is not None, "dim_features or dim_head needs to be passed"
            self.dim_features = math.ceil(dim_head * (1 + math.log2(dim_head)))
            self.dim_features = 2 * (
                self.dim_features // 2
            )  # needs to be even for some variants
            logger.info(
                f"FAVOR: Automatically setting the random mapping dimension to {self.dim_features} from {dim_head}"
            )
        else:
            self.dim_features = dim_features

        feature_map_constructor = {
            FeatureMapType.SMHyp: SMHyperbolic,
            FeatureMapType.SMReg: SMReg,
            FeatureMapType.SMOrf: SMOrf,
        }[self.feature_map_type]

        feature_settings = {
            "dim_features": self.dim_features,
            "iter_before_redraw": self.iter_before_redraw,
            "normalize_inputs": self.normalize_inputs,
        }

        self.feature_map: FeatureMap = feature_map_constructor(**feature_settings)  # type: ignore

        # Properties specific to this attention mechanism
        self.supports_attention_mask = False
        self.supports_key_padding_mask = False

    @staticmethod
    def _maybe_promote(x: torch.Tensor) -> torch.Tensor:
        # Only promote fp16 buffers, bfloat16 would be fine for instance
        return x.float() if x.dtype == torch.float16 else x

    @staticmethod
    def _causal_attention(
        k_prime: torch.Tensor, q_prime: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Algorithm 1 in the paper
        ref_v = torch.ones_like(v.unsqueeze(2))  # BATCH x SEQ x 1 x EMB
        Gps = k_prime.unsqueeze(3) * v.unsqueeze(2)
        Grenorm = k_prime.unsqueeze(3) * ref_v

        # Consolidate against the feature dimension
        att_raw = torch.einsum("bcfe,bcf->bce", Gps, q_prime)
        att_norm = torch.einsum("bcfe,bcf->bce", Grenorm, q_prime)

        # Cumulative sum over the sequence
        att_raw = att_raw.cumsum(2)
        att_norm = att_norm.cumsum(2)

        return att_raw, att_norm

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *_,
        **__,
    ):

        # Project key and queries onto the feature map space
        k_prime = self.feature_map(k)
        q_prime = self.feature_map(q)

        with autocast(enabled=False):
            # The softmax kernel approximation for Favor will easily overflow
            # Force the computations here to stay in fp32 for numerical stability
            # Note that the dimensions are vastly reduced when compared to scaled_dot_product
            k_prime = self._maybe_promote(k_prime)
            q_prime = self._maybe_promote(q_prime)
            v = self._maybe_promote(v)

            if not self.causal:
                att_normalization = q_prime @ (
                    k_prime.transpose(-2, -1) @ torch.ones_like(v)
                )
                att_raw = q_prime @ (k_prime.transpose(-2, -1) @ v)
            else:
                # Actually compute attention
                att_raw, att_normalization = self._causal_attention(k_prime, q_prime, v)

            # Normalize
            att = att_raw / att_normalization

        if self.attn_drop is not None:
            att = self.attn_drop(att)

        return att
