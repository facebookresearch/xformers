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

from xformers.components.attention import Attention, AttentionConfig, register_attention
from xformers.components.attention.feature_maps import (
    FeatureMap,
    FeatureMapType,
    SMHyperbolic,
    SMOrf,
    SMReg,
)


@dataclass
class FavorAttentionConfig(AttentionConfig):
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
        Kernelized attention, as proposed in Performers_.

        FAVOR stands for "Fast Attention Via positive Orthogonal Random features"

        Args:
            dropout (float): the probability of an output to be randomly dropped at training time
            dim_features (int): the dimension of the random features space
            iter_before_redraw (int): the number of iterations before a redraw of the features
            feature_map_type (FeatureMapType): the type of feature map being used,
            for instance orthogonal random features.

        .. _Performers: "Rethinking attention with performers." K. Choromanski et al. (2020).
            https://arxiv.org/pdf/2009.14794v1.pdf
        """
        super().__init__()

        self.causal = causal
        self.iter_before_redraw = iter_before_redraw
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
            logging.info(
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

        self.feature_map_query: FeatureMap = feature_map_constructor(**feature_settings)  # type: ignore
        self.feature_map_key: FeatureMap = feature_map_constructor(**feature_settings)  # type: ignore

    @staticmethod
    def _causal_attention(
        k_prime: torch.Tensor, q_prime: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO(@lefaudeux): Rewrite as suggested in
        # https://github.com/fairinternal/xformers/pull/61#discussion_r628420093
        # Algorithm 1 in the paper
        ref_v = torch.ones_like(v[:, 0, :].unsqueeze(1))

        Gps = k_prime[:, 0, :].unsqueeze(2) * v[:, 0, :].unsqueeze(1)
        Grenorm = k_prime[:, 0, :].unsqueeze(2) * ref_v

        att_raw = [torch.bmm(q_prime[:, 0, :].unsqueeze(1), Gps)]
        att_norm = [torch.bmm(q_prime[:, 0, :].unsqueeze(1), Grenorm)]

        for i in range(1, k_prime.shape[1]):
            Gps += k_prime[:, i, :].unsqueeze(2) * v[:, i, :].unsqueeze(1)
            Grenorm += k_prime[:, i, :].unsqueeze(2) * ref_v

            att_raw.append(torch.bmm(q_prime[:, i, :].unsqueeze(1), Gps))
            att_norm.append(torch.bmm(q_prime[:, i, :].unsqueeze(1), Grenorm))

        return torch.cat(att_raw, dim=1), torch.cat(att_norm, dim=1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *args,
        **kwargs,
    ):
        # Project key and queries onto the feature map space
        k_prime = self.feature_map_key(k)
        q_prime = self.feature_map_query(q)
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
