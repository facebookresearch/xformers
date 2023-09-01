# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import torch
import torch.autograd.profiler as profiler
import torch.nn as nn
import torch.nn.functional as Fn

from xformers.components.attention import (
    Attention,
    AttentionConfig,
    AttentionMask,
    register_attention,
)
from xformers.components.attention.core import (
    scaled_dot_product_attention,
    scaled_query_key_softmax,
)

logger = logging.getLogger("xformers")


class LandmarkSelection(str, Enum):
    Orthogonal = "orthogonal"
    KMeans = "kmeans"
    KMeans_Spherical = "kmeans_spherical"
    Random = "random"


@dataclass
class OrthoformerAttentionConfig(AttentionConfig):
    """
    num_landmarks           Number of landmarks to use for softmax approximation.
    subsample_fraction      Percentage of q_samples matrix to sample per iteration
    landmark_selection      Landmark selection strategy
    """

    num_landmarks: Optional[int]
    subsample_fraction: Optional[float]
    landmark_selection: Optional[LandmarkSelection]


@register_attention("orthoformer", OrthoformerAttentionConfig)
class OrthoFormerAttention(Attention):
    def __init__(
        self,
        dropout: float,
        num_landmarks: int = 32,
        subsample_fraction: float = 1.0,
        landmark_selection: LandmarkSelection = LandmarkSelection.Orthogonal,
        *args,
        **kwargs,
    ):
        """
        Orthoformer_ attention mechanism.
        ::

            "Keeping Your Eye on the Ball: Trajectory Attention in Video Transformers"
            Patrick, M., Campbell, D., Asano, Y., Misra, I., Metze, F., Feichtenhofer,
            C., Vedaldi, A., Henriques, J. (2021)

            Reference codebase: https://github.com/facebookresearch/Motionformer

        .. _Orthoformer: https://arxiv.org/abs/2106.05392

        """
        super().__init__()

        self.num_landmarks = num_landmarks
        self.attn_drop = nn.Dropout(dropout)
        self.subsample_fraction = subsample_fraction
        self.landmark_selection = landmark_selection

        # Properties specific to this attention mechanism
        self.supports_attention_mask = True
        self.supports_key_padding_mask = False

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_mask: Optional[Union[AttentionMask, torch.Tensor]] = None,
        *args,
        **kwargs,
    ):
        N = k.shape[1]

        if self.num_landmarks == N:
            #  Default attention
            x = scaled_dot_product_attention(q, k, v, att_mask)
        else:
            with torch.no_grad(), profiler.record_function("select landmarks"):
                if self.landmark_selection == LandmarkSelection.Orthogonal:
                    landmarks = self._compute_orthogonal_landmarks(q)
                elif self.landmark_selection == LandmarkSelection.Random:
                    half_L = self.num_landmarks // 2
                    landmarks_q = q[:, torch.randint(q.size(1), (half_L,)), :]
                    landmarks_k = k[:, torch.randint(k.size(1), (half_L,)), :]
                    landmarks = torch.cat((landmarks_q, landmarks_k), dim=-2)
                elif self.landmark_selection == LandmarkSelection.KMeans:
                    landmarks = self._cluster_landmarks(q)
                elif self.landmark_selection == LandmarkSelection.KMeans_Spherical:
                    landmarks = self._cluster_landmarks(q, spherical=True)

            if att_mask is not None:
                logger.warning(
                    "Orthoformer: attention mask passed alongside with using landmarks to reduce dimensions. \
                    The two are typically not compatible"
                )
                # FIXME: Should we still accept a mask in that case ?
                att_mask = None

            # pyre-ignore[61]: TODO(T103337542): `landmarks` mistakenly seems
            # like it could be uninitialized.
            kernel_1 = scaled_query_key_softmax(q, landmarks, att_mask)
            # pyre-ignore[61]: TODO(T103337542): `landmarks` mistakenly seems
            # like it could be uninitialized.
            kernel_2 = scaled_query_key_softmax(landmarks, k, att_mask)
            x = torch.matmul(kernel_1, torch.matmul(kernel_2, v))
        x = self.attn_drop(x)
        return x

    def _cluster_landmarks(
        self,
        q: torch.Tensor,
        spherical: bool = False,
        num_iters: int = 6,
    ) -> torch.Tensor:
        """
        Construct set of landmarks by recursively selecting new landmarks
        that are maximally orthogonal to the existing set.
        Returns near orthogonal landmarks with shape (B, M, D).
        """

        num_landmarks = min(self.num_landmarks, q.shape[1])

        if self.subsample_fraction < 1.0:
            num_samples = max(
                int(self.subsample_fraction * q.size(-2)), num_landmarks
            )  # Need at least M/2 samples of queries and keys
            q_samples = q[:, torch.randint(q.size(-2), (num_samples,)), :]  # (B, N, D)
        else:
            q_samples = q  # (B, N, D)

        if spherical:
            q_samples_normalized = Fn.normalize(
                q_samples, p=2, dim=-1
            )  # may need to change default eps to eps=1e-8 for mixed precision compatibility
            landmarks = self._kmeans_spherical(
                q_samples_normalized, num_landmarks, num_iters
            )
        else:
            landmarks = self._kmeans(q_samples, num_landmarks, num_iters)
        return landmarks  # (B, M, D)

    def _kmeans(self, x: torch.Tensor, K: int, num_iters: int = 10):
        """
        Arguments:
            x: (B, N, D)
            K: number of clusters
            num_iters: the number of kmeans updates
        """

        B, N, D = x.size()
        assert K <= N, f"{K} > {N}"

        c = x[
            :, torch.randperm(N, device=x.device)[:K], :
        ].clone()  # initialisation for the centroids

        with profiler.record_function("kmeans"):
            x_i = x.view(B, N, 1, D)
            c_j = c.view(B, 1, K, D)
            counts = c.new_zeros(B, K)
            ones = x.new_ones((B, N))

            for _ in range(num_iters):
                # E step: assign points to the nearest cluster
                D_ij = ((x_i - c_j) ** 2).sum(-1)  # (B, N, K) squared distances
                cl = D_ij.argmin(
                    dim=-1, keepdim=True
                ).long()  # (B, N, 1) index of point to nearest cluster

                # M step: update the centroids
                c.zero_()
                c.scatter_add_(-2, cl.repeat(1, 1, D), x)  # sum of points per cluster
                counts.fill_(1e-6)  # avoid div0
                counts.scatter_add_(
                    -1, cl.squeeze(-1), ones
                )  # number of points per cluster
                c.divide_(counts.unsqueeze(-1))  # compute the average

        return c

    def _kmeans_spherical(self, x: torch.Tensor, K: int, num_iters=10):
        """
        Arguments:
            x: (B, N, D)
        """
        B, N, D = x.size()
        assert K <= N, f"{K} > {N}"

        # initialisation for the centroids
        c = x[:, torch.randperm(N, device=x.device)[:K], :].clone()

        with profiler.record_function("kmeans_spherical"):
            counts = c.new_zeros(B, K)
            ones = x.new_ones((B, N))

            for _ in range(num_iters):
                # E step: assign points to the nearest cluster
                D_ij = torch.matmul(
                    x, c.transpose(-2, -1)
                )  # (B, N, K) cosine similarity
                cl = D_ij.argmax(
                    dim=-1, keepdim=True
                ).long()  # (B, N, 1) index of point to nearest cluster

                # M step: update the centroids
                c.zero_()
                c.scatter_add_(-2, cl.repeat(1, 1, D), x)  # sum of points per cluster
                counts.fill_(1e-6)  # avoid div0
                counts.scatter_add_(
                    -1, cl.squeeze(-1), ones
                )  # number of points per cluster
                c.divide_(counts.unsqueeze(-1))  # compute the average
                c = Fn.normalize(c, p=2, dim=-1)  # renormalise
        return c

    def _compute_orthogonal_landmarks(self, q: torch.Tensor) -> torch.Tensor:
        """
        Construct set of landmarks by recursively selecting new landmarks
        that are maximally orthogonal to the existing set.
        Returns near orthogonal landmarks with shape (B, M, D).
        """

        if self.subsample_fraction < 1.0:
            # Need at least M samples of queries
            num_samples = max(
                int(self.subsample_fraction * q.size(-2)), self.num_landmarks
            )
            q_samples = q[
                :, torch.randint(q.size(-2), (num_samples,), device=q.device), :
            ]
        else:
            # (B, N, D)
            q_samples = q

        # may need to change default eps to eps=1e-8 for mixed precision compatibility
        q_samples_normalized = Fn.normalize(q_samples, p=2, dim=-1)
        B, N, D = q_samples_normalized.shape

        selected_mask = torch.zeros((B, N, 1), device=q_samples_normalized.device)
        landmark_mask = torch.ones(
            (B, 1, 1), dtype=selected_mask.dtype, device=q_samples_normalized.device
        )

        #  Get initial random landmark
        random_idx = torch.randint(
            q_samples_normalized.size(-2), (B, 1, 1), device=q_samples_normalized.device
        )
        selected_mask.scatter_(-2, random_idx, landmark_mask)

        #  Selected landmarks
        selected_landmarks = torch.empty(
            (B, self.num_landmarks, D),
            device=q_samples_normalized.device,
            dtype=q_samples_normalized.dtype,
        )
        selected_landmarks[:, 0, :] = q_samples_normalized[
            torch.arange(q_samples_normalized.size(0)), random_idx.view(-1), :
        ].view(B, D)

        # Store computed cosine similarities
        cos_sims = torch.empty(
            (B, N, self.num_landmarks),
            device=q_samples_normalized.device,
            dtype=q_samples_normalized.dtype,
        )

        for M in range(1, self.num_landmarks):
            with profiler.record_function("find new landmark"):
                #  Calculate absolute cosine similarity between selected and unselected landmarks
                # (B, N, D) * (B, D) -> (B, N)
                cos_sims[:, :, M - 1] = torch.einsum(
                    "b n d, b d -> b n",
                    q_samples_normalized,
                    selected_landmarks[:, M - 1, :],
                ).abs()

                # (B, N, M) cosine similarities of current set of landmarks wrt all queries and keys
                cos_sim_set = cos_sims[:, :, :M]

                #  Get orthogonal landmark: landmark with smallest absolute cosine similarity:
                # set cosine similarity for already selected landmarks to > 1
                cos_sim_set.view(-1, M)[selected_mask.flatten().bool(), :] = 10

                # (B,) - want max for non
                selected_landmark_idx = cos_sim_set.amax(-1).argmin(-1)

                #  Add most orthogonal landmark to selected landmarks:
                selected_landmarks[:, M, :] = q_samples_normalized[
                    torch.arange(q_samples_normalized.size(0)), selected_landmark_idx, :
                ].view(B, D)

                #  Removed selected indices from non-selected mask:
                selected_mask.scatter_(
                    -2, selected_landmark_idx.unsqueeze(-1).unsqueeze(-1), landmark_mask
                )

        # (B, M, D)
        landmarks = torch.masked_select(q_samples, selected_mask.bool()).reshape(
            B, -1, D
        )
        return landmarks  # (B, M, D)
