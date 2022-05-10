# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import math
from enum import Enum, auto
from typing import Optional

import torch
from torch.autograd.profiler import record_function

from .base import FeatureMap

"""
A set of feature maps which approximate the softmax kernel, as per the Performers_ paper.

_Performers: "Rethinking attention with performers." K. Choromanski et al. (2020).
    https://arxiv.org/pdf/2009.14794v1.pdf
"""


class NormDistribution(Enum):
    Xi = auto()
    Uniform = auto()


class SoftMaxPositiveEstimators(FeatureMap):
    def __init__(
        self,
        dim_features: int,
        iter_before_redraw: Optional[int],
        normalize_inputs: bool = False,
        epsilon: float = 1e-6,
        softmax_temp: float = -1,
    ):
        super().__init__(dim_features, iter_before_redraw, normalize_inputs, epsilon)
        self.softmax_temp = softmax_temp

        # Handle the scaling from all kernels by √m.
        # This normalizes for all the feature maps involved
        self.h_scale = math.log(math.sqrt(self.dim_features))

    def pre_scale(self, x: torch.Tensor) -> torch.Tensor:
        with record_function("feature_map::pre_scale"):
            # Re-draw counting logic
            if (
                (
                    self.iter_before_redraw is not None
                    and self._iter_counter > self.iter_before_redraw
                )
                or self.features is None
                or self.features.device != x.device
            ):
                # The feature map is actually using half the dimension, we'll concatenate + and - features
                self._iter_counter = 1
                self.features = self._get_feature_map(
                    x.shape[-1], self.dim_feature_map, x.device
                )

            features = self.features
            assert features is not None

            if features.dtype != x.dtype:
                self.features = features.to(x.dtype)

            self._iter_counter += 1

            # Normalization / softmax
            if self.softmax_temp < 0:
                # A = exp(QK.t/√d), so each input will be scaled by √√d
                self.softmax_temp = x.shape[-1] ** -0.25

            x_scaled = x * self.softmax_temp

            # Compute the scaling factors in logspace, applied from within the exponential
            # - dimnish possible exponential overflow
            # - remove a multiply across the batch, replace by an addition
            norm_x_2 = torch.einsum("...d,...d->...", x_scaled, x_scaled).unsqueeze(-1)
            self.offset = -0.5 * norm_x_2 - self.h_scale + self.epsilon

            if self.normalize_inputs:
                # L0 normalize the exponential term, can be useful for numerical stability
                # This ensures that features +- offset is below 1
                self.offset -= norm_x_2.max(1, keepdim=True)[0]

        # Return the scaled inputs, the rest depends on the kernel being used
        return x_scaled

    @staticmethod
    @torch.no_grad()
    def _get_random_ortho_matrix(
        blocks: int,
        dim: int,
        device: torch.device,
        norm_distribution: NormDistribution = NormDistribution.Uniform,
    ) -> torch.Tensor:
        r"""
        Generate a random matrix whose rows are exactly orthonormal

        "How to generate random matrices from the classical compact groups", Mezzadri, 2007
        https://arxiv.org/pdf/math-ph/0609050v2.pdf

        .. note: the typical qr decomposition does not give uniform results, qr decomposition is not
        unique and the qr decomposition routines are biased towards numerical stability. See the above
        paper for more information.

        .. note: this does not follow the original implementation from the Performers authors.
        see docs/assets/kde plots to visualize the impact of using the R signs to correct Q
        """

        H = torch.randn((blocks, dim, dim), device=device, requires_grad=False)

        # Randomly scale the norms of the features, Xi distributed
        if norm_distribution == NormDistribution.Xi:
            # NOTE: This averages to sqrt(d)
            norms = torch.sqrt(torch.einsum("...d,...d->...", H, H))

        Q, R = torch.linalg.qr(H)
        Q = torch.diag_embed(torch.sign(torch.diagonal(R, dim1=1, dim2=2))) @ Q

        # Normalize if need be. Uniform NormDistribution does nothing, Q is already orthonormal
        if norm_distribution == NormDistribution.Xi:
            return torch.diag_embed(norms) @ Q

        return Q


class SMOrf(SoftMaxPositiveEstimators):
    """
    "Positive random orthogonal features" softmax estimator,
    SM_ort^m+, as proposed in the Performers_ paper, Lemma 1.

    _Performers: "Rethinking attention with performers." K. Choromanski et al. (2020).
    https://arxiv.org/pdf/2009.14794v1.pdf
    """

    @torch.no_grad()
    def _get_feature_map(self, dim_input: int, dim_features: int, device: torch.device):
        """
        Generate the projection matrix onto the random features

        .. note: The heads dimension needs to be taken into account, hence the per-block random matrix
        and not uniformally random.
        """

        # Get per block random unitary matrices.
        # We need enough of them to project the whole input dimension, regardless of the
        # requested dimension of the features
        features = self._get_random_ortho_matrix(
            math.ceil(dim_input / dim_features),
            dim_features,
            norm_distribution=NormDistribution.Xi,
            device=device,
        )

        return features.flatten(0, 1)[:dim_input]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Softmax-dimension related scaling, shared for all kernels
        x_scaled = super().pre_scale(x)
        assert self.features is not None

        # Project onto the random feature map.
        x_scaled = x_scaled @ self.features
        return torch.exp(x_scaled + self.offset)


class SMHyperbolic(SoftMaxPositiveEstimators):
    """
    "Positive random features hyperbolic" estimator, SMHyp+,
    as proposed in the Performers_ paper, Lemma 1.

    _Performers: "Rethinking attention with performers." K. Choromanski et al. (2020).
    https://arxiv.org/pdf/2009.14794v1.pdf
    """

    def __init__(
        self,
        dim_features: int,
        iter_before_redraw: Optional[int],
        normalize_inputs: bool = False,
        epsilon: float = 1e-6,
        softmax_temp: float = -1,
    ):
        super().__init__(
            dim_features, iter_before_redraw, normalize_inputs, epsilon, softmax_temp
        )

        assert (
            dim_features % 2 == 0
        ), "The feature dimension needs to be even with this kernel"
        self.dim_feature_map = self.dim_features // 2

    @torch.no_grad()
    def _get_feature_map(self, dim_input: int, dim_features: int, device: torch.device):
        """
        Generate the projection matrix onto the random features

        .. note: The heads dimension needs to be taken into account, hence the per-block random matrix
        and not uniformally random.
        """

        # Get per block random unitary matrices.
        # We need enough of them to project the whole input dimension, regardless of the
        # requested dimension of the features
        features = self._get_random_ortho_matrix(
            math.ceil(dim_input / dim_features),
            dim_features,
            norm_distribution=NormDistribution.Xi,
            device=device,
        )

        return features.flatten(0, 1)[:dim_input]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Softmax-dimension related scaling, shared for all kernels
        x_scaled = super().pre_scale(x)

        # Project onto the random feature map, concatenate both + and - results
        # This follows Lemma 1 in the original Performers Paper to best approximate a
        # softmax kernel (cosh representation)
        x_scaled = x_scaled @ self.features
        return torch.cat(
            [torch.exp(x_scaled + self.offset), torch.exp(-x_scaled + self.offset)],
            dim=-1,
        )


class SMReg(SoftMaxPositiveEstimators):
    """
    "Regularized softmax kernel" estimator, SMREG+, as proposed in the Performers_ paper.

    _Performers: "Rethinking attention with performers." K. Choromanski et al. (2020).
    https://arxiv.org/pdf/2009.14794v1.pdf
    """

    def __init__(
        self,
        dim_features: int,
        iter_before_redraw: Optional[int],
        normalize_inputs: bool = False,
        epsilon: float = 1e-6,
        softmax_temp: float = -1,
    ):
        super().__init__(
            dim_features, iter_before_redraw, normalize_inputs, epsilon, softmax_temp
        )

        assert (
            dim_features % 2 == 0
        ), "The feature dimension needs to be even with this kernel"
        self.dim_feature_map = self.dim_features // 2

    @torch.no_grad()
    def _get_feature_map(self, dim_input: int, dim_features: int, device: torch.device):
        """
        Generate the projection matrix onto the random features

        .. note: The heads dimension needs to be taken into account, hence the per-block random matrix
        and not uniformally random.
        """

        # Get per block random unitary matrices.
        # We need enough of them to project the whole input dimension, regardless of the
        # requested dimension of the features
        features = self._get_random_ortho_matrix(
            math.ceil(dim_input / dim_features),
            dim_features,
            norm_distribution=NormDistribution.Uniform,
            device=device,
        ).flatten(0, 1)
        norms = math.sqrt(dim_input) * torch.ones(features.shape[0], device=device)
        return (torch.diag(norms) @ features)[:dim_input]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Softmax-dimension related scaling, shared for all kernels
        x_scaled = super().pre_scale(x)

        # Project onto the random feature map, concatenate both + and - results
        # This follows Lemma 1 in the original Performers Paper to best approximate a
        # softmax kernel (cosh representation + sample regularization)
        x_scaled = x_scaled @ self.features
        return torch.cat(
            [torch.exp(x_scaled + self.offset), torch.exp(-x_scaled + self.offset)],
            dim=-1,
        )
