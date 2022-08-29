# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import math

import pytest
import torch

from xformers.components.attention import FavorAttention, ScaledDotProduct
from xformers.components.attention.feature_maps import (
    FeatureMapType,
    NormDistribution,
    SMHyperbolic,
    SMOrf,
    SMReg,
)

_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@pytest.mark.parametrize("features", [SMOrf, SMHyperbolic, SMReg])
def test_random_matrix(features):
    torch.random.manual_seed(0)

    DRAWS = 100
    DIM = 10
    for _ in range(DRAWS):
        q = features._get_random_ortho_matrix(
            1, DIM, device=_device, norm_distribution=NormDistribution.Xi
        ).squeeze(0)

        # Check that the matrix is indeed orthonormal
        torch.allclose(
            torch.diag(q @ q.transpose(0, 1)),
            torch.diag(torch.ones(10, device=_device)),
        )

        # Check that the row norm is in the right ballpark (sqrt(dim))
        assert abs(torch.mean(torch.norm(q, dim=1)).item() - math.sqrt(DIM)) < 1.0


def _plot_distribution(ortho_feature_map):
    # Debug helper, check the uniformity of the random matrix draws
    DRAWS = 1000
    DIM = 50
    q = ortho_feature_map._get_random_ortho_matrix(DRAWS, DIM, device=_device)
    x, y = [], []

    for qq in q:
        # For every matrix, look at the real and imaginary eigen value
        e = torch.linalg.eigvals(qq)
        x.append(e.real)
        y.append(e.imag)

    # Ideally the repartition of the real and imaginary eigenvalues
    # should build a circle in the complex plane
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.kdeplot(x=torch.cat(x).cpu().numpy(), y=torch.cat(y).cpu().numpy())
    plt.axis("equal")
    plt.savefig("kde.png")


def _get_rng_data(device):
    emb = 10
    batch_size = 2
    seq_len = 20
    num_heads = 1

    shape = (batch_size * num_heads, seq_len, emb)
    return torch.randn(shape, device=device)


def test_feature_map_shape():
    # Check the delayed initialization of the feature map
    nb_random_features = 1000
    batch = _get_rng_data(_device)
    att = FavorAttention(
        dropout=0.0,
        dim_features=nb_random_features,
        feature_map_type=FeatureMapType.SMOrf,
    )
    _ = att(batch, batch, batch)

    assert att.feature_map.features.shape[0] == batch.shape[-1]
    assert att.feature_map.features.shape[1] == nb_random_features


def test_feature_map_redraw():
    # Check the delayed initialization of the feature map
    nb_random_features = 1000
    batch = _get_rng_data(_device)

    def check(should_redraw: bool):
        att = FavorAttention(
            dropout=0.0,
            dim_features=nb_random_features,
            feature_map_type=FeatureMapType.SMOrf,
            iter_before_redraw=1 if should_redraw else 100,
        )
        v0 = att(batch, batch, batch)
        assert att.feature_map is not None

        f0 = att.feature_map.features

        v1 = att(batch, batch, batch)
        f1 = att.feature_map.features

        # There should not have been a redraw after v0
        assert should_redraw != torch.allclose(v0, v1)
        assert should_redraw != torch.allclose(f0, f1)  # type: ignore

    check(should_redraw=True)
    check(should_redraw=False)


@pytest.mark.parametrize("feature", ["sm_orf", "sm_hyp", "sm_reg"])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("normalize_inputs", [True, False])
@pytest.mark.parametrize("device", [_device])
def test_favor_approximation_accuracy(feature, causal, normalize_inputs, device):
    # Run two attentions in parallel, the normal scaled dot product and the favor approximation

    torch.random.manual_seed(0)
    query, key, value = (
        _get_rng_data(device),
        _get_rng_data(device),
        _get_rng_data(device),
    )

    for x in (query, key, value):
        x.requires_grad = True

    # Build the two attention heads
    sdp_attention = ScaledDotProduct(dropout=0.0, causal=causal).to(device)
    approx_attention = FavorAttention(
        dropout=0.0,
        causal=causal,
        dim_head=10,
        feature_map_type=FeatureMapType(feature),
        normalize_inputs=normalize_inputs,
    ).to(device)

    with torch.cuda.amp.autocast(enabled=_device.type == "cuda"):
        standard_attention_result = sdp_attention(query, key, value)
        approx_attention_result = approx_attention(query, key, value)

        mismatch = torch.mean(
            (standard_attention_result - approx_attention_result) ** 2
        ).item()

        if causal:
            # FIXME(@lefaudeux) the causal case seems significantly worse, not obvious why,
            # could be worth investigating
            assert mismatch < 0.6
        else:
            assert mismatch < 0.23

        # Check trainability
        torch.sum(approx_attention_result).backward()


if __name__ == "__main__":
    _plot_distribution(SMOrf)
