# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import random

import pytest
import torch

from xformers.components.attention import OrthoFormerAttention, ScaledDotProduct
from xformers.components.attention.utils import maybe_merge_masks


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "landmark_selection", ["orthogonal", "kmeans", "kmeans_spherical", "random"]
)
@pytest.mark.parametrize("num_landmarks", [30, 33, 905])
@pytest.mark.parametrize("subsample_fraction", [1.0, 0.3])
def test_ortho_attention(
    landmark_selection: str, num_landmarks: int, subsample_fraction: float
):
    # TODO: conv_kernel_size parameter not set to None fails this test. Investigate.
    b, s, d = 8, 900, 32
    num_heads = 2
    seed = 42
    torch.random.manual_seed(seed)
    random.seed(seed)

    ortho_config = {
        "name": "orthoformer",
        "dropout": 0.0,
        "num_landmarks": num_landmarks,
        "num_heads": num_heads,
        "landmark_selection": landmark_selection,
        "subsample_fraction": subsample_fraction,
    }

    sdp_config = {
        "name": "scaled_dot_product",
        "dropout": 0.0,
    }

    a = torch.rand(b, s, d, device=torch.device("cuda"))

    def test_close_to_sdp():
        # Make sure that Ortho and Normal attention are not too far off.
        ortho_attention = OrthoFormerAttention(**ortho_config).cuda()
        sdp_attention = ScaledDotProduct(**sdp_config).cuda()

        r_ortho = ortho_attention(a, a, a, att_mask=None)
        r_sdp = sdp_attention(a, a, a, att_mask=None)

        assert torch.allclose(r_ortho, r_sdp, rtol=0.02, atol=1e-1)

        # Make sure that OrthoFormerAttention and Normal attention are not too far off.
        ortho_attention = OrthoFormerAttention(**ortho_config).cuda()
        sdp_attention = ScaledDotProduct(**sdp_config).cuda()

        r_ortho = ortho_attention(a, a, a, att_mask=None)
        r_sdp = sdp_attention(a, a, a, att_mask=None)

        assert torch.allclose(r_ortho, r_sdp, rtol=0.02, atol=1e-1)

    def test_att_mask_ignored():
        # If an sxs attention mask is passed in, it should be ignored.
        # Results should be the same as if no mask was passed in.
        ortho_attention = OrthoFormerAttention(**ortho_config).cuda()
        sdp_attention = ScaledDotProduct(**sdp_config).cuda()

        key_padding_mask = None
        att_mask = torch.randint(0, 2, (s, s), device=torch.device("cuda")).to(
            dtype=torch.bool
        )
        sdp_mask = maybe_merge_masks(
            att_mask=None,
            key_padding_mask=key_padding_mask,
            batch_size=b // num_heads,
            src_len=s,
            num_heads=num_heads,
        )
        r_ortho = ortho_attention(
            a, a, a, att_mask=att_mask, key_padding_mask=key_padding_mask
        )
        r_sdp = sdp_attention(a, a, a, att_mask=sdp_mask)
        assert torch.allclose(r_ortho, r_sdp, rtol=0.02, atol=1e-1)

    test_close_to_sdp()
    test_att_mask_ignored()
