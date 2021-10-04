# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from xformers.components.attention import NystromAttention, ScaledDotProduct


@pytest.mark.parametrize("pinverse_original_init", [True, False])
@pytest.mark.parametrize("use_razavi_pinverse", [True, False])
@pytest.mark.parametrize("num_landmarks", [30, 33])
def test_nystrom_attention(
    pinverse_original_init: bool,
    use_razavi_pinverse: bool,
    num_landmarks: int,
):
    # TODO: conv_kernel_size parameter not set to None fails this test. Investigate.
    b, s, d = 8, 900, 384
    seed = 42
    torch.random.manual_seed(seed)

    def test_close_to_sdp():
        # Make sure that Nystrom and Normal attention are not too far off.
        a = torch.rand(b, s, d)
        nystrom_config = {
            "name": "nystrom",
            "dropout": 0.0,
            "num_landmarks": num_landmarks,
            "num_heads": 2,
            "pinverse_original_init": pinverse_original_init,
            "use_razavi_pinverse": use_razavi_pinverse,
        }

        sdp_config = {
            "name": "scaled_dot_product",
            "dropout": 0.0,
        }

        nystrom_attention = NystromAttention(**nystrom_config)
        sdp_attention = ScaledDotProduct(**sdp_config)

        r_nystrom = nystrom_attention(a, a, a)
        r_sdp = sdp_attention(a, a, a)

        assert torch.allclose(r_nystrom, r_sdp, rtol=0.005, atol=1e-2)

    test_close_to_sdp()
