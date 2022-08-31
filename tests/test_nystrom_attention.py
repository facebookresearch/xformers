# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import random

import pytest
import torch

from xformers.components.attention import NystromAttention, ScaledDotProduct
from xformers.components.attention.utils import maybe_merge_masks


@pytest.mark.parametrize("pinverse_original_init", [True, False])
@pytest.mark.parametrize("use_razavi_pinverse", [True, False])
@pytest.mark.parametrize("num_landmarks", [30, 33, 905])
def test_nystrom_attention_close_to_sdp(
    pinverse_original_init: bool,
    use_razavi_pinverse: bool,
    num_landmarks: int,
):
    # TODO: conv_kernel_size parameter not set to None fails this test. Investigate.
    b, s, d = 2, 900, 40
    num_heads = 2
    seed = 42
    torch.random.manual_seed(seed)
    random.seed(seed)

    nystrom_config = {
        "name": "nystrom",
        "dropout": 0.0,
        "num_landmarks": num_landmarks,
        "num_heads": num_heads,
        "pinverse_original_init": pinverse_original_init,
        "use_razavi_pinverse": use_razavi_pinverse,
    }

    sdp_config = {
        "name": "scaled_dot_product",
        "dropout": 0.0,
    }

    a = torch.rand(b, s, d)

    def test_close_to_sdp():
        # Make sure that Nystrom and Normal attention are not too far off.

        nystrom_attention = NystromAttention(**nystrom_config)
        sdp_attention = ScaledDotProduct(**sdp_config)

        r_nystrom = nystrom_attention(a, a, a, att_mask=None)
        r_sdp = sdp_attention(a, a, a, att_mask=None)

        assert torch.allclose(r_nystrom, r_sdp, rtol=0.005, atol=1e-2)

        # Make sure that Nystrom and Normal attention are not too far off.

        nystrom_attention = NystromAttention(**nystrom_config)
        sdp_attention = ScaledDotProduct(**sdp_config)

        r_nystrom = nystrom_attention(a, a, a, att_mask=None)
        r_sdp = sdp_attention(a, a, a, att_mask=None)

        assert torch.allclose(r_nystrom, r_sdp, rtol=0.005, atol=1e-2)

    test_close_to_sdp()


@pytest.mark.parametrize("pinverse_original_init", [True])
@pytest.mark.parametrize("use_razavi_pinverse", [True])
@pytest.mark.parametrize("num_landmarks", [30])
def test_nystrom_attention(
    pinverse_original_init: bool,
    use_razavi_pinverse: bool,
    num_landmarks: int,
):
    # TODO: conv_kernel_size parameter not set to None fails this test. Investigate.
    b, s, d = 2, 900, 40
    num_heads = 2
    seed = 42
    torch.random.manual_seed(seed)
    random.seed(seed)

    nystrom_config = {
        "name": "nystrom",
        "dropout": 0.0,
        "num_landmarks": num_landmarks,
        "num_heads": num_heads,
        "pinverse_original_init": pinverse_original_init,
        "use_razavi_pinverse": use_razavi_pinverse,
    }

    sdp_config = {
        "name": "scaled_dot_product",
        "dropout": 0.0,
    }

    a = torch.rand(b, s, d)

    def test_att_mask_ignored():
        # If an sxs attention mask is passed in, it should be ignored.
        # Results should be the same as if no mask was passed in.
        nystrom_attention = NystromAttention(**nystrom_config)
        sdp_attention = ScaledDotProduct(**sdp_config)

        key_padding_mask = None
        att_mask = torch.randint(0, 2, (s, s)).to(dtype=torch.bool)
        sdp_mask = maybe_merge_masks(
            att_mask=None,
            key_padding_mask=key_padding_mask,
            batch_size=b // num_heads,
            src_len=s,
            num_heads=num_heads,
        )
        r_nystrom = nystrom_attention(
            a, a, a, att_mask=att_mask, key_padding_mask=key_padding_mask
        )
        r_sdp = sdp_attention(a, a, a, att_mask=sdp_mask)
        assert torch.allclose(r_nystrom, r_sdp, rtol=0.005, atol=1e-2)

    def test_masking():
        # FIXME
        # nystrom_config["causal"] = True
        # sdp_config["causal"] = True

        nystrom_attention = NystromAttention(**nystrom_config)
        sdp_attention = ScaledDotProduct(**sdp_config)

        key_padding_mask = torch.rand((b // num_heads, s)) > 0.1
        att_mask = None
        mask = maybe_merge_masks(
            att_mask,
            key_padding_mask,
            batch_size=b // num_heads,
            src_len=s,
            num_heads=num_heads,
        )
        r_nystrom = nystrom_attention(a, a, a, key_padding_mask=key_padding_mask)
        r_sdp = sdp_attention(a, a, a, att_mask=mask)

        # Not very close, but more so testing functionality.
        assert torch.allclose(
            r_nystrom, r_sdp, rtol=0.1, atol=0.5
        ), f"max diff {torch.max(torch.abs(r_nystrom-r_sdp))}"

        # Error when key padding mask doesn't have expected dimensions.
        key_padding_mask = torch.randint(0, 2, (s, b)).to(dtype=torch.bool)
        with pytest.raises(AssertionError):
            nystrom_attention(a, a, a, key_padding_mask=key_padding_mask)

    test_att_mask_ignored()
    test_masking()
