# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch

from xformers.components.attention import GlobalAttention, ScaledDotProduct


def test_global_attention():
    b, s, d = 2, 90, 40

    torch.cuda.manual_seed(42)
    torch.manual_seed(42)

    def test_ratio(global_attention_ratio: float):
        # Make sure that Global and Normal attention get the same results for the corresponding tokens
        a = torch.rand(b, s, d)
        config = {
            "name": "global",
            "dropout": 0.0,
            "causal": False,
            "max_seq_len": s,
            "attention_query_mask": torch.rand((s, 1)) < global_attention_ratio,
        }

        global_attention = GlobalAttention(**config)
        sdp_attention = ScaledDotProduct(**config)

        r_global = global_attention(a, a, a)
        r_dense = sdp_attention(a, a, a)

        # Check that the tokens which have access to the full attention give the same
        # results as the monolithic dense scaled_dot_product
        mask = config["attention_query_mask"][:, 0]
        assert torch.allclose(r_global[:, mask, :], r_dense[:, mask, :])

    # Test with different levels of sparsity, to make sure that all the paths are covered
    test_ratio(0.02)
    test_ratio(0.5)
    test_ratio(1.0)  # All queries allowed
