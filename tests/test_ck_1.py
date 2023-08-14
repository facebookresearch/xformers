# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import random

import pytest
import torch

from xformers.ops.common import get_xformers_operator

B = 7
M = 1000
N = 1000
H = 13
K = 64
Kv = 64

_types = [torch.float16, torch.bfloat16]

@pytest.mark.parametrize("test_type", _types)
def test_types(test_type):
    query = torch.rand((B, M, H, K), device=torch.device("cuda"), dtype=test_type)
    key = torch.rand((B, N, H, K), device=torch.device("cuda"), dtype=test_type)
    val = torch.rand((B, N, H, Kv), device=torch.device("cuda"), dtype=test_type)

    Operator=get_xformers_operator("efficient_attention_forward_ck")

    out, lse, rng_seed, rng_offset = Operator(query=query, key=key, value=val, attn_bias=None, seqstart_q=None, seqstart_k=None, dropout_p=0.0, compute_logsumexp=False, custom_mask_type=0, scale=None, seqlen_k=None)

    print(rng_seed)

