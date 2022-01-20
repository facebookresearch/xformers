# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch

from xformers.helpers.test_utils import assert_eq, bf16_cuda
from xformers.triton.garbage_pad_ragged_acts import RaggedActivations
from xformers.triton.ragged_inference.seq_kv_cache import _single_seq_kv_cache, \
    scores_via_qk_dotprod
from xformers.triton.ragged_inference.triton_v2_qk_dotprod import qk_dotprod
from xformers.triton.ragged_inference.triton_v2_ragged_qk_dotprod import qk_dotprod_v2


def _make_seq(n_ctx: int, value: int, d_model: int):
    return torch.full([n_ctx, d_model], value, **bf16_cuda())

def test_ragged_qk_dotprod():
    d_model=2



    key = RaggedActivations.from_list(
        [
            _make_seq(n_ctx=4, value=42, d_model=d_model),
        ]
    )
    query = RaggedActivations.from_list(
        [
            _make_seq(n_ctx=3, value=55, d_model=d_model),
        ]
    )
    scores = scores_via_qk_dotprod(query, key)
    print(scores)

"""
pytest -vxs --tb=native tests/ragged_inference/test_triton_v2_ragged_qk_dotprod.py
"""
