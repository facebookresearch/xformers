# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch

from xformers.helpers.test_utils import assert_eq, bf16_cuda
from xformers.triton.garbage_pad_ragged_acts import RaggedActivations


def _make_seq(n_ctx: int, value: int, d_model: int):
    return torch.full([n_ctx, d_model], value, **bf16_cuda())


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test requires a CUDA device"
)
def test_garbage_pad_active_queries_correctness():
    d_model = 6
    seqs = [
        _make_seq(n_ctx=1, value=33, d_model=d_model),
        _make_seq(n_ctx=3, value=42, d_model=d_model),
        _make_seq(n_ctx=7, value=55, d_model=d_model),
    ]
    active_queries = RaggedActivations.from_list(seqs)
    padded_queries = active_queries.to_garbage_padded()

    # Check that the non-garbage portion of each is correct
    assert_eq(padded_queries[0, :1, :], seqs[0])
    assert_eq(padded_queries[1, :3, :], seqs[1])
    assert_eq(padded_queries[2, :7, :], seqs[2])


def test_add_kernel():
    pass
