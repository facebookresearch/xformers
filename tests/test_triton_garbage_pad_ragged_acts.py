# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import torch

from xformers.helpers.test_utils import assert_eq, bf16_cuda
from xformers.triton.garbage_pad_ragged_acts import RaggedActivations, add


def _make_seq(n_ctx: int, value: int, d_model: int):
    return torch.full([n_ctx, d_model], value, **bf16_cuda())


def test_garbage_pad_active_queries_correctness():
    d_model = 1
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


def test_triton_garbage_pad_active_queries_correctness():
    d_model = 256
    seqs = [
        _make_seq(n_ctx=1, value=33, d_model=d_model),
        _make_seq(n_ctx=3, value=42, d_model=d_model),
        _make_seq(n_ctx=7, value=55, d_model=d_model),
    ]
    active_queries = RaggedActivations.from_list(seqs)
    padded_queries = active_queries.triton_to_garbage_padded()

    # Check that the non-garbage portion of each is correct
    assert_eq(padded_queries[0, :1, :], seqs[0])
    assert_eq(padded_queries[1, :3, :], seqs[1])
    assert_eq(padded_queries[2, :7, :], seqs[2])


def test_add_kernel():
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")
    output_torch = x + y
    output_triton = add(x, y)
    print(output_torch)
    print(output_triton)
    print(
        f"The maximum difference between torch and triton is "
        f"{torch.max(torch.abs(output_torch - output_triton))}"
    )


"""
pytest -vsx --tb=native tests/test_triton_garbage_pad_ragged_acts.py \
    -k test_triton_garbage_pad_active_queries_correctness
"""
