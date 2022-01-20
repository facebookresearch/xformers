# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch

from xformers.helpers.test_utils import assert_eq, bf16_cuda
from xformers.triton.garbage_pad_ragged_acts import RaggedActivations
from xformers.triton.ragged_inference.seq_kv_cache import (
    _single_seq_kv_cache,
    scores_via_qk_dotprod,
)
from xformers.triton.ragged_inference.triton_v2_qk_dotprod import qk_dotprod
from xformers.triton.ragged_inference.triton_v2_ragged_qk_dotprod import (
    RaggedQkPidLookupTable,
    ragged_qk_dotprod,
)


def _make_seq(n_ctx: int, value: int, d_model: int):
    return torch.full([n_ctx, d_model], value, **bf16_cuda())


def _make_seq_arange(n_ctx: int, start_value: int, d_model: int):
    return (
        torch.full([n_ctx, d_model], start_value, **bf16_cuda())
        + torch.arange(n_ctx, **bf16_cuda())[:, None]
    )


def test_ragged_qk_dotprod_single_seq():
    d_model = 2

    key = RaggedActivations.from_list(
        [
            _make_seq(n_ctx=3, value=42, d_model=d_model),
        ]
    )
    query = RaggedActivations.from_list(
        [
            _make_seq(n_ctx=4, value=55, d_model=d_model),
        ]
    )
    torch_scores = scores_via_qk_dotprod(query, key)
    print(f"{torch_scores=}")

    lut = RaggedQkPidLookupTable.from_query_and_key_tokens_per_seq(
        n_ctx_q_per_seq=query.n_ctx_per_seq, n_ctx_k_per_seq=key.n_ctx_per_seq
    )

    scores = ragged_qk_dotprod(query, key, lut)
    assert_eq(torch_scores, scores)


def test_ragged_qk_dotprod_multiple_seqs_lut():
    d_model = 2

    key = RaggedActivations.from_list(
        [
            _make_seq_arange(n_ctx=5, start_value=0, d_model=d_model),
            _make_seq_arange(n_ctx=2, start_value=5, d_model=d_model),
            _make_seq_arange(n_ctx=3, start_value=7, d_model=d_model),
        ]
    )
    query = RaggedActivations.from_list(
        [
            _make_seq_arange(n_ctx=3, start_value=0, d_model=d_model),
            _make_seq_arange(n_ctx=2, start_value=3, d_model=d_model),
            _make_seq_arange(n_ctx=2, start_value=5, d_model=d_model),
        ]
    )

    lut = RaggedQkPidLookupTable.from_query_and_key_tokens_per_seq(
        n_ctx_q_per_seq=query.n_ctx_per_seq,
        n_ctx_k_per_seq=key.n_ctx_per_seq,
        block_q_override=2,
        block_k_override=2,
    )
    assert_eq(lut.pid_to_in_q_token_offset.cpu(), [0, 0, 0, 2, 2, 2, 3, 5, 5])
    assert_eq(lut.pid_to_in_k_token_offset.cpu(), [0, 2, 4, 0, 2, 4, 5, 7, 9])
    assert_eq(lut.pid_to_out_q_block.cpu(), [0, 0, 0, 1, 1, 1, 0, 0, 0])
    assert_eq(lut.pid_to_out_k_block.cpu(), [0, 1, 2, 0, 1, 2, 0, 0, 1])
    assert_eq(lut.pid_to_out_seq_idx.cpu(), [0, 0, 0, 0, 0, 0, 1, 2, 2])
    assert_eq(lut.n_pids_total, 9)

def test_ragged_qk_dotprod_multiple_seqs():
    d_model = 2

    key = RaggedActivations.from_list(
        [
            _make_seq_arange(n_ctx=5, start_value=0, d_model=d_model),
            _make_seq_arange(n_ctx=2, start_value=5, d_model=d_model),
            _make_seq_arange(n_ctx=3, start_value=7, d_model=d_model),
        ]
    )
    query = RaggedActivations.from_list(
        [
            _make_seq_arange(n_ctx=3, start_value=0, d_model=d_model),
            _make_seq_arange(n_ctx=2, start_value=3, d_model=d_model),
            _make_seq_arange(n_ctx=2, start_value=5, d_model=d_model),
        ]
    )

    lut = RaggedQkPidLookupTable.from_query_and_key_tokens_per_seq(
        n_ctx_q_per_seq=query.n_ctx_per_seq,
        n_ctx_k_per_seq=key.n_ctx_per_seq,
    )
    torch_scores = scores_via_qk_dotprod(query, key)
    scores = ragged_qk_dotprod(query, key, lut)

    for seq_idx, (n_ctx_q, n_ctx_k) in enumerate(zip(key.n_ctx_per_seq, query.n_ctx_per_seq)):
        print(f'Checking {seq_idx=}')
        assert_eq(torch_scores[seq_idx, :n_ctx_q, :n_ctx_k], scores[seq_idx, :n_ctx_q, :n_ctx_k])


"""
pytest -vxs --tb=native tests/ragged_inference/test_triton_v2_ragged_qk_dotprod.py
"""
