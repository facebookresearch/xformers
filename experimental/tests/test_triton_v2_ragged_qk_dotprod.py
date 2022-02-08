# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import time

import pytest
import torch
from ragged_inference.garbage_pad_ragged_acts import RaggedActivations
from ragged_inference.seq_kv_cache import scores_via_qk_dotprod
from ragged_inference.test_utils import (
    assert_eq,
    bf16_support,
    make_seq,
    make_seq_arange,
)
from ragged_inference.triton_v2_ragged_qk_dotprod import (
    RaggedQkPidLookupTable,
    ragged_qk_dotprod,
)

_dtypes = [{"device": "cuda", "dtype": torch.float16}]

if bf16_support():
    _dtypes.append({"device": "cuda", "dtype": torch.bfloat16})


@pytest.mark.parametrize("dtype", _dtypes)
def test_ragged_qk_dotprod_single_seq(dtype):
    d_head = 2

    key = RaggedActivations.from_list(
        [
            make_seq(n_ctx=3, value=42, d_model=d_head, dtype=dtype),
        ]
    )
    query = RaggedActivations.from_list(
        [
            make_seq(n_ctx=4, value=55, d_model=d_head, dtype=dtype),
        ]
    )
    torch_scores = scores_via_qk_dotprod(query, key)
    print(f"{torch_scores=}")

    lut = RaggedQkPidLookupTable.from_query_and_key_tokens_per_seq(
        n_ctx_q_per_seq=query.n_ctx_per_seq, n_ctx_k_per_seq=key.n_ctx_per_seq
    )

    scores = ragged_qk_dotprod(query, key, lut)
    assert_eq(torch_scores, scores)


@pytest.mark.parametrize("dtype", _dtypes)
def test_ragged_qk_dotprod_multiple_seqs_lut(dtype):
    d_head = 2

    key = RaggedActivations.from_list(
        [
            make_seq_arange(n_ctx=5, start_value=0, d_head=d_head, dtype=dtype),
            make_seq_arange(n_ctx=2, start_value=5, d_head=d_head, dtype=dtype),
            make_seq_arange(n_ctx=3, start_value=7, d_head=d_head, dtype=dtype),
        ]
    )
    query = RaggedActivations.from_list(
        [
            make_seq_arange(n_ctx=3, start_value=0, d_head=d_head, dtype=dtype),
            make_seq_arange(n_ctx=2, start_value=3, d_head=d_head, dtype=dtype),
            make_seq_arange(n_ctx=2, start_value=5, d_head=d_head, dtype=dtype),
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


@pytest.mark.parametrize("dtype", _dtypes)
def test_ragged_qk_dotprod_multiple_seqs(dtype):
    d_head = 2

    key = RaggedActivations.from_list(
        [
            make_seq_arange(n_ctx=5, start_value=0, d_head=d_head, dtype=dtype),
            make_seq_arange(n_ctx=2, start_value=5, d_head=d_head, dtype=dtype),
            make_seq_arange(n_ctx=3, start_value=7, d_head=d_head, dtype=dtype),
        ]
    )
    query = RaggedActivations.from_list(
        [
            make_seq_arange(n_ctx=3, start_value=0, d_head=d_head, dtype=dtype),
            make_seq_arange(n_ctx=2, start_value=3, d_head=d_head, dtype=dtype),
            make_seq_arange(n_ctx=2, start_value=5, d_head=d_head, dtype=dtype),
        ]
    )

    lut = RaggedQkPidLookupTable.from_query_and_key_tokens_per_seq(
        n_ctx_q_per_seq=query.n_ctx_per_seq,
        n_ctx_k_per_seq=key.n_ctx_per_seq,
    )
    torch_scores = scores_via_qk_dotprod(query, key)
    scores = ragged_qk_dotprod(query, key, lut)

    for seq_idx, (n_ctx_q, n_ctx_k) in enumerate(
        zip(key.n_ctx_per_seq, query.n_ctx_per_seq)
    ):
        print(f"Checking {seq_idx=}")
        assert_eq(
            torch_scores[seq_idx, :n_ctx_q, :n_ctx_k],
            scores[seq_idx, :n_ctx_q, :n_ctx_k],
        )


@pytest.mark.parametrize("dtype", _dtypes)
def test_ragged_qk_dotprod_multiple_seqs_perf(dtype):
    n_q_ctx = 5
    n_seqs = 50
    d_head = 256
    n_k_ctx = 8000
    n_iters = 10

    query = RaggedActivations.from_list(
        [
            make_seq_arange(n_ctx=n_q_ctx, start_value=0, d_head=d_head, dtype=dtype)
            for _ in range(n_seqs)
        ]
    )
    key = RaggedActivations.from_list(
        [
            make_seq_arange(n_ctx=n_k_ctx, start_value=0, d_head=d_head, dtype=dtype)
            for _ in range(n_seqs)
        ]
    )

    lut = RaggedQkPidLookupTable.from_query_and_key_tokens_per_seq(
        n_ctx_q_per_seq=query.n_ctx_per_seq,
        n_ctx_k_per_seq=key.n_ctx_per_seq,
    )

    for _ in range(3):
        out = ragged_qk_dotprod(query, key, lut)  # noqa: F841

    torch.cuda.synchronize()
    started_at = time.time()
    for _ in range(n_iters):
        out = ragged_qk_dotprod(query, key, lut)  # noqa: F841
    torch.cuda.synchronize()

    elapsed_micros = (time.time() - started_at) * 1e6

    bytes_in_keys_per_seq = n_k_ctx * d_head * 2  # 2 from bf16
    bytes_in_keys_total = bytes_in_keys_per_seq * n_seqs
    hbm_bw_bytes_per_gpu = 1555e9  # 1.5TB/s

    # If we just read the bytes directly from memory
    theor_load_micros_per_seq = bytes_in_keys_per_seq / hbm_bw_bytes_per_gpu * 1e6

    expected_micros_per_seq = theor_load_micros_per_seq

    micros_per_seq = elapsed_micros / (n_iters * n_seqs)
    micros_per_mb = elapsed_micros / (n_iters)
    print(
        f"""
# Theoretical
{bytes_in_keys_total/1e9=:.3f}GB
{bytes_in_keys_per_seq/1e6=:.2f}MB
{theor_load_micros_per_seq=:.1f}µs per seq (to just load once from memory)
{expected_micros_per_seq=:.1f}µs per seq

# Actual
{micros_per_seq=:.1f}µs per seq
{micros_per_mb=:.1f}µs per seq

{micros_per_seq/expected_micros_per_seq:.1f}x the expected HBM-bandwidth bound time
"""
    )

    # FIXME: write a proper, device agnostic test
    if "A100" in torch.cuda.get_device_name(0):
        assert micros_per_seq / expected_micros_per_seq < 1.5


"""
pytest -vxs --tb=native tests/ragged_inference/test_triton_v2_ragged_qk_dotprod.py -k test_ragged_qk_dotprod_multiple_seqs_perf # noqa
"""
