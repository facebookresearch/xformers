# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import time

import pytest
import torch
from ragged_inference.garbage_pad_ragged_acts import RaggedActivations
from ragged_inference.seq_kv_cache import (
    SingleSeqKVCache,
    _create_indices,
    calculate_scores_via_qk_dotprod,
    extend_kv_caches,
    garbage_pad_keys,
    garbage_pad_seq_kv_cache,
)
from ragged_inference.test_utils import assert_eq, bf16_support

_dtypes = [{"device": "cuda", "dtype": torch.float16}]

if bf16_support():
    _dtypes.append({"device": "cuda", "dtype": torch.bfloat16})


def _single_seq_kv_cache(n_ctx, value, d_model, dtype) -> SingleSeqKVCache:
    return SingleSeqKVCache(
        keys=torch.full([n_ctx, d_model], value, **dtype),
        values=torch.full([n_ctx, d_model], value, **dtype),
    )


@pytest.mark.parametrize("dtype", _dtypes)
def test_garbage_pad_seq_kv_cache_correctness(dtype):
    seq_kv_cache = [
        _single_seq_kv_cache(n_ctx=1, value=33, d_model=2, dtype=dtype),
        _single_seq_kv_cache(n_ctx=3, value=42, d_model=2, dtype=dtype),
        _single_seq_kv_cache(n_ctx=7, value=55, d_model=2, dtype=dtype),
    ]

    padded_keys, padded_values = garbage_pad_seq_kv_cache(seq_kv_cache)

    # Check that the non-garbage portion of each is correct
    assert_eq(padded_keys[0, :1, :], seq_kv_cache[0].keys)
    assert_eq(padded_keys[1, :3, :], seq_kv_cache[1].keys)
    assert_eq(padded_keys[2, :7, :], seq_kv_cache[2].keys)

    assert_eq(padded_values[0, :1, :], seq_kv_cache[0].values)
    assert_eq(padded_values[1, :3, :], seq_kv_cache[1].values)
    assert_eq(padded_values[2, :7, :], seq_kv_cache[2].values)


@pytest.mark.parametrize("dtype", _dtypes)
def test_extend_kv_caches_correctness(dtype):
    d_model = 6
    seq_kv_cache = [
        _single_seq_kv_cache(n_ctx=1, value=33, d_model=d_model, dtype=dtype),
        _single_seq_kv_cache(n_ctx=3, value=42, d_model=d_model, dtype=dtype),
        _single_seq_kv_cache(n_ctx=7, value=55, d_model=d_model, dtype=dtype),
    ]

    n_ctx_new = 1
    active_keys = RaggedActivations.from_list(
        [
            torch.ones(n_ctx_new, d_model, **dtype),
            torch.ones(n_ctx_new, d_model, **dtype),
            torch.ones(n_ctx_new, d_model, **dtype),
        ]
    )
    active_values = RaggedActivations.from_list(
        [
            torch.ones(n_ctx_new, d_model, **dtype) * 2,
            torch.ones(n_ctx_new, d_model, **dtype) * 2,
            torch.ones(n_ctx_new, d_model, **dtype) * 2,
        ]
    )

    new_cache = extend_kv_caches(seq_kv_cache, active_keys, active_values)

    assert_eq(new_cache[0].keys[:, 0].cpu(), [33, 1])
    assert_eq(new_cache[0].values[:, 0].cpu(), [33, 2])

    assert_eq(new_cache[1].keys[:, 0].cpu(), [42, 42, 42, 1])
    assert_eq(new_cache[1].values[:, 0].cpu(), [42, 42, 42, 2])

    assert_eq(new_cache[2].keys[:, 0].cpu(), [55, 55, 55, 55, 55, 55, 55, 1])
    assert_eq(new_cache[2].values[:, 0].cpu(), [55, 55, 55, 55, 55, 55, 55, 2])


@pytest.mark.parametrize("dtype", _dtypes)
def test_index_select_throughput(dtype):
    n_ctx_per_seq = 8192
    n_seqs = 20
    d_model_per_gpu = 12 * 1024 // 8

    keys = _single_seq_kv_cache(
        n_ctx=n_ctx_per_seq * n_seqs, value=42, d_model=d_model_per_gpu, dtype=dtype
    ).keys

    indices = _create_indices(tuple(n_ctx_per_seq for _ in range(n_seqs)))

    for strategy in ["index_select", "gather", "slice"]:
        if strategy == "slice":

            def do_the_op():
                return keys[indices, :]

        elif strategy == "gather":
            stacked_idxs = torch.stack([indices for _ in range(d_model_per_gpu)], dim=1)

            def do_the_op():
                torch.gather(input=keys, dim=0, index=stacked_idxs)

        elif strategy == "index_select":

            def do_the_op():
                torch.index_select(input=keys, dim=0, index=indices)

        else:
            raise ValueError(f"{strategy=}")

        # warmup
        do_the_op()

        torch.cuda.synchronize()
        started_at = time.time()
        n_iters = 10
        for _ in range(n_iters):
            do_the_op()

        torch.cuda.synchronize()
        elapsed_micros = (time.time() - started_at) * 1e6
        micros_per_mb = elapsed_micros / n_iters
        micros_per_seq = micros_per_mb / n_seqs
        print(
            f"""
# Speed when {strategy=}
{micros_per_seq=:.1f}µs per seq
        """
        )


@pytest.mark.parametrize("dtype", _dtypes)
def test_garbage_pad_keys_throughput(dtype, n_ctx_per_seq=1024):
    n_seqs = 100
    d_model_per_gpu = 12 * 1024 // 8
    seq_kv_cache = [
        _single_seq_kv_cache(
            n_ctx=n_ctx_per_seq, value=42, d_model=d_model_per_gpu, dtype=dtype
        )
        for _ in range(n_seqs)
    ]

    bytes_in_keys_per_seq = n_ctx_per_seq * d_model_per_gpu * 2  # 2 from bf16
    bytes_in_keys_total = bytes_in_keys_per_seq * n_seqs
    hbm_bw_bytes_per_gpu = 1555e9  # 1.5TB/s

    # If we just read the bytes directly from memory
    theor_load_micros_per_seq = bytes_in_keys_per_seq / hbm_bw_bytes_per_gpu * 1e6

    # Doing our operation should be slower than the theoretical minimum because we
    # do the following to the items
    #
    # 1. Read them from the per-seq areas
    # 2. Write them back into the buffer
    expected_micros_per_seq = theor_load_micros_per_seq * 2

    # warmup
    garbage_pad_keys(seq_kv_cache)

    torch.cuda.synchronize()
    started_at = time.time()
    n_iters = 10
    for _ in range(n_iters):
        garbage_pad_keys(seq_kv_cache)

    torch.cuda.synchronize()
    elapsed_micros = (time.time() - started_at) * 1e6

    micros_per_mb = elapsed_micros / n_iters
    micros_per_seq = micros_per_mb / n_seqs
    print(
        f"""
# Theoretical
{bytes_in_keys_total/1e9=:.3f}GB
{bytes_in_keys_per_seq/1e6=:.2f}MB
{theor_load_micros_per_seq=:.1f}µs per seq (to just load once from memory)
{expected_micros_per_seq=:.1f}µs per seq

# Actual
{micros_per_mb=:.1f}µs per microbatch
{micros_per_seq=:.1f}µs per seq

{micros_per_seq/expected_micros_per_seq:.1f}x the expected HBM-bandwidth bound time
"""
    )


@pytest.mark.parametrize("dtype", _dtypes)
def test_garbage_pad_active_queries_throughput(dtype, n_active_ctx_per_seq=5):
    n_seqs = 100
    d_model_per_gpu = 12 * 1024 // 8
    active_queries = RaggedActivations.from_list(
        [
            torch.ones(n_active_ctx_per_seq, d_model_per_gpu, **dtype) * 2
            for _ in range(n_seqs)
        ]
    )

    bytes_in_queries_per_seq = n_active_ctx_per_seq * d_model_per_gpu * 2  # 2 from bf16
    bytes_in_queries_total = bytes_in_queries_per_seq * n_seqs
    hbm_bw_bytes_per_gpu = 1555e9  # 1.5TB/s

    # If we just read the bytes directly from memory
    theor_load_micros_per_seq = bytes_in_queries_per_seq / hbm_bw_bytes_per_gpu * 1e6

    # Doing our operation should be slower than the theoretical minimum because we
    # do the following to the items
    #
    # 1. Read them from the per-seq areas
    # 2. Write them back into the buffer
    expected_micros_per_seq = theor_load_micros_per_seq * 2

    # warmup
    active_queries.to_garbage_padded()

    torch.cuda.synchronize()
    started_at = time.time()
    n_iters = 10
    for _ in range(n_iters):
        active_queries.to_garbage_padded()

    torch.cuda.synchronize()
    elapsed_micros = (time.time() - started_at) * 1e6

    micros_per_mb = elapsed_micros / n_iters
    micros_per_seq = micros_per_mb / n_seqs
    print(
        f"""
# Theoretical
{bytes_in_queries_total/1e9=:.3f}GB
{bytes_in_queries_per_seq/1e6=:.2f}MB
{theor_load_micros_per_seq=:.1f}µs per seq (to just load once from memory)
{expected_micros_per_seq=:.1f}µs per seq

# Actual
{micros_per_mb=:.1f}µs per microbatch
{micros_per_seq=:.1f}µs per seq

{micros_per_seq/expected_micros_per_seq:.1f}x the expected HBM-bandwidth bound time
"""
    )


@pytest.mark.parametrize("dtype", _dtypes)
def test_calculate_scores_via_qk_dotprod_throughput(
    dtype, n_key_ctx_per_seq=1024, n_active_query_ctx_per_seq=5
):
    n_seqs = 100
    d_model_per_gpu = 12 * 1024 // 8
    seq_kv_cache = [
        _single_seq_kv_cache(
            n_ctx=n_key_ctx_per_seq, value=42, d_model=d_model_per_gpu, dtype=dtype
        )
        for _ in range(n_seqs)
    ]

    active_queries = RaggedActivations.from_list(
        [
            torch.ones(n_active_query_ctx_per_seq, d_model_per_gpu, **dtype) * 2
            for _ in range(n_seqs)
        ]
    )
    assert n_key_ctx_per_seq > n_active_query_ctx_per_seq * 10, (
        "n_active_query_ctx_per_seq must be much larger than "
        "n_key_ctx_per_seq for our simulator to be useful because "
        "we round the HBM memory bandwidth for the active_queries and "
        "for the scores down to zero"
    )

    bytes_in_keys_per_seq = n_key_ctx_per_seq * d_model_per_gpu * 2  # 2 from bf16
    bytes_in_keys_total = bytes_in_keys_per_seq * n_seqs
    hbm_bw_bytes_per_gpu = 1555e9  # 1.5TB/s

    # If we just read the bytes directly from memory
    theor_load_micros_per_seq = bytes_in_keys_per_seq / hbm_bw_bytes_per_gpu * 1e6

    # Doing our operation should be slower than the theoretical minimum because we
    # do the following to the items
    #
    # 1. Read them from the per-seq areas
    # 2. Write them back into the buffer
    expected_micros_per_seq = theor_load_micros_per_seq * 2

    # warmup
    calculate_scores_via_qk_dotprod(seq_kv_cache, active_queries)

    torch.cuda.synchronize()
    started_at = time.time()
    n_iters = 10
    for _ in range(n_iters):
        calculate_scores_via_qk_dotprod(seq_kv_cache, active_queries)

    torch.cuda.synchronize()
    elapsed_micros = (time.time() - started_at) * 1e6

    micros_per_mb = elapsed_micros / n_iters
    micros_per_seq = micros_per_mb / n_seqs
    print(
        f"""
# Theoretical
{bytes_in_keys_total/1e9=:.3f}GB
{bytes_in_keys_per_seq/1e6=:.2f}MB
{theor_load_micros_per_seq=:.1f}µs per seq (to just load once from memory)
{expected_micros_per_seq=:.1f}µs per seq

# Actual
{micros_per_mb=:.1f}µs per microbatch
{micros_per_seq=:.1f}µs per seq

{micros_per_seq/expected_micros_per_seq:.1f}x the expected HBM-bandwidth bound time
"""
    )


"""
# Run tests with the following
pytest -vsx tests/ragged_inference/test_seq_kv_cache.py


# Profile with the following
pytest -vsx tests/ragged_inference/test_seq_kv_cache.py -k test_calculate_scores_via_qk_dotprod_throughput

"""
