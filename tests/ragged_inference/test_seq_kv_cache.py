# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import time
from functools import lru_cache
from typing import List, Tuple

import pytest
import torch

from xformers.helpers.test_utils import assert_eq


def bf16_cuda():
    return dict(device="cuda", dtype=torch.bfloat16)


class RaggedActivations:
    def __init__(self, raw_tensor: torch.Tensor, n_ctx_per_seq: List[int]):
        self.raw_tensor = raw_tensor
        self.n_ctx_per_seq = n_ctx_per_seq

    @classmethod
    def from_list(cls, tensors: List[torch.Tensor]):
        """Tensors must all be of shape [n_ctx, d_model]."""
        return cls(
            raw_tensor=torch.cat(tensors),
            n_ctx_per_seq=[tensor.shape[0] for tensor in tensors],
        )

    def iter_full_tensors(self):
        idx_so_far = 0
        for n_ctx_in_this_seq in self.n_ctx_per_seq:
            yield self.raw_tensor[idx_so_far : idx_so_far + n_ctx_in_this_seq]
            idx_so_far += n_ctx_in_this_seq

    def to_garbage_padded(self) -> torch.Tensor:
        """
        Create a tensor of shape (n_seqs, n_ctx_max, d_model) where the
        sequences are right-padded with garbage data
        """
        n_seqs = len(self.n_ctx_per_seq)
        n_ctx_max = max(self.n_ctx_per_seq)

        n_dim = self.raw_tensor.shape[-1]
        padded_acts = torch.empty(
            n_seqs, n_ctx_max, n_dim, dtype=self.raw_tensor.dtype, device="cuda"
        )

        idx_so_far = 0
        for seq_idx, n_ctx_in_this_seq in enumerate(self.n_ctx_per_seq):
            this_seq = self.raw_tensor[idx_so_far : idx_so_far + n_ctx_in_this_seq]
            idx_so_far += n_ctx_in_this_seq
            padded_acts[seq_idx, :n_ctx_in_this_seq, :] = this_seq
        return padded_acts


class SingleSeqKVCache:
    def __init__(self, keys: torch.Tensor, values: torch.Tensor):
        # Tensor of shape [2, n_ctx, d_model_per_gpu]
        # - keys are cache[0]
        # - values are cache[1]
        self.raw_keys = keys
        self.raw_values = values

    @property
    def keys(self) -> torch.Tensor:
        return self.raw_keys

    @property
    def values(self) -> torch.Tensor:
        return self.raw_values

    @property
    def n_ctx(self):
        return self.raw_values.shape[0]

    @property
    def d_model_per_gpu(self):
        return self.raw_values.shape[-1]

    @property
    def is_cuda(self):
        return self.raw_values.is_cuda

    @property
    def dtype(self):
        return self.raw_values.dtype


def _single_seq_kv_cache(n_ctx, value, d_model) -> SingleSeqKVCache:
    return SingleSeqKVCache(
        keys=torch.full([n_ctx, d_model], value, **bf16_cuda()),
        values=torch.full([n_ctx, d_model], value, **bf16_cuda()),
    )


def extend_kv_caches(
    seq_kv_cache: List[SingleSeqKVCache],
    active_keys: RaggedActivations,
    active_values: RaggedActivations,
) -> List[SingleSeqKVCache]:
    assert seq_kv_cache[0].is_cuda

    updated_seq_kv_cache = []
    for cache, keys, values in zip(
        seq_kv_cache, active_keys.iter_full_tensors(), active_values.iter_full_tensors()
    ):

        # Dim 1 is the context
        new_cache = SingleSeqKVCache(
            keys=torch.cat([cache.keys, keys], dim=0),
            values=torch.cat([cache.values, values], dim=0),
        )
        updated_seq_kv_cache.append(new_cache)

    return updated_seq_kv_cache


def garbage_pad_seq_kv_cache(
    seq_kv_cache: List[SingleSeqKVCache],
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert seq_kv_cache[0].is_cuda
    dtype = seq_kv_cache[0].dtype
    n_ctx_per_kv_cache = [seq.n_ctx for seq in seq_kv_cache]

    # Create a view so that the output is (n_seqs, n_ctx_max, d_model)
    # This should not incur an extra memcopy
    n_seqs = len(n_ctx_per_kv_cache)
    n_ctx_max = max(n_ctx_per_kv_cache)

    padded_keys = torch.empty(
        n_seqs,
        n_ctx_max,
        seq_kv_cache[0].d_model_per_gpu,
        dtype=dtype,
        device="cuda",
    )

    padded_values = torch.empty(
        n_seqs,
        n_ctx_max,
        seq_kv_cache[0].d_model_per_gpu,
        dtype=dtype,
        device="cuda",
    )

    for seq_idx, seq in enumerate(seq_kv_cache):
        padded_keys[seq_idx, : seq.n_ctx, :] = seq.keys
        padded_values[seq_idx, : seq.n_ctx, :] = seq.values
    return (padded_keys, padded_values)


def garbage_pad_keys(
    seq_kv_cache: List[SingleSeqKVCache],
) -> torch.Tensor:
    assert seq_kv_cache[0].is_cuda
    dtype = seq_kv_cache[0].dtype
    n_ctx_per_kv_cache = [seq.n_ctx for seq in seq_kv_cache]

    # Create a view so that the output is (n_seqs, n_ctx_max, d_model)
    # This should not incur an extra memcopy
    n_seqs = len(n_ctx_per_kv_cache)
    n_ctx_max = max(n_ctx_per_kv_cache)

    padded_keys = torch.empty(
        n_seqs,
        n_ctx_max,
        seq_kv_cache[0].d_model_per_gpu,
        dtype=dtype,
        device="cuda",
    )

    for seq_idx, seq in enumerate(seq_kv_cache):
        padded_keys[seq_idx, : seq.n_ctx, :] = seq.keys
    return padded_keys


@lru_cache(maxsize=1)  # Memoize because we repeat this for consecutive resblocks
def _create_indices(n_ctx_per_kv_cache):
    """
    We cache this because it requires some substantial CPU work and it's done multiple
    times sequentially (once per resblock)
    """
    indices_list = []
    ragged_idx = 0
    max_n_ctx = max(n_ctx_per_kv_cache)
    for n_ctx in n_ctx_per_kv_cache:
        for idx_into_seq in range(max_n_ctx):
            if idx_into_seq < n_ctx:
                indices_list.append(ragged_idx)
                ragged_idx += 1
            else:
                indices_list.append(0)  # Add a placeholder
    return torch.tensor(indices_list, device="cuda")


def test_garbage_pad_seq_kv_cache_correctness():
    seq_kv_cache = [
        _single_seq_kv_cache(n_ctx=1, value=33, d_model=2),
        _single_seq_kv_cache(n_ctx=3, value=42, d_model=2),
        _single_seq_kv_cache(n_ctx=7, value=55, d_model=2),
    ]

    padded_keys, padded_values = garbage_pad_seq_kv_cache(seq_kv_cache)

    # Check that the non-garbage portion of each is correct
    assert_eq(padded_keys[0, :1, :], seq_kv_cache[0].keys)
    assert_eq(padded_keys[1, :3, :], seq_kv_cache[1].keys)
    assert_eq(padded_keys[2, :7, :], seq_kv_cache[2].keys)

    assert_eq(padded_values[0, :1, :], seq_kv_cache[0].values)
    assert_eq(padded_values[1, :3, :], seq_kv_cache[1].values)
    assert_eq(padded_values[2, :7, :], seq_kv_cache[2].values)


def test_extend_kv_caches_correctness():
    d_model = 6
    seq_kv_cache = [
        _single_seq_kv_cache(n_ctx=1, value=33, d_model=d_model),
        _single_seq_kv_cache(n_ctx=3, value=42, d_model=d_model),
        _single_seq_kv_cache(n_ctx=7, value=55, d_model=d_model),
    ]

    n_ctx_new = 1
    active_keys = RaggedActivations.from_list(
        [
            torch.ones(n_ctx_new, d_model, **bf16_cuda()),
            torch.ones(n_ctx_new, d_model, **bf16_cuda()),
            torch.ones(n_ctx_new, d_model, **bf16_cuda()),
        ]
    )
    active_values = RaggedActivations.from_list(
        [
            torch.ones(n_ctx_new, d_model, **bf16_cuda()) * 2,
            torch.ones(n_ctx_new, d_model, **bf16_cuda()) * 2,
            torch.ones(n_ctx_new, d_model, **bf16_cuda()) * 2,
        ]
    )

    new_cache = extend_kv_caches(seq_kv_cache, active_keys, active_values)

    assert_eq(new_cache[0].keys[:, 0].cpu(), [33, 1])
    assert_eq(new_cache[0].values[:, 0].cpu(), [33, 2])

    assert_eq(new_cache[1].keys[:, 0].cpu(), [42, 42, 42, 1])
    assert_eq(new_cache[1].values[:, 0].cpu(), [42, 42, 42, 2])

    assert_eq(new_cache[2].keys[:, 0].cpu(), [55, 55, 55, 55, 55, 55, 55, 1])
    assert_eq(new_cache[2].values[:, 0].cpu(), [55, 55, 55, 55, 55, 55, 55, 2])


def test_index_select_throughput():
    n_ctx_per_seq = 8192
    n_seqs = 100
    d_model_per_gpu = 12 * 1024 // 8

    keys = _single_seq_kv_cache(
        n_ctx=n_ctx_per_seq * n_seqs, value=42, d_model=d_model_per_gpu
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


def test_garbage_pad_keys_throughput(n_ctx_per_seq=1024):
    n_seqs = 100
    d_model_per_gpu = 12 * 1024 // 8
    seq_kv_cache = [
        _single_seq_kv_cache(n_ctx=n_ctx_per_seq, value=42, d_model=d_model_per_gpu)
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


def test_garbage_pad_active_queries_throughput(n_active_ctx_per_seq=5):
    n_seqs = 100
    d_model_per_gpu = 12 * 1024 // 8
    active_queries = RaggedActivations.from_list(
        [
            torch.ones(n_active_ctx_per_seq, d_model_per_gpu, **bf16_cuda()) * 2
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


def calculate_scores_via_qk_dotprod(
    seq_kv_cache: List[SingleSeqKVCache],  # These have already been extended
    active_queries: RaggedActivations,
) -> torch.Tensor:
    padded_keys = garbage_pad_keys(seq_kv_cache)
    padded_active_queries = active_queries.to_garbage_padded()

    return torch.einsum("bkd,bqd->bkq", padded_keys, padded_active_queries)


def test_calculate_scores_via_qk_dotprod_throughput(
    n_key_ctx_per_seq=1024, n_active_query_ctx_per_seq=5
):
    n_seqs = 100
    d_model_per_gpu = 12 * 1024 // 8
    seq_kv_cache = [
        _single_seq_kv_cache(n_ctx=n_key_ctx_per_seq, value=42, d_model=d_model_per_gpu)
        for _ in range(n_seqs)
    ]

    active_queries = RaggedActivations.from_list(
        [
            torch.ones(n_active_query_ctx_per_seq, d_model_per_gpu, **bf16_cuda()) * 2
            for _ in range(n_seqs)
        ]
    )
    assert (
        n_key_ctx_per_seq > n_active_query_ctx_per_seq * 10
    ), f"n_active_query_ctx_per_seq must be much larger than n_key_ctx_per_seq for our simulator to be useful because we round the HBM memory bandwidth for the active_queries and for the scores down to zero"

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
