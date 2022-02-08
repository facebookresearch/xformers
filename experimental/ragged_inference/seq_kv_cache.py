# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from functools import lru_cache
from typing import List, Tuple

import torch
from ragged_inference.garbage_pad_ragged_acts import RaggedActivations


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


def calculate_scores_via_qk_dotprod(
    seq_kv_cache: List[SingleSeqKVCache],  # These have already been extended
    active_queries: RaggedActivations,
) -> torch.Tensor:
    padded_keys = garbage_pad_keys(seq_kv_cache)
    padded_active_queries = active_queries.to_garbage_padded()
    return torch.einsum("bkd,bqd->bqk", padded_keys, padded_active_queries)


def scores_via_qk_dotprod(
    query: RaggedActivations,
    key: RaggedActivations,
) -> torch.Tensor:
    padded_query = query.to_garbage_padded()
    padded_key = key.to_garbage_padded()
    return torch.einsum("bkd,bqd->bqk", padded_key, padded_query)
