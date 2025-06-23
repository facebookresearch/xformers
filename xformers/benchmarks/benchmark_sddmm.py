# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import torch
from torch.utils import benchmark

from xformers.components.attention._sputnik_sparse import _csr_to_coo
from xformers.components.attention.core import _create_random_sparsity, SparseCS

MIN_RUN_TIME = 0.2


def _get_fn(backend):
    if backend == "csr_ge":
        fn = torch.ops.xformers.csr_sddmm
    elif backend == "csr_sputnik":
        fn = torch.ops.xformers.sddmm_sputnik
    elif backend == "coo_ge":

        def fn(a, b, row_indices, row_offsets, column_indices):
            row_coo, _ = _csr_to_coo(
                a.shape[-2], b.shape[-2], row_offsets, column_indices
            )
            return torch.ops.xformers.coo_sddmm(
                a, b, row_indices, row_coo, column_indices
            )

    elif backend == "csr_to_coo":

        def fn(a, b, row_indices, row_offsets, column_indices):
            row_coo, _ = _csr_to_coo(
                a.shape[-2], b.shape[-2], row_offsets, column_indices
            )
            return row_coo

    return fn


def bench_sddmm(configs):
    min_run_time = MIN_RUN_TIME

    device = torch.device("cuda")
    results = []

    for (B, M, K), prob in configs:
        a = torch.rand(B, M, K, device=device)
        b = torch.rand(B, M, K, device=device)

        mask = _create_random_sparsity(
            torch.ones(1, M, M, dtype=torch.bool), prob, divisible_by=16
        )
        aa = a
        bb = b
        mask = SparseCS(mask, device)
        row_indices = mask.row_indices
        row_offsets = mask.row_offsets
        column_indices = mask.column_indices

        for backend in ["csr_sputnik", "csr_ge", "coo_ge", "csr_to_coo"]:

            fn_str = "fn(a, b, row_indices, row_offsets, column_indices)"
            fn = _get_fn(backend)

            results.append(
                benchmark.Timer(
                    stmt=fn_str,
                    globals={
                        "a": aa,
                        "b": bb,
                        "mask": mask,
                        "row_indices": row_indices,
                        "row_offsets": row_offsets,
                        "column_indices": column_indices,
                        "fn": fn,
                    },
                    label="sddmm",
                    sub_label=f"B={B:>4d}, M={M:>4d}, K={K:>3d}, prob={prob:0.4f}",
                    description=backend,
                ).blocked_autorange(min_run_time=min_run_time)
            )

    compare = benchmark.Compare(results)
    compare.print()
    return results


# batch size 32, for different layers
SWIN_T_SIZES = [(96, 3136, 32), (192, 784, 32), (384, 196, 32), (768, 49, 32)]
swin_t_config = list(zip(SWIN_T_SIZES, (0.9844, 0.9375, 0.75, 0.0)))

# some random values
BASIC_SIZES = [(32, 1024, 32), (32, 1024, 128), (8, 4096, 32), (8, 4096, 128)]
SPARSITIES = [0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]
basic_config = list(itertools.product(BASIC_SIZES, SPARSITIES))

# batch size 32 here
vit_sizes = [
    (192, 785, 64),  # deit_small_patch8_224
    (192, 197, 64),  # deit_small_patch16_224
    (384, 785, 64),  # deit_base_patch8_224
    (384, 197, 64),  # deit_base_patch16_224
]
SPARSITIES = [0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97]
vit_config = list(itertools.product(vit_sizes, SPARSITIES))

results = []

if torch.version.hip:
    print("This benchmark could not be done on ROCM!")
else:
    print("Swin Transformer")
    results += bench_sddmm(swin_t_config)
    print("ViT")
    results += bench_sddmm(vit_config)
    print("Basic cases")
    results += bench_sddmm(basic_config)
