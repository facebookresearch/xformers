# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import torch
from torch.utils import benchmark

from xformers.components.attention.core import (
    SparseCS,
    _create_random_sparsity,
    _matmul_with_mask,
    _softmax,
    bmm,
)

MIN_RUN_TIME = 1
SHAPES = [[8, 8], [256, 1024], [128, 256]]
SPARSITIES = [0.5, 0.8, 0.9, 0.95, 0.99]


def bench_sddmm():
    min_run_time = MIN_RUN_TIME
    SPARSITIES = [0.95, 0.98, 0.99, 0.995, 0.999]

    device = torch.device("cuda")
    results = []

    for B, M, K in zip(*SHAPES):
        a = torch.rand(B, M, K, device=device)
        b = torch.rand(B, M, K, device=device)

        for backend, prob in itertools.product(
            ["coo_pytorch", "csr_sputnik", "csr_ge"], SPARSITIES
        ):
            mask = _create_random_sparsity(torch.ones(B, M, M, dtype=torch.bool), prob)
            aa = a
            bb = b
            if "csr" in backend:
                mask = SparseCS(mask, device)
                aa = a
                bb = b
                row_indices = mask.row_indices
                row_offsets = mask.row_offsets
                column_indices = mask.column_indices
                if "_ge" in backend:
                    fn = torch.ops.xformers.csr_sddmm
                else:
                    fn = torch.ops.xformers.sddmm_sputnik
                fn_str = "fn(a, b, row_indices, row_offsets, column_indices)"
            else:
                mask = mask.to_sparse().to(device)
                _, row_offsets, column_indices = mask.indices().int().unbind()
                row_offsets = row_offsets.contiguous()
                column_indices = column_indices.contiguous()
                row_indices = row_offsets

                bb = b.transpose(-2, -1)
                fn = _matmul_with_mask
                fn_str = "fn(a, b, mask)"

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
                    sub_label=f"sparsity {backend}: {prob:0.4f}",
                    description=f"B={B}, M={M}, K={K}",
                ).blocked_autorange(min_run_time=min_run_time)
            )

    compare = benchmark.Compare(results)
    compare.print()


def bench_matmul_with_mask():
    min_run_time = MIN_RUN_TIME
    prob = 0.9
    device = torch.device("cuda")
    results = []

    for B, M, K in zip(*SHAPES):
        a = torch.rand(B, M, K, device=device)
        b = torch.rand(B, K, M, device=device)
        mask = torch.rand(B, M, M, device=device) > prob

        results.extend(
            [
                benchmark.Timer(
                    stmt="_matmul_with_mask(a, b, mask)",
                    globals={
                        "a": a,
                        "b": b,
                        "mask": None,
                        "_matmul_with_mask": _matmul_with_mask,
                    },
                    label="matmul_with_mask",
                    sub_label="dense",
                    description=f"B={B}, M={M}, K={K}",
                ).blocked_autorange(min_run_time=min_run_time),
                benchmark.Timer(
                    stmt="_matmul_with_mask(a, b, mask)",
                    globals={
                        "a": a,
                        "b": b,
                        "mask": mask,
                        "_matmul_with_mask": _matmul_with_mask,
                    },
                    label="matmul_with_mask",
                    sub_label="dense with masking",
                    description=f"B={B}, M={M}, K={K}",
                ).blocked_autorange(min_run_time=min_run_time),
            ]
        )
        for sputnik, prob in itertools.product([False, True], SPARSITIES):
            mask = _create_random_sparsity(
                torch.ones(B, M, M, dtype=torch.bool, device=device), prob
            )
            aa = a
            bb = b
            if sputnik:
                mask = SparseCS(mask, device)
                aa = a
                bb = b.transpose(-2, -1).contiguous().transpose(-2, -1)
            else:
                mask = mask.to_sparse()
            results.append(
                benchmark.Timer(
                    stmt="_matmul_with_mask(a, b, mask)",
                    globals={
                        "a": aa,
                        "b": bb,
                        "mask": mask,
                        "_matmul_with_mask": _matmul_with_mask,
                    },
                    label="matmul_with_mask",
                    sub_label=f"sparsity {'sputnik' if sputnik else 'pytorch'}: {prob:0.2f}",
                    description=f"B={B}, M={M}, K={K}",
                ).blocked_autorange(min_run_time=min_run_time)
            )

    compare = benchmark.Compare(results)
    compare.print()


def bench_softmax():
    min_run_time = MIN_RUN_TIME
    prob = 0.9
    device = torch.device("cuda")
    results = []

    for B, M, K in zip(*SHAPES):
        a = torch.rand(B, M, M, device=device)
        a[a < prob] = 0

        results.extend(
            [
                benchmark.Timer(
                    stmt="_softmax(a)",
                    globals={
                        "a": a,
                        "_softmax": _softmax,
                    },
                    label="softmax",
                    sub_label="dense",
                    description=f"B={B}, M={M}, K={K}",
                ).blocked_autorange(min_run_time=min_run_time),
            ]
        )
        for sputnik, prob in itertools.product([False, True], SPARSITIES):
            a = _create_random_sparsity(torch.rand(B, M, M, device=device), prob)
            if sputnik:
                a = SparseCS(a, device)
            else:
                a = a.to_sparse()
            results.append(
                benchmark.Timer(
                    stmt="_softmax(a)",
                    globals={
                        "a": a,
                        "_softmax": _softmax,
                    },
                    label="softmax",
                    sub_label=f"sparsity {'sputnik' if sputnik else 'pytorch'}: {prob:0.2f}",
                    description=f"B={B}, M={M}, K={K}",
                ).blocked_autorange(min_run_time=min_run_time)
            )

    compare = benchmark.Compare(results)
    compare.print()


def bench_bmm():
    min_run_time = MIN_RUN_TIME
    prob = 0.9
    device = torch.device("cuda")
    results = []

    for B, M, K in zip(*SHAPES):
        a = torch.rand(B, M, M, device=device)
        a[a < prob] = 0
        b = torch.rand(B, M, K, device=device)

        results.extend(
            [
                benchmark.Timer(
                    stmt="bmm(a, b)",
                    globals={
                        "a": a,
                        "b": b,
                        "bmm": bmm,
                    },
                    label="bmm",
                    sub_label="dense",
                    description=f"B={B}, M={M}, K={K}",
                ).blocked_autorange(min_run_time=min_run_time),
            ]
        )
        for sputnik, prob in itertools.product([False, True], SPARSITIES):
            a = _create_random_sparsity(torch.rand(B, M, M, device=device), prob)
            bb = b
            if sputnik:
                a = SparseCS(a, device)
                bb = b
            else:
                a = a.to_sparse()
            results.append(
                benchmark.Timer(
                    stmt="bmm(a, b)",
                    globals={
                        "a": a,
                        "b": bb,
                        "bmm": bmm,
                    },
                    label="bmm",
                    sub_label=f"sparsity {'sputnik' if sputnik else 'pytorch'}: {prob:0.2f}",
                    description=f"B={B}, M={M}, K={K}",
                ).blocked_autorange(min_run_time=min_run_time)
            )

    compare = benchmark.Compare(results)
    compare.print()


if torch.version.hip:
    print("This benchmark could not be done on ROCM!")
else:
    bench_sddmm()
    bench_matmul_with_mask()
    bench_softmax()
    bench_bmm()
