# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import torch
from torch.utils import benchmark

from xformers.ops import masked_matmul, softmax
from xformers.sparse import SparseCOOTensor, SparseCSRTensor
from xformers.testing import _create_tensor

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
            tensor_type = SparseCSRTensor if "csr" in backend else SparseCOOTensor
            mask = _create_tensor(tensor_type, device, torch.bool, (B, M, M), prob)
            aa = a
            bb = b
            if "csr" in backend:
                aa = a
                bb = b
                row_indices = mask._csr_row_indices
                row_offsets = mask._csr_row_offsets
                column_indices = mask._csr_column_indices
                if "_ge" in backend:
                    fn = torch.ops.xformers.csr_sddmm
                else:
                    fn = torch.ops.xformers.sddmm_sputnik
                fn_str = "fn(a, b, row_indices, row_offsets, column_indices)"
            else:
                _, row_offsets, column_indices = mask.indices().int().unbind()
                row_offsets = row_offsets.contiguous()
                column_indices = column_indices.contiguous()
                row_indices = row_offsets

                bb = b.transpose(-2, -1)
                fn = masked_matmul
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
                    stmt="masked_matmul(a, b, mask)",
                    globals={
                        "a": a,
                        "b": b,
                        "mask": None,
                        "masked_matmul": masked_matmul,
                    },
                    label="masked_matmul",
                    sub_label="dense",
                    description=f"B={B}, M={M}, K={K}",
                ).blocked_autorange(min_run_time=min_run_time),
                benchmark.Timer(
                    stmt="masked_matmul(a, b, mask)",
                    globals={
                        "a": a,
                        "b": b,
                        "mask": mask,
                        "masked_matmul": masked_matmul,
                    },
                    label="masked_matmul",
                    sub_label="dense with masking",
                    description=f"B={B}, M={M}, K={K}",
                ).blocked_autorange(min_run_time=min_run_time),
            ]
        )
        for sputnik, prob in itertools.product([False, True], SPARSITIES):
            tensor_type = SparseCSRTensor if sputnik else SparseCOOTensor
            mask = _create_tensor(tensor_type, device, torch.bool, (B, M, M), prob)
            aa = a
            bb = b
            if sputnik:
                aa = a
                bb = b.transpose(-2, -1).contiguous().transpose(-2, -1)
            results.append(
                benchmark.Timer(
                    stmt="masked_matmul(a, b, mask)",
                    globals={
                        "a": aa,
                        "b": bb,
                        "mask": mask,
                        "masked_matmul": masked_matmul,
                    },
                    label="masked_matmul",
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
                    stmt="softmax(a)",
                    globals={
                        "a": a,
                        "softmax": softmax,
                    },
                    label="softmax",
                    sub_label="dense",
                    description=f"B={B}, M={M}, K={K}",
                ).blocked_autorange(min_run_time=min_run_time),
            ]
        )
        for sputnik, prob in itertools.product([False, True], SPARSITIES):
            tensor_type = SparseCSRTensor if sputnik else SparseCOOTensor
            a = _create_tensor(tensor_type, device, torch.float32, (B, M, M), prob)
            results.append(
                benchmark.Timer(
                    stmt="softmax(a)",
                    globals={
                        "a": a,
                        "softmax": softmax,
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
                    stmt="fn(a, b)",
                    globals={
                        "a": a,
                        "b": b,
                        "fn": torch.matmul,
                    },
                    label="bmm",
                    sub_label="dense",
                    description=f"B={B}, M={M}, K={K}",
                ).blocked_autorange(min_run_time=min_run_time),
            ]
        )
        for sputnik, prob in itertools.product([False, True], SPARSITIES):
            tensor_type = SparseCSRTensor if sputnik else SparseCOOTensor
            a = _create_tensor(tensor_type, device, torch.float32, (B, M, M), prob)
            results.append(
                benchmark.Timer(
                    stmt="fn(a, b)",
                    globals={
                        "a": a,
                        "b": b,
                        "fn": torch.matmul,
                    },
                    label="bmm",
                    sub_label=f"sparsity {'sputnik' if sputnik else 'pytorch'}: {prob:0.2f}",
                    description=f"B={B}, M={M}, K={K}",
                ).blocked_autorange(min_run_time=min_run_time)
            )

    compare = benchmark.Compare(results)
    compare.print()


bench_sddmm()
bench_matmul_with_mask()
bench_softmax()
bench_bmm()
