# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

import torch
from torch.utils import benchmark

from xformers.components.attention.utils import iterative_pinv

MIN_RUN_TIME = 1
SHAPES = [[8, 8], [256, 1024], [128, 256]]
SPARSITIES = [0.5, 0.8, 0.9, 0.95, 0.99]


def bench_inverse(inverse_fn: Callable[[torch.Tensor], torch.Tensor]):
    min_run_time = MIN_RUN_TIME
    prob = 0.9
    device = torch.device("cuda")
    results = []

    for B, M, K in zip(*SHAPES):
        a = torch.rand(B, M, M, device=device)
        a[a < prob] = 0
        a = torch.softmax(a, dim=-1)

        results.extend(
            [
                benchmark.Timer(
                    stmt=f"{inverse_fn.__name__}(a)",
                    globals={
                        "a": a,
                        f"{inverse_fn.__name__}": inverse_fn,
                    },
                    label=f"{inverse_fn.__name__}",
                    sub_label="dense",
                    description=f"B={B}, M={M}, K={K}",
                ).blocked_autorange(min_run_time=min_run_time),
            ]
        )
        for prob in SPARSITIES:
            a = torch.rand(B, M, M, device=device)
            a[a < prob] = 0
            a = a.to_sparse()
            results.append(
                benchmark.Timer(
                    stmt=f"{inverse_fn.__name__}(a)",
                    globals={
                        "a": a,
                        f"{inverse_fn.__name__}": inverse_fn,
                    },
                    label=f"{inverse_fn.__name__}",
                    sub_label=f"sparsity: {prob:0.2f}",
                    description=f"B={B}, M={M}, K={K}",
                ).blocked_autorange(min_run_time=min_run_time)
            )

    compare = benchmark.Compare(results)
    compare.print()


def iterative_pinv_analysis(
    identity_tolerance: float = 1e-1,
    pinv_tolerance: float = 5e-1,
    max_iters: int = 30,
    plot: bool = True,
):

    for i in range(1, 10):
        B, M = 1, 2**i
        a = torch.rand(B, M, M)
        a = torch.softmax(a, dim=-1)

        for n_iter in range(1, max_iters + 1):
            result = iterative_pinv(a, n_iter=n_iter)
            expected = torch.linalg.pinv(a)

            result_identity = torch.matmul(a, result)
            identity = torch.eye(M)

            # Default is frobenius norm.
            identity_error = torch.linalg.norm(identity - result_identity, dim=(-2, -1))
            inverse_error = torch.linalg.norm(expected - result, dim=(-2, -1))

            if (identity_error < identity_tolerance).all() or n_iter == max_iters:
                print(
                    f"Size {M}, n_iters {n_iter}: \n\t \
                    Final Error from Identity: {identity_error.item()} \n\t \
                    Final Error from linalg.pinv {inverse_error.item()}"
                )
                break


if torch.version.hip:
    print("This benchmark could not be done on ROCM!")
else:
    iterative_pinv_analysis()
    bench_inverse(iterative_pinv)
    bench_inverse(torch.linalg.pinv)
