import torch
from torch.utils import benchmark

from xformers.components.attention.core import _matmul_with_mask, _softmax, bmm

MIN_RUN_TIME = 1
SHAPES = [[8, 8], [256, 1024], [128, 256]]
SPARSITIES = [0.5, 0.8, 0.9, 0.95, 0.99]


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
        for prob in SPARSITIES:
            mask = (torch.rand(B, M, M, device=device) > prob).to_sparse()
            results.append(
                benchmark.Timer(
                    stmt="_matmul_with_mask(a, b, mask)",
                    globals={
                        "a": a,
                        "b": b,
                        "mask": mask,
                        "_matmul_with_mask": _matmul_with_mask,
                    },
                    label="matmul_with_mask",
                    sub_label=f"sparsity: {prob:0.2f}",
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
        for prob in SPARSITIES:
            a = torch.rand(B, M, M, device=device)
            a[a < prob] = 0
            a = a.to_sparse()
            results.append(
                benchmark.Timer(
                    stmt="_softmax(a)",
                    globals={
                        "a": a,
                        "_softmax": _softmax,
                    },
                    label="softmax",
                    sub_label=f"sparsity: {prob:0.2f}",
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
        for prob in SPARSITIES:
            a = torch.rand(B, M, M, device=device)
            a[a < prob] = 0
            a = a.to_sparse()
            results.append(
                benchmark.Timer(
                    stmt="bmm(a, b)",
                    globals={
                        "a": a,
                        "b": b,
                        "bmm": bmm,
                    },
                    label="bmm",
                    sub_label=f"sparsity: {prob:0.2f}",
                    description=f"B={B}, M={M}, K={K}",
                ).blocked_autorange(min_run_time=min_run_time)
            )

    compare = benchmark.Compare(results)
    compare.print()


bench_matmul_with_mask()
bench_softmax()
bench_bmm()
