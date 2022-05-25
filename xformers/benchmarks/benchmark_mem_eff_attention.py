# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import itertools
from functools import partial

import torch
from torch.utils import benchmark
from utils import benchmark_main_helper

import xformers.ops


def ref_attention(q, k, v, attn_bias=None, p=0.0):
    q = q * (1.0 / q.shape[-1] ** 0.5)
    if attn_bias is None:
        attn = q @ k.transpose(-2, -1)
    else:
        # equivalent to (q @ k.transpose(-2, -1) + m).softmax(-1) @ v
        # but faster, and is what is used in PyTorch now
        attn = torch.baddbmm(attn_bias, q, k.transpose(-2, -1))
    attn = attn.softmax(-1)
    if p > 0:
        attn = torch.nn.functional.dropout(attn, p=p)
    return attn @ v


min_run_time = 2
device = torch.device("cuda")

NUM_THREADS = [1] if device.type == "cuda" else [1, 40]
SHAPES = list(
    itertools.product([1, 8, 32, 256], [127, 128, 512, 513, 1023, 1024], [16, 32])
) + list(itertools.product([32, 256], [128, 512, 1024, 2048], [16, 32, 64, 128, 256]))
SHAPES = list(set(SHAPES))
SHAPES.sort()


p = 0.0
op = xformers.ops.MemoryEfficientAttentionOp
# op = xformers.ops.MemoryEfficientAttentionGenericForwardOp


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


CASES = list(
    product_dict(
        shape=SHAPES,
        num_threads=NUM_THREADS,
        use_attn_bias=[False, True],
    )
)


def benchmark_forward(shape, num_threads: int, use_attn_bias: bool):
    B, M, K = shape
    q = torch.rand(shape, device=device)
    attn_bias = None
    if use_attn_bias:
        attn_bias = torch.rand(shape[0], 1, shape[1], device=device).expand(
            shape[0], shape[1], shape[1]
        )
    sub_label = f"B={B}, M={M}, K={K}"

    if K > op.SUPPORTED_MAX_K or (use_attn_bias and not op.SUPPORTS_ATTN_BIAS):
        raise NotImplementedError()
    if True:
        r = xformers.ops.memory_efficient_attention(q, q, q, attn_bias, op=op)
        rr = ref_attention(q, q, q, attn_bias)
        assert (r - rr).abs().max() < 1e-5
        del r, rr

    yield benchmark.Timer(
        stmt="fn(q, q, q, attn_bias, p)",
        globals={
            "q": q,
            "attn_bias": attn_bias,
            "p": p,
            "fn": partial(xformers.ops.memory_efficient_attention, op=op),
        },
        label=f"attention (use_attn_bias={use_attn_bias})",
        description="optimized",
        sub_label=sub_label,
        num_threads=num_threads,
    )
    yield benchmark.Timer(
        stmt="fn(q, q, q, attn_bias, p)",
        globals={
            "q": q,
            "attn_bias": attn_bias,
            "p": p,
            "fn": ref_attention,
        },
        label=f"attention (use_attn_bias={use_attn_bias})",
        description="vanilla",
        sub_label=sub_label,
        num_threads=num_threads,
    )


def benchmark_backward(shape, num_threads: int, use_attn_bias: bool):
    B, M, K = shape
    q = torch.rand(shape, device=device, requires_grad=True)
    attn_bias = None
    if use_attn_bias:
        attn_bias = torch.rand(shape[0], 1, shape[1], device=device).expand(
            shape[0], shape[1], shape[1]
        )
    sub_label = f"B={B}, M={M}, K={K}"

    if K > op.SUPPORTED_MAX_K or (use_attn_bias and not op.SUPPORTS_ATTN_BIAS):
        raise NotImplementedError()
    if True:
        r = xformers.ops.memory_efficient_attention(q, q, q, attn_bias, op=op)
        r.backward(torch.ones_like(q))

        grad = q.grad
        q.grad = None

        rr = ref_attention(q, q, q, attn_bias)
        rr.backward(torch.ones_like(q))
        assert (grad - q.grad).abs().max() < 1e-5, f"{(grad - q.grad).abs().max()}"
        q.grad = None
        del r, rr, grad

    out = xformers.ops.memory_efficient_attention(q, q, q, attn_bias, p, op=op)
    grad = torch.ones_like(q)

    yield benchmark.Timer(
        stmt="out.backward(grad, retain_graph=True)",
        globals={
            "out": out,
            "grad": grad,
        },
        label=f"attention backward (use_attn_bias={use_attn_bias})",
        description="optimized",
        sub_label=sub_label,
        num_threads=num_threads,
    )
    del out

    yield benchmark.Timer(
        stmt="out.backward(grad, retain_graph=True)",
        globals={
            "out": ref_attention(q, q, q, attn_bias, p),
            "grad": grad,
        },
        label=f"attention backward (use_attn_bias={use_attn_bias})",
        description="vanilla",
        sub_label=sub_label,
        num_threads=num_threads,
    )


benchmark_main_helper(benchmark_forward, CASES, min_run_time=min_run_time)
benchmark_main_helper(benchmark_backward, CASES, min_run_time=min_run_time)
