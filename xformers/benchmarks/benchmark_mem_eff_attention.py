# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import itertools
import math
from functools import partial
from typing import cast

import torch
from torch.utils import benchmark
from utils import benchmark_main_helper

import xformers.ops

CHECK_CORRECTNESS = True
torch.backends.cuda.matmul.allow_tf32 = False


def create_attn_bias(
    bias_type, batch_size: int, q_len: int, kv_len: int, device, dtype
):
    NoneType = type(None)
    if bias_type is NoneType:
        return None
    if bias_type is torch.Tensor:
        attn_bias = torch.randn((batch_size, 1, kv_len), device=device, dtype=dtype) * 3
        return attn_bias.expand(batch_size, q_len, kv_len)
    if bias_type is xformers.ops.LowerTriangularMask:
        return bias_type([batch_size, q_len, kv_len], dtype=dtype, device=device)
    assert False, f"Unsupported bias type: {bias_type}"


def ref_attention(q, k, v, attn_bias=None, p=0.0):
    if isinstance(attn_bias, xformers.ops.AttentionMask):
        attn_bias = attn_bias.to_tensor().to(q.dtype)
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


min_run_time = 0.5
device = torch.device("cuda")

NUM_THREADS = [1] if device.type == "cuda" else [1, 40]
SHAPES = list(itertools.product([256, 1024], [128, 512, 1024], [16, 32, 64, 128]))
SHAPES += [
    # ViT
    (384, 197, 88),
    (384, 197, 80),
    (384, 197, 64),
    (1024, 197, 88),
    (1024, 197, 80),
    (1024, 197, 64),
    # GPT-Z
    (1 * 160, 4096, 128),
    (2 * 160, 4096, 128),
    (1 * 160, 8192, 128),
    (2 * 160, 8192, 128),
    # FB models
    (1024 * 8, 82, 64),
    (150 * 16, 256, 64),
    (64 * 12, 256, 64),
    # Stable diffusion (https://github.com/huggingface/diffusers/pull/532)
    (16, 4096, 40),  # 512x512
    (16, 16384, 40),  # 1024x1024
    # ParlAI model
    (256 * 16, 4096, 64),
]

SHAPES = list(set(SHAPES))
SHAPES.sort()


p = 0.0
FORCE_OP = None
# FORCE_OP = xformers.ops.MemoryEfficientAttentionOp
# FORCE_OP = xformers.ops.MemoryEfficientAttentionCutlassOp
# FORCE_OP = xformers.ops.MemoryEfficientAttentionFlashAttentionOp


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


CASES = list(
    product_dict(
        shape=SHAPES,
        num_threads=NUM_THREADS,
        attn_bias_type=[type(None), torch.Tensor, xformers.ops.LowerTriangularMask],
        dtype=[torch.half, torch.bfloat16, torch.float],
    )
)


def benchmark_forward(shape, num_threads: int, attn_bias_type, dtype):
    B, M, K = shape
    q = torch.rand(shape, device=device, dtype=dtype)

    dispatch = xformers.ops.AttentionOpDispatch(
        dtype=dtype,
        device=device,
        k=K,
        attn_bias_type=attn_bias_type,
        has_dropout=False,
        kv_len=M,
        q_len=M,
    )
    try:
        op = dispatch.op if FORCE_OP is None else FORCE_OP
    except NotImplementedError:
        return
    if not op.supports(dispatch):
        return

    attn_bias = create_attn_bias(
        attn_bias_type,
        batch_size=B,
        q_len=M,
        kv_len=M,
        device=device,
        dtype=dtype,
    )

    dtype_str = {
        torch.bfloat16: "b16",
        torch.half: "f16",
        torch.float: "f32",
    }[dtype]
    sub_label = f"{dtype_str} B={B}, M={M}, K={K}"

    try:
        r = xformers.ops.memory_efficient_attention(q, q, q, attn_bias, op=op).float()
        rr = ref_attention(
            q.float(),
            q.float(),
            q.float(),
            attn_bias,
        )
        assert not CHECK_CORRECTNESS or (r - rr).abs().max() < 4e-3, (
            (r - rr).abs().max()
        )
        del r, rr
    except RuntimeError:  # OOM
        pass

    yield benchmark.Timer(
        stmt="fn(q, k, v, attn_bias, p)",
        globals={
            "q": q,
            "k": q.clone(),
            "v": q.clone(),
            "attn_bias": attn_bias,
            "p": p,
            "fn": partial(xformers.ops.memory_efficient_attention, op=op),
        },
        label=f"attention (attn_bias={attn_bias_type})",
        description=op.NAME,
        sub_label=sub_label,
        num_threads=num_threads,
    )
    yield benchmark.Timer(
        stmt="fn(q, k, v, attn_bias, p)",
        globals={
            "q": q,
            "k": q.clone(),
            "v": q.clone(),
            "attn_bias": attn_bias,
            "p": p,
            "fn": ref_attention,
        },
        label=f"attention (attn_bias={attn_bias_type})",
        description="vanilla",
        sub_label=sub_label,
        num_threads=num_threads,
    )


def benchmark_backward(shape, num_threads: int, attn_bias_type, dtype):
    B, M, K = shape
    q = torch.rand(shape, device=device, dtype=dtype, requires_grad=True)

    dispatch = xformers.ops.AttentionOpDispatch(
        dtype=dtype,
        device=device,
        k=K,
        attn_bias_type=attn_bias_type,
        has_dropout=False,
        kv_len=M,
        q_len=M,
    )
    try:
        op = dispatch.op if FORCE_OP is None else FORCE_OP
    except NotImplementedError:
        return
    if not op.supports(dispatch):
        return

    attn_bias = create_attn_bias(
        attn_bias_type,
        batch_size=B,
        q_len=M,
        kv_len=M,
        device=device,
        dtype=dtype,
    )

    dtype_str = {
        torch.bfloat16: "b16",
        torch.half: "f16",
        torch.float: "f32",
    }[dtype]
    sub_label = f"{dtype_str} B={B}, M={M}, K={K}"

    out = xformers.ops.memory_efficient_attention(
        q, q.clone(), q.clone(), attn_bias, p, op=op
    )
    grad_benchmark = torch.ones_like(q)

    yield benchmark.Timer(
        stmt="out.backward(grad, retain_graph=True)",
        globals={
            "out": out,
            "grad": grad_benchmark,
        },
        label=f"attention backward (attn_bias={attn_bias_type})",
        description=op.NAME,
        sub_label=sub_label,
        num_threads=num_threads,
    )
    del out

    try:
        q.grad = None
        r = xformers.ops.memory_efficient_attention(q, q, q, attn_bias, op=op)
        r.backward(torch.ones_like(q))

        grad = cast(torch.Tensor, q.grad)
        q.grad = None

        rr = ref_attention(q, q, q, attn_bias)
        rr.backward(torch.ones_like(q))
        atol = 2e-4 + 2e-6 * K * M * math.sqrt(B) * math.sqrt(M)
        # type: ignore
        assert (
            not CHECK_CORRECTNESS or (grad - q.grad).abs().max() < atol
        ), f"{(grad - q.grad).abs().max()}"
        q.grad = None
        del r, grad

        yield benchmark.Timer(
            stmt="out.backward(grad, retain_graph=True)",
            globals={
                "out": ref_attention(q, q.clone(), q.clone(), attn_bias),
                "grad": grad_benchmark,
            },
            label=f"attention backward (attn_bias={attn_bias_type})",
            description="vanilla",
            sub_label=sub_label,
            num_threads=num_threads,
        )
    except RuntimeError:  # OOM
        pass


benchmark_main_helper(benchmark_forward, CASES, min_run_time=min_run_time)
benchmark_main_helper(benchmark_backward, CASES, min_run_time=min_run_time)
