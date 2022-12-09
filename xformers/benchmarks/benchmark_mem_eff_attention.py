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
import xformers.ops.fmha as fmha

CHECK_CORRECTNESS = True
torch.backends.cuda.matmul.allow_tf32 = False


def create_attn_bias(
    bias_type, batch_size: int, num_heads: int, q_len: int, kv_len: int, device, dtype
):
    NoneType = type(None)
    if bias_type is NoneType:
        return None
    if bias_type is torch.Tensor:
        attn_bias = (
            torch.randn((batch_size * num_heads, 1, kv_len), device=device, dtype=dtype)
            * 3
        )
        return attn_bias.expand(batch_size * num_heads, q_len, kv_len)
    if bias_type is xformers.ops.LowerTriangularMask:
        return bias_type([1, q_len, kv_len], dtype=dtype, device=device)
    assert False, f"Unsupported bias type: {bias_type}"


def ref_attention_bmk(q, k, v, attn_bias=None, p=0.0):
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


def ref_attention(q, k, v, attn_bias, p=0.0):
    assert q.ndim == 4

    def T(t):
        return t.permute((0, 2, 1, 3)).reshape(
            [t.shape[0] * t.shape[2], t.shape[1], t.shape[3]]
        )

    out = ref_attention_bmk(T(q), T(k), T(v), attn_bias, p)
    out = out.reshape([q.shape[0], q.shape[2], q.shape[1], v.shape[3]])
    return out.permute((0, 2, 1, 3))


min_run_time = 0.5
device = torch.device("cuda")

NUM_THREADS = [1] if device.type == "cuda" else [1, 40]
SHAPES = [
    # ViT
    (384, 197, 1, 88),
    (384, 197, 1, 80),
    (384, 197, 1, 64),
    (1024, 197, 1, 88),
    (1024, 197, 1, 80),
    (1024, 197, 1, 64),
    # ViT-Huge
    (32 * 16, 197, 1, 80),
    (32, 197, 16, 80),
    (32, 197, 16, 64),
    (32, 197, 16, 128),
    # ViT-Giant
    (16 * 16, 197, 1, 88),
    (16, 197, 16, 88),
    (16, 197, 16, 64),
    (16, 197, 16, 128),
    # GPT-Z
    (1, 4096, 160, 128),
    (2, 4096, 160, 128),
    (1, 8192, 160, 128),
    (2, 8192, 160, 128),
    # FB models
    (1024, 82, 8, 64),
    (150, 256, 16, 64),
    (64, 256, 12, 64),
    # Stable diffusion (https://github.com/huggingface/diffusers/pull/532)
    (1, 4096, 16, 40),  # 512x512
    (1, 16384, 16, 40),  # 1024x1024
    # ParlAI model
    (256, 4096, 16, 64),
    # Zetta B M H K
    (8, 2048, 20, 128),
    *sorted(
        list(itertools.product([16, 64], [128, 512, 1024], [16], [16, 32, 64, 128]))
    ),
]


p = 0.0
FORCE_OP = None
# FORCE_OP = xformers.ops.MemoryEfficientAttentionOp
# FORCE_OP = xformers.ops.MemoryEfficientAttentionCutlassOp
# FORCE_OP = xformers.ops.MemoryEfficientAttentionFlashAttentionOp
# FORCE_OP = xformers.ops.MemoryEfficientAttentionCutlassFwdFlashBwOp
# FORCE_OP = xformers.ops.TritonFlashAttentionOp
# FORCE_OP = xformers.ops.MemoryEfficientAttentionTritonFwdFlashBwOp


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


def create_tensors(shape, dtype, requires_grad=False):
    B, M, H, K = shape
    qkv = torch.rand(
        [B, M, 3, H, K], device=device, dtype=dtype, requires_grad=requires_grad
    )
    q, k, v = xformers.ops.unbind(qkv, 2)
    return qkv, q, k, v


def benchmark_forward(shape, num_threads: int, attn_bias_type, dtype):
    B, M, H, K = shape
    _, q, k, v = create_tensors(shape, dtype)

    inp = fmha.Inputs(query=q, key=k, value=v)
    try:
        op = (fmha._dispatch_fw(inp), None) if FORCE_OP is None else FORCE_OP
    except NotImplementedError:
        return
    if not op[0].supports(inp):
        return

    inp.attn_bias = create_attn_bias(
        attn_bias_type,
        batch_size=B,
        num_heads=H,
        q_len=M,
        kv_len=M,
        device=device,
        dtype=dtype,
    )
    if not op[0].supports(inp):
        return

    dtype_str = {
        torch.bfloat16: "b16",
        torch.half: "f16",
        torch.float: "f32",
    }[dtype]
    sub_label = f"{dtype_str} B={B}, M={M}, H={H}, K={K}"

    try:
        r = xformers.ops.memory_efficient_attention(
            q, k, v, inp.attn_bias, op=op
        ).float()
        rr = ref_attention(
            q.float(),
            k.float(),
            v.float(),
            inp.attn_bias,
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
            "k": k,
            "v": v,
            "attn_bias": inp.attn_bias,
            "p": p,
            "fn": partial(xformers.ops.memory_efficient_attention, op=op),
        },
        label=f"attention (attn_bias={attn_bias_type})",
        description=op[0].NAME,
        sub_label=sub_label,
        num_threads=num_threads,
    )
    yield benchmark.Timer(
        stmt="fn(q, k, v, attn_bias, p)",
        globals={
            "q": q,
            "k": k,
            "v": v,
            "attn_bias": inp.attn_bias,
            "p": p,
            "fn": ref_attention,
        },
        label=f"attention (attn_bias={attn_bias_type})",
        description="eager",
        sub_label=sub_label,
        num_threads=num_threads,
    )


def benchmark_backward(shape, num_threads: int, attn_bias_type, dtype):
    B, M, H, K = shape
    qkv, q, k, v = create_tensors(shape, dtype, requires_grad=True)

    inp = fmha.Inputs(query=q, key=k, value=v)
    try:
        op = (
            (fmha._dispatch_fw(inp), fmha._dispatch_bw(inp))
            if FORCE_OP is None
            else FORCE_OP
        )
    except NotImplementedError:
        return
    if not op[0].supports(inp) or not op[1].supports(inp):
        return

    inp.attn_bias = create_attn_bias(
        attn_bias_type,
        batch_size=B,
        num_heads=H,
        q_len=M,
        kv_len=M,
        device=device,
        dtype=dtype,
    )
    if not op[0].supports(inp) or not op[1].supports(inp):
        return

    dtype_str = {
        torch.bfloat16: "b16",
        torch.half: "f16",
        torch.float: "f32",
    }[dtype]
    sub_label = f"{dtype_str} B={B}, M={M}, H={H}, K={K}"

    out = xformers.ops.memory_efficient_attention(q, k, v, inp.attn_bias, p, op=op)
    grad_benchmark = torch.ones_like(q)

    yield benchmark.Timer(
        stmt="out.backward(grad, retain_graph=True)",
        globals={
            "out": out,
            "grad": grad_benchmark,
        },
        label=f"attention backward (attn_bias={attn_bias_type})",
        description=op[1].NAME,
        sub_label=sub_label,
        num_threads=num_threads,
    )
    del out

    try:
        qkv.grad = None
        r = xformers.ops.memory_efficient_attention(q, k, v, inp.attn_bias, op=op)
        r.backward(torch.ones_like(q))

        grad = cast(torch.Tensor, qkv.grad)
        qkv.grad = None

        rr = ref_attention(q, k, v, inp.attn_bias)
        rr.backward(torch.ones_like(q))
        atol = 2e-4 + 2e-6 * K * M * math.sqrt(B) * math.sqrt(M)
        # type: ignore
        assert (
            not CHECK_CORRECTNESS or (grad - qkv.grad).abs().max() < atol
        ), f"{(grad - qkv.grad).abs().max()}"
        qkv.grad = None
        del r, grad

        yield benchmark.Timer(
            stmt="out.backward(grad, retain_graph=True)",
            globals={
                "out": ref_attention(q, k, v, inp.attn_bias),
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
