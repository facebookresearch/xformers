# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import itertools
import random
from functools import partial

import torch

import xformers.ops
import xformers.ops.fmha as fmha
from torch.utils import benchmark
from xformers.attn_bias_utils import create_attn_bias, ref_attention
from xformers.benchmarks.utils import benchmark_main_helper, create_argparser

torch.backends.cuda.matmul.allow_tf32 = False

min_run_time = 0.5
device = torch.device("cuda")

NUM_THREADS = [1] if device.type == "cuda" else [1, 40]
VISION_SHAPES = [
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
    # FB models
    (1024, 82, 8, 64),
    (150, 256, 16, 64),
    (64, 256, 12, 64),
    # Stable diffusion (https://github.com/huggingface/diffusers/pull/532)
    (1, 4096, 16, 40),  # 512x512
    (1, 16384, 16, 40),  # 1024x1024
    (1, 4096, 16, 80),
    (1, 16384, 16, 80),
    # + bs4
    (4, 4096, 16, 40),
    (4, 16384, 16, 40),
    (4, 4096, 16, 80),
    (4, 16384, 16, 80),
    # ParlAI model
    (256, 4096, 16, 64),
    # Zetta B M H K
    (8, 2048, 20, 128),
]

LLM_SHAPES = [
    # LLaMa 70b - mp=8/16
    *sorted(itertools.product([1, 2], [2048, 4096, 8192], [4, 8], [128])),
    *sorted(
        itertools.product([16], [128, 512, 1024], [16], [16, 32, 64, 128, 160, 256])
    ),
]


OPS = [
    (xformers.ops.fmha.cutlass.FwOp, xformers.ops.fmha.cutlass.BwOp),
    (xformers.ops.fmha.flash.FwOp, xformers.ops.fmha.flash.BwOp),
    (xformers.ops.fmha.flash3.FwOp, xformers.ops.fmha.flash3.BwOp),
    (xformers.ops.fmha.ck.FwOp, xformers.ops.fmha.ck.BwOp),
]


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


VISION_CASES, LLM_CASES = [
    list(
        product_dict(
            shape_q=SHAPES,
            num_threads=NUM_THREADS,
            dropout_p=[0.0],
            attn_bias_cfg=[(type(None), False)],
            dtype=[torch.half],
        )
    )
    for SHAPES in (VISION_SHAPES, LLM_SHAPES)
]

# Add more cases with some variations
for c in VISION_CASES.copy():
    c = c.copy()
    c.update(
        random.Random(str(c["shape_q"])).choice(
            [
                {"dropout_p": 0.3},
                {"attn_bias_cfg": (torch.Tensor, False)},
                {"attn_bias_cfg": (torch.Tensor, True)},
                {"dtype": torch.bfloat16},
                {"dtype": torch.float},
            ]
        )
    )
    VISION_CASES.append(c)


LLM_CASE_UPDATES = [
    {"attn_bias_cfg": (torch.Tensor, True)},
    {"attn_bias_cfg": (xformers.ops.LowerTriangularMask, False)},
    *[
        {
            "attn_bias_cfg": (
                xformers.ops.fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask,
                False,
            ),
            "Hkv": Hkv,
            "dtype": torch.bfloat16,
        }
        for Hkv in [1, 2]
    ],
]

for c in LLM_CASES.copy():
    for update in LLM_CASE_UPDATES:
        c = c.copy()
        c.update(update)
        LLM_CASES.append(c)

CASES = VISION_CASES + LLM_CASES


def create_tensors(shape_q, Hkv, dtype, requires_grad=False, packed=True):
    stacked_shape = list(shape_q)  # B, M, H, K
    Hq = shape_q[2]
    stacked_dim = 2 if packed else 0
    stacked_shape.insert(stacked_dim, 3)
    qkv = torch.rand(
        stacked_shape, device=device, dtype=dtype, requires_grad=requires_grad
    )
    q = torch.rand(shape_q, device=device, dtype=dtype, requires_grad=requires_grad)
    shape_kv = (shape_q[0], shape_q[1], Hkv, shape_q[3])
    k = (
        torch.rand(shape_kv, device=device, dtype=dtype, requires_grad=requires_grad)
        .reshape(shape_q[0], shape_q[1], 1, Hkv, shape_q[3])
        .expand(shape_q[0], shape_q[1], Hq // Hkv, Hkv, shape_q[3])
        .reshape(shape_q)
    )
    v = (
        torch.rand(shape_kv, device=device, dtype=dtype, requires_grad=requires_grad)
        .reshape(shape_q[0], shape_q[1], 1, Hkv, shape_q[3])
        .expand(shape_q[0], shape_q[1], Hq // Hkv, Hkv, shape_q[3])
        .reshape(shape_q)
    )

    return qkv, q, k, v


def mem_eff_attention_fw(
    shape_q,
    num_threads: int,
    attn_bias_cfg,
    dropout_p,
    dtype,
    packed=True,
    Hkv=None,
):
    B, M, Hq, K = shape_q
    Hkv = Hkv or Hq
    _, q, k, v = create_tensors(
        shape_q,
        Hkv,
        dtype,
        requires_grad=False,
        packed=packed,
    )
    attn_bias_type, attn_bias_requires_grad = attn_bias_cfg
    if attn_bias_requires_grad:
        return

    dtype_str = {
        torch.bfloat16: "b16",
        torch.half: "f16",
        torch.float: "f32",
    }[dtype]
    sub_label = (
        f"{dtype_str} {B}-{M}-{Hq}-{Hkv}-{K}, p={dropout_p}, "
        f"BiasT={attn_bias_type.__name__}"
    )

    has_run = False
    for fw_op, bw_op in OPS:
        bias = create_attn_bias(
            attn_bias_type,
            batch_size=B,
            num_heads=Hq,
            num_heads_groups=Hq // Hkv,
            q_len=M,
            kv_len=M,
            dtype=dtype,
            device=device,
            requires_grad=attn_bias_requires_grad,
            fmt="BMHK",
            op=fw_op,
        )
        inp = fmha.Inputs(query=q, key=k, value=v, attn_bias=bias, p=dropout_p)
        if isinstance(
            bias,
            (
                fmha.attn_bias.BlockDiagonalMask,
                fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask,
            ),
        ):
            q, k, v = [x.reshape([1, -1, *x.shape[2:]]) for x in [q, k, v]]
        if not fw_op.supports(inp):
            continue

        yield benchmark.Timer(
            stmt="fn(q, k, v, attn_bias, p)",
            globals={
                "q": q,
                "k": k,
                "v": v,
                "attn_bias": inp.attn_bias,
                "p": dropout_p,
                "fn": partial(
                    xformers.ops.memory_efficient_attention, op=(fw_op, bw_op)
                ),
            },
            label=f"attention (attn_bias={attn_bias_type})",
            description=fw_op.NAME,
            sub_label=sub_label,
            num_threads=num_threads,
        )
        has_run = True

    if not has_run:
        return

    yield benchmark.Timer(
        stmt="fn(q, k, v, attn_bias, p)",
        globals={
            "q": q,
            "k": k,
            "v": v,
            "attn_bias": inp.attn_bias,
            "p": dropout_p,
            "fn": ref_attention,
        },
        label=f"attention (attn_bias={attn_bias_type})",
        description="eager",
        sub_label=sub_label,
        num_threads=num_threads,
    )


def mem_eff_attention_bw(
    shape_q, num_threads: int, attn_bias_cfg, dropout_p, dtype, Hkv=None
):
    B, M, Hq, K = shape_q
    Hkv = Hkv or Hq
    _, q, k, v = create_tensors(
        shape_q,
        Hkv,
        dtype,
        requires_grad=True,
    )

    attn_bias_type, attn_bias_requires_grad = attn_bias_cfg

    dtype_str = {
        torch.bfloat16: "b16",
        torch.half: "f16",
        torch.float: "f32",
    }[dtype]
    sub_label = (
        f"{dtype_str} {B}-{M}-{Hq}-{Hkv}-{K}, p={dropout_p}, "
        f"BiasT={attn_bias_type.__name__}, BiasGrad={attn_bias_requires_grad}"
    )

    has_run = False
    for fw_op, bw_op in OPS:
        bias = create_attn_bias(
            attn_bias_type,
            batch_size=B,
            num_heads=Hq,
            num_heads_groups=Hq // Hkv,
            q_len=M,
            kv_len=M,
            dtype=dtype,
            device=device,
            requires_grad=attn_bias_requires_grad,
            fmt="BMHK",
            op=bw_op,
        )
        inp = fmha.Inputs(query=q, key=k, value=v, attn_bias=bias, p=dropout_p)

        if not fw_op.supports(inp) or not bw_op.supports(inp):
            continue
        has_run = True
        out = xformers.ops.memory_efficient_attention(
            inp.query, inp.key, inp.value, inp.attn_bias, inp.p, op=(fw_op, bw_op)
        )
        grad_benchmark = torch.ones_like(q)

        yield benchmark.Timer(
            stmt="out.backward(grad, retain_graph=True)",
            globals={
                "out": out,
                "grad": grad_benchmark,
            },
            label=f"attention backward (attn_bias={attn_bias_type})",
            description=bw_op.NAME,
            sub_label=sub_label,
            num_threads=num_threads,
        )
        del out

    if not has_run:
        return
    yield benchmark.Timer(
        stmt="out.backward(grad, retain_graph=True)",
        globals={
            "out": ref_attention(q, k, v, inp.attn_bias, dropout_p),
            "grad": grad_benchmark,
        },
        label=f"attention backward (attn_bias={attn_bias_type})",
        description="vanilla",
        sub_label=sub_label,
        num_threads=num_threads,
    )


def main():
    arg_parser = create_argparser()
    arg_parser.add_argument(
        "--omit-forward",
        action="store_true",
        help="Do not run forward benchmarks",
    )
    arg_parser.add_argument(
        "--omit-backward",
        action="store_true",
        help="Do not run backward benchmarks",
    )
    args = arg_parser.parse_args()
    if not args.omit_forward:
        benchmark_main_helper(
            mem_eff_attention_fw,
            CASES,
            arg_parser=arg_parser,
            min_run_time=min_run_time,
        )
    if not args.omit_backward:
        benchmark_main_helper(
            mem_eff_attention_bw,
            CASES,
            arg_parser=arg_parser,
            min_run_time=min_run_time,
        )


if __name__ == "__main__":
    main()
