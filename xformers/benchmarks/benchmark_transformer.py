# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import itertools
from functools import partial, reduce
from typing import Iterator

import timm
import torch
import torch.nn as nn
from timm.models.layers import Mlp as TimmMlp
from timm.models.vision_transformer import Attention as TimmAttention
from timm.models.vision_transformer import Block as TimmBlock
from torch.utils import benchmark

import xformers.ops as xops
from xformers.benchmarks.utils import benchmark_main_helper


def replace_module(module: nn.Module, replace_class, factory):
    if isinstance(module, replace_class):
        return factory(module)
    module_output = module
    for name, child in module.named_children():
        module_output.add_module(name, replace_module(child, replace_class, factory))
    del module
    return module_output


class TimmMemEffAttention(nn.Module):
    def __init__(self, attn: TimmAttention, op=None):
        super().__init__()
        self.op = None
        self.num_heads = attn.num_heads
        self.scale = attn.scale

        self.qkv = attn.qkv
        self.attn_drop = attn.attn_drop
        self.proj = attn.proj
        self.proj_drop = attn.proj_drop

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = xops.unbind(qkv, dim=2)

        x = xops.memory_efficient_attention(q, k, v, op=self.op).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TimmSwiGLU(nn.Module):
    def __init__(self, mlp: TimmMlp, op=None) -> None:
        super().__init__()
        self.fc1 = mlp.fc1
        self.swiglu = xops.SwiGLU(
            in_features=mlp.fc1.in_features,
            hidden_features=mlp.fc1.out_features,
            bias=True,
        )
        self.op = op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swiglu(x)


def mod_memeff_attn(model: nn.Module, op=None) -> nn.Module:
    return replace_module(model, TimmAttention, partial(TimmMemEffAttention, op=op))


def mod_mlp_to_swiglu(model: nn.Module, op=None) -> nn.Module:
    def _mlp_to_swiglu(block: TimmBlock):
        block.mlp = TimmSwiGLU(block.mlp, op=op)
        return block

    return replace_module(model, TimmBlock, _mlp_to_swiglu)


mod_mlp_to_eagr_swiglu = partial(mod_mlp_to_swiglu, op=xops.SwiGLUEagerOp)
mod_mlp_to_fast_swiglu = partial(mod_mlp_to_swiglu, op=None)


def compose(*fns):
    def compose2(f, g):
        return lambda *a, **kw: f(g(*a, **kw))

    return reduce(compose2, fns)


MODELS = [
    # model_name, model_factory, input_shape
    ("ViT-B/16", timm.models.vit_base_patch16_224, [512, 3, 224, 224]),
    ("ViT-B/8", timm.models.vit_base_patch8_224, [64, 3, 224, 224]),
    ("ViT-L/16", timm.models.vit_large_patch16_224, [128, 3, 224, 224]),
    ("ViT-g/14", timm.models.vit_giant_patch14_224, [32, 3, 224, 224]),
]

MODIFIERS = [
    ["mlp", lambda x: x],
    ["mlp+memeff", compose(mod_mlp_to_fast_swiglu, mod_memeff_attn)],
    ["swiglu", mod_mlp_to_eagr_swiglu],
    ["swiglu+fast_swiglu", mod_mlp_to_fast_swiglu],
    ["swiglu+fast_swiglu+memeff", compose(mod_mlp_to_fast_swiglu, mod_memeff_attn)],
]


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


CASES = list(
    product_dict(
        model_info=MODELS,
        dtype=[torch.half],
    )
)


def benchmark_transformer(model_info, dtype) -> Iterator[benchmark.Timer]:
    device = "cuda"

    model_name, model_factory, input_shape = model_info

    inp = torch.randn(input_shape, dtype=dtype, device=device)

    for mod_name, mod_apply in MODIFIERS:
        model: nn.Module = model_factory()
        model = mod_apply(model).to(device).to(dtype)

        # Make sure we don't have errors
        out = model(inp)
        grad = out.clone()
        out.backward(grad)

        yield benchmark.Timer(
            stmt="model(inp).backward(grad)",
            globals={
                "model": model,
                "inp": inp,
                "grad": grad,
            },
            label="fw+bw",
            description=mod_name,
            sub_label=model_name,
        )


if torch.version.hip:
    print("This benchmark could not be done on ROCM!")
else:
    benchmark_main_helper(benchmark_transformer, CASES)
