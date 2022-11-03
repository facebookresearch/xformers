# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import itertools
from functools import partial, reduce

import timm
import torch
import torch.nn as nn
from timm.models.layers import Mlp as TimmMlp
from timm.models.vision_transformer import Attention as TimmAttention
from timm.models.vision_transformer import Block as TimmBlock
from torch.utils import benchmark
from utils import benchmark_main_helper

import xformers.ops as xops


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
        self.swiglu = xops.swiglu._SwiGLUModule(
            in_features=mlp.fc1.in_features,
            hidden_features=mlp.fc1.out_features,
            pack_weights=True,
            bias=True,
        )
        self.op = op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, M, C = x.shape

        x = x.reshape([B * M, C])
        x = xops.functional_swiglu(x, *self.swiglu._ordered_params_for_op(), op=self.op)
        x = x.reshape([B, M, C])

        return x


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
    ("ViT-B/16", timm.models.vit_base_patch16_224, [256, 3, 224, 224])
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


def benchmark_transformer(model_info, dtype):
    device = "cuda"

    model_name, model_factory, input_shape = model_info

    inp = torch.randn(input_shape, dtype=dtype, device=device)

    for mod_name, mod_apply in MODIFIERS:
        model: nn.Module = model_factory()
        model_modified = mod_apply(model).to(device).to(dtype)

        # Make sure we don't have errors
        out = model_modified(inp)
        grad = out.clone()
        out.backward(grad)

        yield benchmark.Timer(
            stmt="model(inp).backward(grad)",
            globals={
                "model": model_modified,
                "inp": inp,
                "grad": grad,
            },
            label="fw+bw",
            description=mod_name,
            sub_label=model_name,
        )


benchmark_main_helper(benchmark_transformer, CASES)
