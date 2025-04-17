# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, Type

import pytest
import torch

import xformers.ops
from xformers.ops import fmha

from .utils import disable_on_rocm, ref_attention_bmhk_for_test

compute_capability = (0, 0)
if torch.cuda.is_available():
    compute_capability = torch.cuda.get_device_capability("cuda")
sm90_or_better_only = pytest.mark.skipif(
    compute_capability < (9, 0), reason="requires sm90+"
)


@disable_on_rocm
@sm90_or_better_only
@pytest.mark.parametrize(
    "B,Mq,Mkv,Hq,Hkv,Kqk,Kv",
    [
        pytest.param(3, 13, 17, 7, 7, 128, 128, id="regular"),
        pytest.param(3, 13, 17, 7, 1, 128, 128, id="mqa"),
        # pytest.param(3, 13, 17, 21, 7, 128, 128, id="gqa"),  # unsupported
    ],
)
@pytest.mark.parametrize(
    "op",
    [
        (fmha.flash3.FwOp, fmha.flash3.BwOp),
    ],
    ids=lambda op: f"{op[0].NAME}-{op[1].NAME}",
)
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("varseq", [False, True], ids=["batched", "varseq"])
@pytest.mark.parametrize("worst_case", [False, True], ids=["tight", "loose"])
def test_flop_formula(
    B: int,
    Mq: int,
    Mkv: int,
    Hq: int,
    Hkv: int,
    Kqk: int,
    Kv: int,
    op: Tuple[Type[fmha.AttentionFwOpBase], Type[fmha.AttentionBwOpBase]],
    causal: bool,
    varseq: bool,
    worst_case: bool,
    monkeypatch,
):
    if (op[0] is fmha.flash3.FwOp and not op[0].is_available()) or (
        op[1] is fmha.flash3.BwOp and not op[1].is_available()
    ):
        pytest.skip("Flash3 not available")
    dtype = torch.float16

    if varseq:
        B = 1
    if causal:
        Mkv = Mq

    # No MQA/GQA in the reference impl
    ref_q = torch.randn(
        [B, Mq, Hq, Kqk], dtype=dtype, device="cuda", requires_grad=True
    )
    ref_k = torch.randn(
        [B, Mkv, Hq, Kqk], dtype=dtype, device="cuda", requires_grad=True
    )
    ref_v = torch.randn(
        [B, Mkv, Hq, Kv], dtype=dtype, device="cuda", requires_grad=True
    )

    with torch.utils.flop_counter.FlopCounterMode(display=False) as fc:
        ref_out = ref_attention_bmhk_for_test(ref_q, ref_k, ref_v, attn_bias=None)
    ref_fwd_flops = fc.get_total_flops()
    with torch.utils.flop_counter.FlopCounterMode(display=False) as fc:
        ref_out.backward(torch.randn_like(ref_out))
    ref_bwd_flops = fc.get_total_flops()

    q = torch.randn([B, Mq, Hq, Kqk], dtype=dtype, device="cuda", requires_grad=True)
    k = torch.randn([B, Mkv, Hkv, Kqk], dtype=dtype, device="cuda", requires_grad=True)
    v = torch.randn([B, Mkv, Hkv, Kv], dtype=dtype, device="cuda", requires_grad=True)

    if Hkv == 1:
        k = k.expand(-1, -1, Hq, -1)
        v = v.expand(-1, -1, Hq, -1)
    elif 1 < Hkv < Hq:
        G = Hq // Hkv
        q = q.unflatten(2, (Hkv, -1))
        k = k.unflatten(2, (Hkv, -1)).expand(-1, -1, -1, G, -1)
        v = v.unflatten(2, (Hkv, -1)).expand(-1, -1, -1, G, -1)

    if varseq:
        seqlens = ([5, Mq - 5], [5, Mkv - 5])

    bias: Optional[fmha.attn_bias.AttentionBias]
    if varseq and causal:
        bias = fmha.attn_bias.BlockDiagonalCausalFromBottomRightMask.from_seqlens(
            *seqlens
        )
    elif varseq:
        bias = fmha.attn_bias.BlockDiagonalMask.from_seqlens(*seqlens)
    elif causal:
        bias = fmha.attn_bias.LowerTriangularMask()
    else:
        bias = None

    bias_for_flops: Optional[fmha.attn_bias.AttentionBias] = None
    if worst_case:
        if causal:
            bias_for_flops = fmha.attn_bias.LowerTriangularMask()
    else:
        bias_for_flops = bias
    if bias_for_flops is not None:
        flops_div = Mkv * Mq
        flops_mul = (
            (bias_for_flops.materialize((Mq, Mkv), device="cpu") == 0)
            .int()
            .sum()
            .item()
        )
        ref_fwd_flops = ref_fwd_flops * flops_mul // flops_div
        ref_bwd_flops = ref_bwd_flops * flops_mul // flops_div

    if worst_case:
        monkeypatch.setenv("XFORMERS_FLOP_FORMULA_WORST_CASE", "1")
    else:
        # We disable it explicitly in case it's set in the user's environment
        monkeypatch.setenv("XFORMERS_FLOP_FORMULA_WORST_CASE", "0")

    with torch.utils.flop_counter.FlopCounterMode(display=False) as fc:
        out = xformers.ops.memory_efficient_attention(q, k, v, op=op, attn_bias=bias)
    assert fc.get_total_flops() == ref_fwd_flops

    with torch.utils.flop_counter.FlopCounterMode(display=False) as fc:
        out.backward(torch.randn_like(out))
    # Flash's backward recomputes the first matmul of the fwd.
    assert fc.get_total_flops() == ref_bwd_flops + ref_fwd_flops / 2


def test_mask_nonzeros() -> None:
    assert fmha.flash3.mask_non_zeros(13, 17, -1, 0) == 143
    assert fmha.flash3.mask_non_zeros(13, 17, -1, -1) == 221

    assert fmha.flash3.mask_non_zeros(8, 8, 32, 32) == 64
    assert fmha.flash3.mask_non_zeros(8, 8, 4, 3) == 48
    assert fmha.flash3.mask_non_zeros(8, 8, 4, 0) == 30
    assert fmha.flash3.mask_non_zeros(8, 8, -1, -1) == 64
