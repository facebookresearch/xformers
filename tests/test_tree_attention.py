# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from functools import reduce
from itertools import accumulate
from typing import List, Optional, Tuple, Type

import pytest
import torch

from xformers.ops import fmha
from xformers.ops.fmha.attn_bias import (
    BlockDiagonalPaddedKeysMask,
    PagedBlockDiagonalPaddedKeysMask,
)
from xformers.ops.fmha.common import AttentionFwOpBase
from xformers.ops.tree_attention import (
    construct_full_tree_choices,
    construct_tree_choices,
    tree_attention,
    TreeAttnMetadata,
    use_triton_splitk_for_prefix,
)
from xformers.utils import do_bench_cudagraph

compute_capability = (0, 0)
if torch.cuda.is_available():
    compute_capability = torch.cuda.get_device_capability("cuda")
sm80_or_better_only = pytest.mark.skipif(
    compute_capability < (8, 0), reason="requires sm80+"
)


# fmt: off
# Tree defintions - see doctring of
# xformers.ops.tree_attention.TreeAttnMetadata.from_tree_choices
# for the format description
# from https://github.com/SafeAILab/EAGLE/blob/e98fc7c/model/choices.py
eagle_mc_sim_7b_63 = [(0,),(1,),(2,),(3,),(0,0),(0,1),(0,2),(1,0),(1,1),(2,0),(2,1),(3,0)  # noqa
                ,(0,0,0),(0,0,1),(0,0,2),(0,1,0),(0,1,1),(0,2,0),(0,2,1),(1,0,0),  # noqa
                (0,0,0,0),(0,0,0,1),(0,0,0,2),(0,0,0,0,0),(0,0,0,0,1)]  # noqa

# Medusa choices from
# https://github.com/FasterDecoding/Medusa/blob/5e98053/medusa/model/medusa_choices.py
mc_sim_7b_63 = [(0,), (0, 0), (1,), (0, 1), (2,), (0, 0, 0), (1, 0), (0, 2), (3,), (0, 3), (4,), (0, 4), (2, 0), (0, 5), (0, 0, 1), (5,), (0, 6), (6,), (0, 7), (0, 1, 0), (1, 1), (7,), (0, 8), (0, 0, 2), (3, 0), (0, 9), (8,), (9,), (1, 0, 0), (0, 2, 0), (1, 2), (0, 0, 3), (4, 0), (2, 1), (0, 0, 4), (0, 0, 5), (0, 0, 0, 0), (0, 1, 1), (0, 0, 6), (0, 3, 0), (5, 0), (1, 3), (0, 0, 7), (0, 0, 8), (0, 0, 9), (6, 0), (0, 4, 0), (1, 4), (7, 0), (0, 1, 2), (2, 0, 0), (3, 1), (2, 2), (8, 0), (0, 5, 0), (1, 5), (1, 0, 1), (0, 2, 1), (9, 0), (0, 6, 0), (0, 0, 0, 1), (1, 6), (0, 7, 0)]  # noqa
vicuna_7b_stage2 = [(0,), (0, 0), (1,), (0, 1), (0, 0, 0), (1, 0), (2,), (0, 2), (0, 0, 1), (0, 3), (3,), (0, 1, 0), (2, 0), (4,), (0, 0, 2), (0, 4), (1, 1), (1, 0, 0), (0, 0, 0, 0), (5,), (0, 0, 3), (0, 5), (0, 2, 0), (3, 0), (0, 1, 1), (0, 6), (6,), (0, 7), (0, 0, 4), (4, 0), (1, 2), (0, 8), (7,), (0, 3, 0), (0, 0, 0, 1), (0, 0, 5), (2, 1), (0, 0, 6), (1, 0, 1), (0, 0, 1, 0), (2, 0, 0), (5, 0), (0, 9), (0, 1, 2), (8,), (0, 4, 0), (0, 2, 1), (1, 3), (0, 0, 7), (0, 0, 0, 2), (0, 0, 8), (1, 1, 0), (0, 1, 0, 0), (6, 0), (9,), (0, 1, 3), (0, 0, 0, 3), (1, 0, 2), (0, 5, 0), (3, 1), (0, 0, 2, 0), (7, 0), (1, 4)]  # noqa
vicuna_7b_stage1_ablation = [(0,), (0, 0), (1,), (0, 0, 0), (0, 1), (1, 0), (2,), (0, 2), (0, 0, 1), (3,), (0, 3), (0, 1, 0), (2, 0), (0, 0, 2), (0, 4), (4,), (0, 0, 0, 0), (1, 0, 0), (1, 1), (0, 0, 3), (0, 2, 0), (0, 5), (5,), (3, 0), (0, 1, 1), (0, 6), (6,), (0, 0, 4), (1, 2), (0, 0, 0, 1), (4, 0), (0, 0, 5), (0, 7), (0, 8), (0, 3, 0), (0, 0, 1, 0), (1, 0, 1), (7,), (2, 0, 0), (0, 0, 6), (2, 1), (0, 1, 2), (5, 0), (0, 2, 1), (0, 9), (0, 0, 0, 2), (0, 4, 0), (8,), (1, 3), (0, 0, 7), (0, 1, 0, 0), (1, 1, 0), (6, 0), (9,), (0, 0, 8), (0, 0, 9), (0, 5, 0), (0, 0, 2, 0), (1, 0, 2), (0, 1, 3), (0, 0, 0, 3), (3, 0, 0), (3, 1)]  # noqa
vicuna_7b_stage1 = [(0,), (0, 0), (1,), (2,), (0, 1), (1, 0), (3,), (0, 2), (4,), (0, 0, 0), (0, 3), (5,), (2, 0), (0, 4), (6,), (0, 5), (1, 1), (0, 0, 1), (7,), (3, 0), (0, 6), (8,), (9,), (0, 1, 0), (0, 7), (0, 8), (4, 0), (0, 0, 2), (1, 2), (0, 9), (2, 1), (5, 0), (1, 0, 0), (0, 0, 3), (1, 3), (0, 2, 0), (0, 1, 1), (0, 0, 4), (6, 0), (1, 4), (0, 0, 5), (2, 2), (0, 3, 0), (3, 1), (0, 0, 6), (7, 0), (1, 5), (1, 0, 1), (2, 0, 0), (0, 0, 7), (8, 0), (0, 0, 0, 0), (4, 1), (0, 1, 2), (0, 4, 0), (9, 0), (0, 2, 1), (2, 3), (1, 6), (0, 0, 8), (0, 5, 0), (3, 2), (5, 1)]  # noqa
vicuna_13b_stage2 = [(0,), (0, 0), (1,), (0, 0, 0), (0, 1), (1, 0), (2,), (0, 2), (0, 0, 1), (0, 1, 0), (3,), (0, 3), (2, 0), (0, 0, 2), (0, 0, 0, 0), (0, 4), (1, 0, 0), (1, 1), (4,), (0, 0, 3), (0, 5), (0, 2, 0), (5,), (3, 0), (0, 1, 1), (0, 6), (0, 0, 4), (0, 0, 0, 1), (0, 7), (0, 0, 5), (1, 2), (0, 0, 1, 0), (0, 3, 0), (1, 0, 1), (4, 0), (0, 0, 6), (0, 8), (2, 0, 0), (0, 9), (6,), (7,), (2, 1), (5, 0), (0, 1, 2), (0, 0, 0, 2), (8,), (0, 4, 0), (0, 1, 0, 0), (0, 2, 1), (0, 0, 7), (1, 1, 0), (1, 3), (0, 0, 2, 0), (9,), (0, 0, 8), (0, 5, 0), (0, 0, 0, 3), (0, 0, 9), (0, 1, 3), (1, 0, 2), (0, 0, 1, 1), (3, 0, 0), (1, 0, 0, 0)]  # noqa
vicuna_13b_stage1 = [(0,), (0, 0), (1,), (0, 1), (2,), (1, 0), (0, 0, 0), (0, 2), (3,), (0, 3), (4,), (2, 0), (0, 4), (0, 0, 1), (0, 5), (5,), (1, 1), (0, 1, 0), (6,), (0, 6), (0, 0, 2), (7,), (3, 0), (8,), (0, 7), (0, 8), (1, 0, 0), (0, 0, 3), (4, 0), (1, 2), (9,), (0, 9), (2, 1), (0, 2, 0), (0, 0, 4), (1, 3), (0, 1, 1), (0, 0, 5), (5, 0), (0, 3, 0), (0, 0, 0, 0), (0, 0, 6), (6, 0), (1, 4), (2, 0, 0), (0, 1, 2), (3, 1), (0, 4, 0), (1, 0, 1), (2, 2), (0, 0, 7), (1, 5), (7, 0), (0, 0, 8), (8, 0), (0, 5, 0), (0, 0, 9), (0, 2, 1), (1, 1, 0), (0, 1, 3), (4, 1), (2, 3), (1, 6)]  # noqa
vicuna_33b_stage2 = [(0,), (0, 0), (1,), (0, 1), (0, 0, 0), (1, 0), (2,), (0, 2), (0, 0, 1), (0, 3), (3,), (0, 1, 0), (2, 0), (0, 4), (4,), (0, 0, 2), (1, 1), (1, 0, 0), (0, 5), (5,), (0, 0, 0, 0), (0, 0, 3), (3, 0), (0, 2, 0), (0, 6), (0, 1, 1), (6,), (0, 0, 4), (0, 7), (7,), (1, 2), (4, 0), (8,), (0, 3, 0), (0, 0, 5), (0, 0, 0, 1), (0, 8), (2, 1), (0, 9), (1, 0, 1), (2, 0, 0), (0, 0, 6), (5, 0), (0, 0, 1, 0), (1, 3), (0, 1, 2), (0, 4, 0), (0, 0, 7), (0, 2, 1), (9,), (1, 1, 0), (0, 0, 0, 2), (6, 0), (0, 0, 8), (0, 1, 0, 0), (7, 0), (0, 1, 3), (0, 5, 0), (1, 4), (0, 0, 9), (3, 1), (1, 0, 2), (2, 2)]  # noqa
vicuna_33b_stage1 = [(0,), (1,), (0, 0), (2,), (0, 1), (3,), (1, 0), (4,), (0, 2), (5,), (0, 3), (0, 0, 0), (6,), (0, 4), (2, 0), (7,), (1, 1), (0, 5), (3, 0), (8,), (9,), (0, 6), (0, 7), (0, 0, 1), (1, 2), (4, 0), (0, 1, 0), (0, 8), (0, 9), (2, 1), (0, 0, 2), (5, 0), (1, 3), (0, 0, 3), (1, 0, 0), (1, 4), (6, 0), (0, 2, 0), (3, 1), (2, 2), (0, 0, 4), (7, 0), (0, 1, 1), (1, 5), (4, 1), (0, 0, 5), (0, 3, 0), (9, 0), (8, 0), (1, 6), (0, 0, 6), (2, 3), (0, 1, 2), (3, 2), (0, 4, 0), (2, 0, 0), (1, 7), (1, 0, 1), (0, 0, 7), (5, 1), (2, 4), (0, 0, 8), (0, 2, 1)]  # noqa
zephyr_stage2 = [(0,), (0, 0), (1,), (0, 1), (2,), (0, 0, 0), (1, 0), (0, 2), (3,), (0, 3), (4,), (2, 0), (0, 0, 1), (0, 4), (5,), (0, 5), (0, 1, 0), (1, 1), (6,), (0, 0, 2), (3, 0), (0, 6), (7,), (0, 7), (0, 8), (0, 0, 3), (1, 0, 0), (0, 9), (0, 2, 0), (1, 2), (4, 0), (8,), (9,), (2, 1), (0, 1, 1), (0, 0, 4), (0, 0, 0, 0), (5, 0), (0, 3, 0), (1, 3), (0, 0, 5), (0, 0, 6), (6, 0), (2, 0, 0), (1, 0, 1), (0, 1, 2), (0, 4, 0), (1, 4), (3, 1), (2, 2), (0, 0, 7), (7, 0), (0, 2, 1), (0, 0, 8), (0, 1, 3), (0, 5, 0), (1, 5), (0, 0, 9), (1, 1, 0), (0, 0, 0, 1), (0, 0, 1, 0), (4, 1), (2, 3)]  # noqa

medusa_choices = [
    mc_sim_7b_63,
    vicuna_7b_stage2,
    vicuna_7b_stage1_ablation,
    vicuna_7b_stage1,
    vicuna_13b_stage2,
    vicuna_13b_stage1,
    vicuna_33b_stage2,
    vicuna_33b_stage1,
    zephyr_stage2,
]

# fmt: on


@sm80_or_better_only
@pytest.mark.parametrize(
    "tree_choices",
    [
        eagle_mc_sim_7b_63,
        *medusa_choices,
        construct_full_tree_choices(1, 1),
        construct_full_tree_choices(1, 4),
        construct_full_tree_choices(4, 1),
        construct_full_tree_choices(3, 2),
        construct_full_tree_choices(2, 3),
        construct_full_tree_choices(5, 3),
        construct_full_tree_choices(10, 2),
    ],
)
@pytest.mark.parametrize("B", [1, 16])
@pytest.mark.parametrize("G", [1, 2])
@pytest.mark.parametrize("square", [True, False])
@pytest.mark.parametrize("paged", [True, False])
@torch.no_grad()
def test_tree_attention(
    tree_choices: List[Tuple[int, ...]], B: int, G: int, square: bool, paged: bool
) -> None:
    H = 8
    D = 128
    Mk = 8192
    dtype = torch.bfloat16
    tree_size_q = len(tree_choices) + 1 if square else max(len(tree_choices) // 2, 1)
    run_tree_attention_inner(
        tree_choices, B, Mk, G, H, D, tree_size_q, dtype, paged=paged
    )


class SplitKAutotune(fmha.triton_splitk.FwOp):
    AUTOTUNE = True


def run_tree_attention_inner(
    tree_choices: List[Tuple[int, ...]],
    B: int,
    Mk: int,
    G: int,
    H: int,
    D: int,
    tree_size_q: int,
    dtype: torch.dtype,
    benchmark: bool = False,
    autotune: bool = False,
    randomize_lengths: bool = True,
    paged: bool = False,
) -> None:
    """
    Test Medusa-style tree attention.
    """
    if (
        paged
        and PagedBlockDiagonalPaddedKeysMask
        not in SplitKAutotune.SUPPORTED_ATTN_BIAS_TYPES
    ):
        pytest.skip("Does not supported paged attention bias")

    torch.manual_seed(0)
    # + 1 comes from the root node
    tree_size_kv = len(tree_choices) + 1

    # Simulate output of speculative heads
    q_full = torch.randn([B, tree_size_kv, G, H, D], device="cuda", dtype=dtype)
    q = q_full[:, -tree_size_q:].clone()
    spec_k = torch.randn([B, tree_size_kv, G, 1, D], device="cuda", dtype=dtype)
    spec_v = torch.randn_like(spec_k)
    spec_k = spec_k.expand(-1, -1, -1, H, -1)
    spec_v = spec_v.expand(-1, -1, -1, H, -1)

    # Create K/V cache before the speculative tokens
    cache_k = torch.randn([B, Mk, G, 1, D], device="cuda", dtype=dtype)
    cache_v = torch.randn_like(cache_k)
    cache_k = cache_k.expand(-1, -1, -1, H, -1)
    cache_v = cache_v.expand(-1, -1, -1, H, -1)

    if randomize_lengths:
        kv_lens_device = torch.randint(1, Mk + 1, size=(B,), device=q.device)
    else:
        kv_lens_device = torch.full(size=(B,), fill_value=Mk, device=q.device)
    kv_lens = kv_lens_device.tolist()

    triton_splitk_op = SplitKAutotune if autotune else fmha.triton_splitk.FwOp

    # Compute attention on the full context using merge_attentions: it will use
    # padded non-causal block-diagonal on the first part (before spec tokens) and
    # explicit attn_bias mask on the second part (spec tokens)
    attn = tree_attention_with_sync(
        q,
        spec_k,
        spec_v,
        cache_k,
        cache_v,
        tree_choices,
        kv_lens,
        prefix_op=triton_splitk_op,
        suffix_op=triton_splitk_op,
        paged=paged,
    )

    tree_attn_metadata = TreeAttnMetadata.from_tree_choices(
        tree_choices, q.dtype, q.device
    )

    t_optimized_ms_fa = 0.0
    t_optimized_ms_triton = 0.0
    if benchmark:
        # Construct attention bias for "left" part of the attention - the exising K/V context
        prefix_attn_bias = fmha.attn_bias.BlockDiagonalPaddedKeysMask.from_seqlens(
            q_seqlen=[tree_size_q for _ in range(B)], kv_seqlen=kv_lens, kv_padding=Mk
        )
        # Create an explit attention bias for the speculative part of the attention
        spec_attn_bias = tree_attn_metadata.attention_bias

        torch.cuda.synchronize()
        bench_stream = torch.cuda.Stream()
        with torch.cuda.stream(bench_stream):
            t_optimized_ms_triton = do_bench_cudagraph(
                lambda: tree_attention(
                    q,
                    spec_k,
                    spec_v,
                    cache_k,
                    cache_v,
                    spec_attn_bias,
                    prefix_attn_bias,
                    prefix_op=triton_splitk_op,
                    suffix_op=triton_splitk_op,
                )
            )

        torch.cuda.synchronize()
        bench_stream = torch.cuda.Stream()
        with torch.cuda.stream(bench_stream):
            t_optimized_ms_fa = do_bench_cudagraph(
                lambda: tree_attention(
                    q,
                    spec_k,
                    spec_v,
                    cache_k,
                    cache_v,
                    spec_attn_bias,
                    prefix_attn_bias,
                    prefix_op=fmha.flash.FwOp if torch.version.cuda else fmha.ck.FwOp,
                    suffix_op=triton_splitk_op,
                )
            )

    # Compute attention on the full context in a slow way, unrolling every path in the tree
    paths_w_unpadded_indices = [
        path.tolist()[:path_len]
        for path, path_len in zip(
            tree_attn_metadata.retrieval_indices, tree_attn_metadata.path_lengths
        )
    ]

    # Reference implementation, which does path unrolling, will compute attention on the full query
    # of length tree_size_kv > tree_size_q.
    # Then we'll pick only relevant elements to compare with the optimized implementation.
    paths_w_unpadded_indices_q = [
        [x - tree_size_kv + tree_size_q for x in path]
        for path in paths_w_unpadded_indices
    ]
    paths_w_unpadded_indices_q_mask = [
        x >= 0 for path in paths_w_unpadded_indices_q for x in path
    ]

    paths_w_unpadded_indices_q = [
        [x for x in path if x >= 0] for path in paths_w_unpadded_indices_q
    ]
    attn_unrolled = torch.cat(
        [attn[:, path, :, :] for path in paths_w_unpadded_indices_q], dim=1
    )
    ref_attn_full = ref_tree_attention(
        q_full, spec_k, spec_v, cache_k, cache_v, paths_w_unpadded_indices, kv_lens
    )
    ref_attn = ref_attn_full[:, paths_w_unpadded_indices_q_mask, :, :]

    if benchmark:
        # Here we compute vanilla attention with the the shapes similar to the tree attention we benchmarked above.
        # The output of this vanilla attention will be different - we are just measuring the runtime to get
        # some order of magnitude estimation of runtime.
        torch.cuda.synchronize()
        attn_bias_ref = (
            fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
                q_seqlen=[tree_size_q for _ in range(B)],
                kv_seqlen=kv_lens,
                kv_padding=Mk,
            )
        )
        bench_stream = torch.cuda.Stream()
        with torch.cuda.stream(bench_stream):
            t_ref_ms = do_bench_cudagraph(
                lambda: fmha.memory_efficient_attention_forward(
                    q.view(1, B * tree_size_q, G, H, D),
                    cache_k.view(1, B * Mk, G, H, D),
                    cache_v.view(1, B * Mk, G, H, D),
                    attn_bias=attn_bias_ref,
                )
            )

        gap = (1 - t_ref_ms / min(t_optimized_ms_fa, t_optimized_ms_triton)) * 100
        triton_faster = t_optimized_ms_fa > t_optimized_ms_triton
        triton_chosen = torch.version.hip is not None or use_triton_splitk_for_prefix(
            B, G, tree_size_kv
        )
        print(
            f"{B=}, {G=}, {Mk=}, {tree_size_q=}, {tree_size_kv=}: "
            f"Tree attention with FA2/CK took {t_optimized_ms_fa * 1e3:.1f}us, "
            f"with Triton Split-K took {t_optimized_ms_triton * 1e3:.1f}us, "
            f"vanilla attention took {t_ref_ms * 1e3:.1f}us, gap {gap:.2f}%. "
            f"Triton faster: {triton_faster}. "
            f"Choice optimal: {triton_chosen == triton_faster}"
        )

    torch.testing.assert_close(attn_unrolled, ref_attn, atol=1e-2, rtol=3e-3)


def ref_tree_attention(
    q: torch.Tensor,
    spec_k: torch.Tensor,
    spec_v: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    paths_w_unpadded_indices: List[List[int]],
    kv_lens: List[int],
) -> torch.Tensor:

    attns = []

    B, _, G, H, D = q.shape

    # Compute attention for every path separately and then concatenate

    for path in paths_w_unpadded_indices:
        q_path = q[:, path, ...]

        extra_k = torch.empty_like(spec_k[:, path, ...])
        extra_v = torch.empty_like(extra_k)

        full_k = torch.cat([cache_k, extra_k], dim=1)
        full_v = torch.cat([cache_v, extra_v], dim=1)

        tree_depth = len(path)

        for b in range(B):
            full_k[b, kv_lens[b] : kv_lens[b] + tree_depth] = spec_k[b, path]
            full_v[b, kv_lens[b] : kv_lens[b] + tree_depth] = spec_v[b, path]

        Mk = full_k.shape[1]
        attn_bias = (
            fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
                q_seqlen=[tree_depth for _ in range(B)],
                kv_seqlen=[s + tree_depth for s in kv_lens],
                kv_padding=Mk,
            )
        )
        attn_path = fmha.memory_efficient_attention_forward(
            q_path.view(1, B * tree_depth, G, H, D),
            full_k.view(1, B * Mk, G, H, D),
            full_v.view(1, B * Mk, G, H, D),
            attn_bias=attn_bias,
        )
        attn_path = attn_path.reshape(B, tree_depth, G, H, D)

        attns.append(attn_path)

    attn = torch.cat(attns, dim=1)

    return attn


def tree_attention_with_sync(
    q: torch.Tensor,
    spec_k: torch.Tensor,
    spec_v: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    tree_choices: List[Tuple[int, ...]],
    kv_lens: List[int],
    prefix_op: Optional[Type[AttentionFwOpBase]] = None,
    suffix_op: Optional[Type[AttentionFwOpBase]] = None,
    paged: bool = False,
) -> torch.Tensor:
    """
    A wrapper around tree_attention which constructs the biases.
    Arguments are the same as in xformers.ops.tree_attention.tree_attention, but instead of
    spec_attn_bias and prefix_attn_bias this function takes in tree definition in the form of tree_choices
    and K/V sequence lengths kv_lens.
    For the format of tree_choices see docstring of
    xformers.ops.tree_attention.TreeAttnMetadata.from_tree_choices.
    """
    B, tree_size_q = q.shape[:2]
    Mk = cache_k.shape[1]

    # attn_prefix ~ (B, Mk, H, D)
    prefix_attn_bias = BlockDiagonalPaddedKeysMask.from_seqlens(
        q_seqlen=[tree_size_q for _ in range(B)], kv_seqlen=kv_lens, kv_padding=Mk
    )

    if paged:
        # Create a paged K/V cache by randomly permuting blocks and storing the permutation in block_tables.
        page_size = 256
        assert Mk % page_size == 0
        max_blocks_per_row = Mk // page_size

        block_tables = torch.randperm(
            B * max_blocks_per_row, device=q.device, dtype=torch.int32
        ).view(B, max_blocks_per_row)

        cache_v = cache_v.clone()
        cache_k = cache_k.clone()

        v_view = cache_v.view(1, -1, page_size, *cache_v.shape[2:])
        k_view = cache_k.view(1, -1, page_size, *cache_k.shape[2:])

        block_tables_expanded = (
            block_tables.to(torch.int64)
            .flatten()[None, :, None, None, None, None]
            .expand(v_view.shape)
        )
        v_view.scatter_(1, block_tables_expanded, v_view.clone())
        k_view.scatter_(1, block_tables_expanded, k_view.clone())

        cache_k = k_view.view(1, -1, *cache_k.shape[2:])
        cache_v = v_view.view(1, -1, *cache_k.shape[2:])

        prefix_attn_bias = prefix_attn_bias.make_paged(  # type: ignore
            block_tables,
            page_size=page_size,
            paged_type=PagedBlockDiagonalPaddedKeysMask,
        )

    # Create an explit attention bias for the speculative part of the attention
    spec_attn_bias = TreeAttnMetadata.from_tree_choices(
        tree_choices, q.dtype, q.device
    ).attention_bias
    return tree_attention(
        q,
        spec_k,
        spec_v,
        cache_k,
        cache_v,
        spec_attn_bias[-tree_size_q:],
        prefix_attn_bias,
        prefix_op,
        suffix_op,
    )


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
@pytest.mark.parametrize("branching", [1, 2, 3, 4])
def test_tree_attention_metadata_full_tree(depth: int, branching: int) -> None:
    """
    Here we test that fields tree_seq_position_ids, tree_indices, and path_lengths
    are constructed correctly; attention_bias and retrieval_indices are tested in test_tree_attention.
    """
    tree_choices = construct_full_tree_choices(depth, branching)
    tree_attn_metadata = TreeAttnMetadata.from_tree_choices(tree_choices)

    num_paths = branching**depth
    assert len(tree_attn_metadata.path_lengths) == num_paths
    assert all(length == depth + 1 for length in tree_attn_metadata.path_lengths)

    seq_pos = []
    for i in range(depth + 1):
        seq_pos.extend([i] * branching**i)

    assert seq_pos == tree_attn_metadata.tree_seq_position_ids.tolist()

    tree_indices = []
    for i in range(depth):
        tree_indices.extend(
            [i * branching + j + 1 for j in range(branching)] * branching**i
        )

    assert [0] + tree_indices == tree_attn_metadata.tree_indices.tolist()

    tree_size = len(tree_choices) + 1
    ref_child_node_indices = torch.arange(tree_size * branching).reshape(
        tree_size, branching
    )
    ref_child_node_indices[ref_child_node_indices >= tree_size - 1] = 0
    torch.testing.assert_allclose(
        tree_attn_metadata.child_node_indices, ref_child_node_indices
    )

    ref_num_children_per_node = torch.full_like(
        tree_attn_metadata.tree_indices, fill_value=branching
    )
    ref_num_children_per_node[-num_paths:] = 0
    torch.testing.assert_close(
        tree_attn_metadata.num_children_per_node, ref_num_children_per_node
    )
    ref_num_nodes_per_level = torch.tensor([branching**i for i in range(depth + 1)])
    torch.testing.assert_close(
        tree_attn_metadata.num_nodes_per_level, ref_num_nodes_per_level
    )

    ref_subtree_sizes = (
        torch.tensor([branching**i for i in range(depth + 1)]).cumsum(dim=0).tolist()
    )
    assert all(
        subtree_size == subtree_sizes_ref
        for subtree_size, subtree_sizes_ref in zip(
            tree_attn_metadata.subtree_sizes, ref_subtree_sizes
        )
    )

    ref_num_children_per_node_at_level: List[torch.Tensor] = [
        torch.tensor([branching if i < depth else 0] * (branching**i))
        for i in range(0, depth + 1)
    ]
    assert all(
        torch.allclose(num_children, ref_num_children)
        for num_children, ref_num_children in zip(
            tree_attn_metadata.num_children_per_node_at_level,
            ref_num_children_per_node_at_level,
        )
    )


@pytest.mark.parametrize("branching", [[2, 3], [2, 3, 2], [2] * 3, [4, 3, 2, 1]])
def test_tree_attention_metadata_arbitrary_tree(branching: List[int]) -> None:
    tree_choices = construct_tree_choices(branching)
    tree_attn_metadata = TreeAttnMetadata.from_tree_choices(tree_choices)
    assert all(
        length == len(branching) + 1 for length in tree_attn_metadata.path_lengths
    )

    num_paths = reduce(lambda x, y: x * y, branching)
    assert len(tree_attn_metadata.path_lengths) == num_paths

    num_nodes_per_level = [*accumulate(branching, lambda x, y: x * y)]
    seq_pos = []
    for i in range(len(num_nodes_per_level)):
        seq_pos.extend([i + 1] * num_nodes_per_level[i])
    assert [0] + seq_pos == tree_attn_metadata.tree_seq_position_ids.tolist()

    tree_indices = []
    start_idx = 1
    num_children_per_node_ref = []
    num_children_per_node_at_level_ref: List[torch.Tensor] = []
    subtree_sizes_ref: List[int] = []
    total_num_nodes = 0
    for i in range(len(branching)):
        num_nodes_prev_level = 1 if i == 0 else num_nodes_per_level[i - 1]
        tree_indices.extend(
            [start_idx + j for j in range(branching[i])] * num_nodes_prev_level
        )
        start_idx += branching[i]
        num_children_per_node_ref.extend([branching[i]] * num_nodes_prev_level)
        num_children_per_node_at_level_ref.append(
            torch.tensor([branching[i]] * num_nodes_prev_level)
        )
        total_num_nodes += num_nodes_prev_level
        subtree_sizes_ref.append(total_num_nodes)
    num_children_per_node_at_level_ref.append(
        torch.tensor([0] * num_nodes_per_level[-1])
    )
    subtree_sizes_ref.append(len(tree_indices) + 1)
    assert [
        0
    ] + tree_indices == tree_attn_metadata.tree_indices.tolist(), (
        f"{tree_indices=} {tree_attn_metadata.tree_indices.tolist()=}"
    )

    num_children_per_node_ref.extend([0] * num_paths)
    num_children_per_node_ref_tensor = torch.tensor(num_children_per_node_ref)

    assert all(
        torch.allclose(num_children, num_children_ref)
        for num_children, num_children_ref in zip(
            tree_attn_metadata.num_children_per_node_at_level,
            num_children_per_node_at_level_ref,
        )
    )
    assert all(
        subtree_size == subtree_sizes_ref
        for subtree_size, subtree_sizes_ref in zip(
            tree_attn_metadata.subtree_sizes, subtree_sizes_ref
        )
    )
    torch.testing.assert_close(
        num_children_per_node_ref_tensor, tree_attn_metadata.num_children_per_node
    )
    torch.testing.assert_close(
        tree_attn_metadata.num_nodes_per_level,
        torch.tensor([1] + num_nodes_per_level),
    )
    assert tree_attn_metadata.child_node_indices.shape == (
        len(tree_indices) + 1,
        max(branching),
    )
