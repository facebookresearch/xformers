# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch

from xformers.ops import fmha
from xformers.utils import do_bench_cudagraph


def _merge_attentions_varargs_ref(attn_split, lse_split):
    """
    attn_split: list of [B, M, (G,) H, Kq]
    lse_split: list of [B, (G,) H, M]
    """
    attn_split = torch.stack(attn_split)
    lse_split = torch.stack(lse_split)

    lse_split = lse_split[..., None].moveaxis(4, 2)  # [split_k, B, M, G, H, 1]

    lse_max, _ = torch.max(lse_split, dim=0)  # [B, M, G, H, 1]
    sumexp_normalized = torch.exp(lse_split - lse_max)  # [split_k, B, M, G, H, 1]
    denominator = sumexp_normalized.sum(dim=0)  # [B, M, G, H, 1]
    numerator = (sumexp_normalized * attn_split).sum(dim=0)  # [B, M, G, H, K]

    attn_out = numerator / denominator  # [B, M_ceil, G, H, Kq]
    lse_out = lse_max + torch.log(denominator)
    lse_out = lse_out.squeeze(4).permute(0, 2, 3, 1)  # [B, G, H, M]

    return attn_out, lse_out


def benchmark_merge_attentions_backward(split_k, B, M, G, N_H_L, D_H, dtype):
    """
    Benchmark backward pass for merge_attentions. Assumes "varargs" path,
    i.e. LSE and attention of chunks are provided as two lists of tensors, and not as two stacked tensors.
    """

    bench_stream = torch.cuda.Stream()
    with torch.cuda.stream(bench_stream):

        attn_split = [
            torch.randn(
                [B, M, G, N_H_L, D_H], dtype=dtype, device="cuda", requires_grad=True
            )
            for _ in range(split_k)
        ]
        lse_split = [
            torch.randn(
                [B, G, N_H_L, M], dtype=dtype, device="cuda", requires_grad=True
            )
            for _ in range(split_k)
        ]

        attn_out_ref, lse_out_ref = _merge_attentions_varargs_ref(attn_split, lse_split)
        out_grad = torch.randn_like(attn_out_ref)
        attn_out_ref.backward(out_grad, retain_graph=True)
        t_ms_ref = do_bench_cudagraph(
            lambda: attn_out_ref.backward(out_grad, retain_graph=True)
        )

        for x in attn_split + lse_split:
            x.detach_()
            x.requires_grad_(True)

        attn_out, lse_out = fmha.merge_attentions(attn_split, lse_split)
        attn_out.backward(out_grad, retain_graph=True)
        t_ms = do_bench_cudagraph(
            lambda: attn_out.backward(out_grad, retain_graph=True)
        )

        print(
            f"{split_k=}, {B=}, {M=}, {G=}, {N_H_L=}, {D_H=}, {dtype=}. "
            f"Baseline: {t_ms_ref * 1e3:.2f}us, "
            f"Triton: {t_ms * 1e3:.2f}us, {t_ms_ref/t_ms:.1f}x faster"
        )


def main():
    G = 2
    N_H_L = 8
    D_H = 128
    dtype = torch.float32
    for split_k in [2, 4, 8, 16]:
        for B in [1, 32, 128]:
            for M in [1, 32, 512]:
                benchmark_merge_attentions_backward(split_k, B, M, G, N_H_L, D_H, dtype)


if __name__ == "__main__":
    main()
