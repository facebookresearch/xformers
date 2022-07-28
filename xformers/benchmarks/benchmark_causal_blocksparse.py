# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import os
from typing import Any, Dict

import torch
import triton

from xformers.benchmarks.utils import TestCase, pretty_plot, pretty_print
from xformers.components.attention.attention_mask import AttentionMask
from xformers.components.attention.core import scaled_dot_product_attention

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

SHAPES = [
    (8, 128, 2096),
    (8, 1024, 256),
    (12, 512, 1024),
    (128, 128, 512),
    (8, 2048, 4096),
    (16, 1024, 5120),
    (512, 128, 2560),
]

BLOCK_SIZES = [128]
N_HEADS = [8, 32]


def bench_blocksparse_compare(backward: bool):
    device = torch.device("cuda")
    bw = "+bw" if backward else ""
    use_amp = True
    _use_cuda = True

    for dtype in [torch.float16, torch.float32]:
        datatype = "fp16" if dtype == torch.float16 else "fp32"
        results: Dict[str, Any] = {}
        results_mem: Dict[str, Any] = {}
        for BS in BLOCK_SIZES:
            for heads in N_HEADS:
                for B, M, K in SHAPES:
                    q = torch.randn(
                        (B, heads, M, K // heads),
                        requires_grad=backward,
                        device=device,
                        dtype=dtype,
                    )
                    k = q
                    v = q

                    # Mask with causal flag
                    m_att_mask = AttentionMask.make_causal(
                        M, M, device=device, dtype=dtype
                    )
                    # Custom causal tensor mask
                    m_custom = torch.triu(
                        torch.ones(M, M, device=device, dtype=dtype) * float("-inf"),
                        diagonal=1,
                    )

                    def blocksparse_attention():
                        with torch.cuda.amp.autocast(enabled=use_amp):
                            y = scaled_dot_product_attention(
                                q=q, k=k, v=v, att_mask=m_att_mask, block_size=BS
                            )
                            if backward:
                                torch.norm(y).backward()
                            return y

                    def sdp_attention():
                        with torch.cuda.amp.autocast(enabled=use_amp):
                            y = scaled_dot_product_attention(
                                q=q, k=k, v=v, att_mask=m_custom, block_size=BS
                            )
                            if backward:
                                torch.norm(y).backward()
                            return y

                    for testcase in [
                        TestCase(blocksparse_attention, f"blocksparse - fw{bw}"),
                        TestCase(sdp_attention, f"standard sdp - fw{bw}"),
                    ]:
                        if _use_cuda:
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats()
                            torch.cuda.synchronize()
                        time = triton.testing.do_bench(testcase.function)[0]

                        if _use_cuda:
                            torch.cuda.synchronize()
                            max_memory = torch.cuda.max_memory_allocated() / 2**20
                        else:
                            max_memory = -1

                        key = f"B={B},M={M},K={K},NH={heads}"

                        if key not in results_mem:
                            results_mem[key] = {}
                        results_mem[key][testcase.name] = f"{max_memory:.1f}"

                        if key not in results:
                            results[key] = {}
                        results[key][testcase.name] = f"{time:.2f}"

            pretty_print(
                results,
                title=f"\n --- Type: {datatype} Block Size: {BS} --- ",
                units="runtime in ms",
            )
            pretty_print(
                results_mem,
                title=f"\n --- Type: {datatype} Block Size: {BS} --- ",
                units="peak memory usage in MB",
            )

            pretty_plot(
                results,
                title=f"Causal Blocksparse Runtime FW{bw.upper()} {datatype} Blocksize:{BS}",
                units="runtime in ms",
                dash_key="torch",
                legend_loc="upper left",
            )
            pretty_plot(
                results_mem,
                title=f"Causal Blocksparse Memory FW{bw.upper()} {datatype} Blocksize:{BS}",
                units="peak memory usage in MB",
                dash_key="torch",
                legend_loc="upper left",
            )


for bw in [False, True]:
    bench_blocksparse_compare(bw)
