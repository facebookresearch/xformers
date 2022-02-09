# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


# Benchmark the blocksparse operations:
# matrix multiply and softmax

# Matmul can be of three types:
# - Dense x Dense (COO) -> Sparse
# - Sparse x Dense -> Dense
# - Dense x Sparse -> Dense

from typing import Any, Dict

import torch
import triton
from triton.ops.blocksparse import matmul as blocksparse_matmul

from xformers.benchmarks.utils import TestCase, pretty_plot, pretty_print
from xformers.components.attention.core import SparseCS, _matmul_with_mask


def bench_matmul(dtype: torch.dtype, shapes):
    results: Dict[str, Any] = {}
    Z, H = 1, 1

    for M, N, K in shapes:

        modes = [(mode, block) for mode in ["sdd", "dsd"] for block in [16, 32, 64]]

        for mode, block in modes:
            # create inputs
            a = torch.randn((Z, H, M, K), dtype=dtype, device="cuda")
            b = torch.randn((Z, H, K, N), dtype=dtype, device="cuda")
            shape = {
                "sdd": (M, N),
                "dsd": (a.shape[2], a.shape[3]),
                "dds": (b.shape[2], b.shape[3]),
            }[mode]

            # Pre-sparsify everything
            _layout = torch.eye(shape[0] // block, shape[1] // block, dtype=torch.long)

            # - blocksparse
            layout = _layout.unsqueeze(0).expand(H, -1, -1)
            a_triton = (
                triton.testing.sparsify_tensor(a, layout, block) if mode == "dsd" else a
            )
            b_triton = (
                triton.testing.sparsify_tensor(b, layout, block) if mode == "dds" else b
            )
            bsmm = blocksparse_matmul(
                layout=layout,
                block=block,
                mode=mode,
                device=torch.device("cuda"),
                trans_a=False,
                trans_b=False,
            )

            # - dense
            ta = triton.testing.mask_tensor(a, layout, block) if mode == "dsd" else a
            tb = triton.testing.mask_tensor(b, layout, block) if mode == "dds" else b

            # - sparse / sputnik
            mask = torch.ones_like(a, dtype=torch.float, device="cuda")
            mask = triton.testing.mask_tensor(mask, layout, block, value=0.0)
            a_cs = a.flatten(start_dim=0, end_dim=1).to(
                torch.float32
            )  # Sputnik kernels only handle fp32
            b_cs = b.flatten(start_dim=0, end_dim=1).to(torch.float32)
            a_cs = a_cs.contiguous()
            b_cs = b_cs.transpose(-2, -1).contiguous()

            if mode == "sdd":
                b_cs = b_cs.transpose(-2, -1)

            # pyre-fixme[16]: TODO(T101400990): Pyre did not recognize the
            # `SparseCS` import.
            sparse_cs_mask = SparseCS(
                mask.flatten(start_dim=0, end_dim=1).contiguous(),
                device=torch.device("cuda"),
            )

            # The raw compute steps
            op_flops = {
                "sdd": 2 * Z * K * float(layout.sum()) * block * block,
                "dsd": 2 * Z * N * float(layout.sum()) * block * block,
                "dds": 2 * Z * M * float(layout.sum()) * block * block,
            }[
                mode
            ] * 1e-12  # TFlops

            def torch_step():
                return torch.matmul(ta, tb)

            def triton_step():
                return bsmm(a_triton, b_triton)

            def sparse_step():
                if mode == "sdd":
                    return _matmul_with_mask(a_cs, b_cs, sparse_cs_mask)
                else:
                    return sparse_cs_mask.spmm(b_cs)

            # Run and measure, report perf in terms of TFlops
            for testcase in [
                TestCase(
                    torch_step,
                    f"pytorch - {mode} - {block}: ",
                ),
                TestCase(
                    sparse_step,
                    f"sparse - {mode} - {block}: ",
                ),
                TestCase(
                    triton_step,
                    f"triton  - {mode} - {block}: ",
                ),
            ]:
                ms = triton.testing.do_bench(lambda: testcase.function())[0]
                key = f"M={M}, N={N}, K={K}"

                if key not in results:
                    results[key] = {}

                num_flops = op_flops / ms * 1e3  # Get to  TFlop per second
                results[key][testcase.name] = f"{num_flops:.1f}"
                print(f"{key} - {testcase.name} - {num_flops:.2f}TFlops")

    pretty_print(
        results,
        title="\n ------------- Type: {} -------------".format(dtype),
        units="TFlops/s",
    )

    pretty_plot(
        results,
        title=f"Sparse/Blocksparse throughput - {dtype}",
        filename=f"blocksparse_{dtype}.png",
        dash_key="pytorch",
        units="TFlops/s",
    )


shapes = [(k, k, k) for k in [128, 512, 1024, 2048, 4096]]
bench_matmul(torch.float16, shapes)
bench_matmul(torch.float32, shapes)
