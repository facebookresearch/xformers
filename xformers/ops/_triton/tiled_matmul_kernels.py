# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import itertools
from typing import List, Tuple

import torch
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time


def init_to_zero(*names):
    def result(nargs):
        for name in names:
            nargs[name].zero_()

    return result


def gen_config(
    block_m: int,
    block_n: int,
    block_k: int,
    stages: int,
    warps: int,
    split_k: int = 1,
    group_m: int = 8,
) -> triton.Config:
    """A more compact way to define a triton.Config, so it fits on one line"""

    return triton.Config(
        {
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "SPLIT_K": split_k,
            "GROUP_M": group_m,
        },
        num_stages=stages,
        num_warps=warps,
        pre_hook=init_to_zero(*[f"C{i+1}{j+1}" for i in range(3) for j in range(3)])
        if split_k > 1
        else init_to_zero(),
    )


BASIC_MATMUL_CONFIGS = [
    gen_config(block_m=128, block_n=256, block_k=32, stages=3, warps=8),
    gen_config(block_m=256, block_n=128, block_k=32, stages=3, warps=8),
    gen_config(block_m=256, block_n=64, block_k=32, stages=4, warps=4),
    gen_config(block_m=64, block_n=256, block_k=32, stages=4, warps=4),
    gen_config(block_m=128, block_n=128, block_k=32, stages=4, warps=4),
    gen_config(block_m=128, block_n=64, block_k=32, stages=4, warps=4),
    gen_config(block_m=64, block_n=128, block_k=32, stages=4, warps=4),
    gen_config(block_m=128, block_n=32, block_k=32, stages=4, warps=4),
    gen_config(block_m=64, block_n=32, block_k=32, stages=5, warps=2),
]


INT8_MATMUL_CONFIGS = [
    gen_config(block_m=128, block_n=256, block_k=128, stages=3, warps=8),
    gen_config(block_m=256, block_n=128, block_k=128, stages=3, warps=8),
    gen_config(block_m=256, block_n=64, block_k=128, stages=4, warps=4),
    gen_config(block_m=64, block_n=256, block_k=128, stages=4, warps=4),
    gen_config(block_m=128, block_n=128, block_k=128, stages=4, warps=4),
    gen_config(block_m=128, block_n=64, block_k=64, stages=4, warps=4),
    gen_config(block_m=64, block_n=128, block_k=64, stages=4, warps=4),
    gen_config(block_m=128, block_n=32, block_k=64, stages=4, warps=4),
    gen_config(block_m=64, block_n=32, block_k=64, stages=5, warps=2),
]


IO_BOUND_MATMUL_CONFIGS_STAGES = [2, 3, 4, 5, 6]
IO_BOUND_MATMUL_CONFIGS_BLOCK_M = [16, 32]
IO_BOUND_MATMUL_CONFIGS_BLOCK_K = [32, 64]
IO_BOUND_MATMUL_CONFIGS_BLOCK_N = [32, 64, 128, 256]
IO_BOUND_MATMUL_CONFIGS_SPLIT_K = [1, 2, 4, 8, 16]


IO_BOUND_MATMUL_CONFIGS = [
    gen_config(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        stages=stages,
        warps=2 if block_n <= 64 else 4,
        split_k=split_k,
    )
    for stages, block_m, block_k, block_n, split_k in itertools.product(
        IO_BOUND_MATMUL_CONFIGS_STAGES,
        IO_BOUND_MATMUL_CONFIGS_BLOCK_M,
        IO_BOUND_MATMUL_CONFIGS_BLOCK_K,
        IO_BOUND_MATMUL_CONFIGS_BLOCK_N,
        IO_BOUND_MATMUL_CONFIGS_SPLIT_K,
    )
]


TRITON_CONFIGS = BASIC_MATMUL_CONFIGS + INT8_MATMUL_CONFIGS + IO_BOUND_MATMUL_CONFIGS


def our_estimate_matmul_time(
    A11, B11, C11, M1, M2, M3, N1, N2, N3, K1, K2, K3, **kwargs
):
    """Call into Triton's upstream cost model, with the right args

    The upstream function expects arguments to have certain names. Since we
    renamed a few of them in our implementation, we rename them back.

    At the time of writing (July 2023) the arguments that Triton expects are:
    M, N, K, A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages.

    """
    return estimate_matmul_time(
        M=M1 + M2 + M3, N=N1 + N2 + N3, K=K1 + K2 + K3, A=A11, B=B11, C=C11, **kwargs
    )


def our_early_config_prune(config, named_args):
    new_named_args = named_args.copy()
    new_named_args["M"] = named_args["M1"] + named_args["M2"] + named_args["M3"]
    new_named_args["N"] = named_args["N1"] + named_args["N2"] + named_args["N3"]
    new_named_args["K"] = named_args["K1"] + named_args["K2"] + named_args["K3"]
    new_named_args["A"] = named_args["A11"]
    new_named_args["B"] = named_args["B11"]
    new_named_args["C"] = named_args["C11"]
    return early_config_prune(config, new_named_args)


@triton.autotune(
    configs=TRITON_CONFIGS,
    key=["M1", "M2", "M3", "N1", "N2", "N3", "K1", "K2", "K3"],
    prune_configs_by={
        "early_config_prune": our_early_config_prune,
        "perf_model": our_estimate_matmul_time,
        "top_k": 10,
    },
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: all(
            k % (args["BLOCK_K"] * args["SPLIT_K"]) == 0
            for k in [args["K1"], args["K2"], args["K3"]]
        ),
    }
)
@triton.jit()
def _xformers_tiled_matmul_kernel(
    A11,
    A12,
    A13,
    A21,
    A22,
    A23,
    A31,
    A32,
    A33,
    B11,
    B12,
    B13,
    B21,
    B22,
    B23,
    B31,
    B32,
    B33,
    C11,
    C12,
    C13,
    C21,
    C22,
    C23,
    C31,
    C32,
    C33,
    M1,
    M2,
    M3,
    N1,
    N2,
    N3,
    K1,
    K2,
    K3,
    stride_am1,
    stride_am2,
    stride_am3,
    stride_ak1,
    stride_ak2,
    stride_ak3,
    stride_bk1,
    stride_bk2,
    stride_bk3,
    stride_bn1,
    stride_bn2,
    stride_bn3,
    stride_cm1,
    stride_cm2,
    stride_cm3,
    stride_cn1,
    stride_cn2,
    stride_cn3,
    BLOCK_M: tl.constexpr,  # DO NOT CHANGE NAME: MUST MATCH PERF MODEL
    BLOCK_N: tl.constexpr,  # DO NOT CHANGE NAME: MUST MATCH PERF MODEL
    BLOCK_K: tl.constexpr,  # DO NOT CHANGE NAME: MUST MATCH PERF MODEL
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,  # DO NOT CHANGE NAME: MUST MATCH PERF MODEL
    EVEN_K: tl.constexpr,
    ACC_TYPE: tl.constexpr,
):
    # matrix multiplication
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    grid_m1 = tl.cdiv(M1, BLOCK_M)
    grid_m2 = tl.cdiv(M2, BLOCK_M)
    grid_m3 = tl.cdiv(M3, BLOCK_M)
    grid_n1 = tl.cdiv(N1, BLOCK_N)
    grid_n2 = tl.cdiv(N2, BLOCK_N)
    grid_n3 = tl.cdiv(N3, BLOCK_N)
    grid_m = grid_m1 + grid_m2 + grid_m3
    grid_n = grid_n1 + grid_n2 + grid_n3

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    # We use tl.where to circumvent a regression in alignment auto-detection:
    # https://github.com/openai/triton/issues/1784

    A1 = tl.where(pid_m < grid_m1, A11, tl.where(pid_m < grid_m1 + grid_m2, A21, A31))
    A2 = tl.where(pid_m < grid_m1, A12, tl.where(pid_m < grid_m1 + grid_m2, A22, A32))
    A3 = tl.where(pid_m < grid_m1, A13, tl.where(pid_m < grid_m1 + grid_m2, A23, A33))
    B1 = tl.where(pid_n < grid_n1, B11, tl.where(pid_n < grid_n1 + grid_n2, B12, B13))
    B2 = tl.where(pid_n < grid_n1, B21, tl.where(pid_n < grid_n1 + grid_n2, B22, B23))
    B3 = tl.where(pid_n < grid_n1, B31, tl.where(pid_n < grid_n1 + grid_n2, B32, B33))
    C = tl.where(
        pid_m < grid_m1,
        tl.where(pid_n < grid_n1, C11, tl.where(pid_n < grid_n1 + grid_n2, C12, C13)),
        tl.where(
            pid_m < grid_m1 + grid_m2,
            tl.where(
                pid_n < grid_n1, C21, tl.where(pid_n < grid_n1 + grid_n2, C22, C23)
            ),
            tl.where(
                pid_n < grid_n1, C31, tl.where(pid_n < grid_n1 + grid_n2, C32, C33)
            ),
        ),
    )
    M = tl.where(pid_m < grid_m1, M1, tl.where(pid_m < grid_m1 + grid_m2, M2, M3))
    N = tl.where(pid_n < grid_n1, N1, tl.where(pid_n < grid_n1 + grid_n2, N2, N3))
    stride_ak = tl.where(
        pid_m < grid_m1,
        stride_ak1,
        tl.where(pid_m < grid_m1 + grid_m2, stride_ak2, stride_ak3),
    )
    stride_bk = tl.where(
        pid_n < grid_n1,
        stride_bk1,
        tl.where(pid_n < grid_n1 + grid_n2, stride_bk2, stride_bk3),
    )
    stride_cn = tl.where(
        pid_m < grid_m1,
        stride_cn1,
        tl.where(pid_m < grid_m1 + grid_m2, stride_cn2, stride_cn3),
    )
    stride_cm = tl.where(
        pid_n < grid_n1,
        stride_cm1,
        tl.where(pid_n < grid_n1 + grid_n2, stride_cm2, stride_cm3),
    )
    pid_m = tl.where(
        pid_m < grid_m1,
        pid_m,
        tl.where(pid_m < grid_m1 + grid_m2, pid_m - grid_m1, pid_m - grid_m1 - grid_m2),
    )
    pid_n = tl.where(
        pid_n < grid_n1,
        pid_n,
        tl.where(pid_n < grid_n1 + grid_n2, pid_n - grid_n1, pid_n - grid_n1 - grid_n2),
    )

    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    # pointers
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    grid_k1 = tl.cdiv(K1, BLOCK_K)
    grid_k2 = tl.cdiv(K2, BLOCK_K)
    grid_k3 = tl.cdiv(K3, BLOCK_K)
    for tile in range(pid_k, grid_k1 + grid_k2 + grid_k3, SPLIT_K):
        A = tl.where(tile < grid_k1, A1, tl.where(tile < grid_k1 + grid_k2, A2, A3))
        B = tl.where(tile < grid_k1, B1, tl.where(tile < grid_k1 + grid_k2, B2, B3))
        K = tl.where(tile < grid_k1, K1, tl.where(tile < grid_k1 + grid_k2, K2, K3))
        stride_am = tl.where(
            tile < grid_k1,
            stride_am1,
            tl.where(tile < grid_k1 + grid_k2, stride_am2, stride_am3),
        )
        stride_bn = tl.where(
            tile < grid_k1,
            stride_bn1,
            tl.where(tile < grid_k1 + grid_k2, stride_bn2, stride_bn3),
        )
        my_tile = tl.where(
            tile < grid_k1,
            tile,
            tl.where(
                tile < grid_k1 + grid_k2, tile - grid_k1, tile - grid_k1 - grid_k2
            ),
        )
        rk = my_tile * BLOCK_K + tl.arange(0, BLOCK_K)
        Ain = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
        Bin = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
        if EVEN_K:
            a = tl.load(Ain)
            b = tl.load(Bin)
        else:
            a = tl.load(Ain, mask=rk[None, :] < K, other=0.0)
            b = tl.load(Bin, mask=rk[:, None] < K, other=0.0)
        acc += tl.dot(a, b, allow_tf32=False)
    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


def _check_row_or_column(row_or_col_type, row_or_col_idx, tensor_name, dim_name, vals):
    assert len(vals) > 0
    for pos, val in enumerate(vals[1:]):
        assert val == vals[0], (
            f"the tensors on {row_or_col_type} {row_or_col_idx} of the {tensor_name} "
            f"must all have the same stride along the {dim_name} dimension, got "
            f"{vals[0]} at position 0 and {val} at position {pos + 1}"
        )
    return vals[0]


def _get_strides(
    ts: List[List[torch.Tensor]], tensor_name, dim_0_name, dim_1_name
) -> Tuple[List[int], List[int]]:
    strides_0 = [
        _check_row_or_column(
            "column", idx, tensor_name, dim_0_name, [y.stride(0) for y in x]
        )
        for idx, x in enumerate(zip(*ts))
    ]
    strides_1 = [
        _check_row_or_column(
            "row", idx, tensor_name, dim_1_name, [y.stride(1) for y in x]
        )
        for idx, x in enumerate(ts)
    ]
    assert all(s == 1 for s in strides_0) or all(s == 1 for s in strides_1)
    while len(strides_0) < 3:
        strides_0.append(1 if strides_0[0] == 1 else 0)
    while len(strides_1) < 3:
        strides_1.append(1 if strides_1[0] == 1 else 0)
    return strides_0, strides_1


def _launch_triton_matmul(
    a: List[List[torch.Tensor]],
    b: List[List[torch.Tensor]],
    c: List[List[torch.Tensor]],
    ms: List[int],
    ns: List[int],
    ks: List[int],
) -> None:
    strides_am, strides_ak = _get_strides(a, "first operand", "m", "k")
    strides_bk, strides_bn = _get_strides(b, "second operand", "k", "n")
    strides_cm, strides_cn = _get_strides(c, "output", "m", "n")

    # accumulator types
    ACC_TYPE = (
        tl.float32
        if c[0][0].dtype in [torch.float16, torch.bfloat16, torch.float32]
        else tl.int32
    )

    # launch kernel
    def grid(META):
        return (
            sum(triton.cdiv(m, META["BLOCK_M"]) for m in ms)
            * sum(triton.cdiv(n, META["BLOCK_N"]) for n in ns),
            META["SPLIT_K"],
        )

    _xformers_tiled_matmul_kernel[grid](
        *[
            a[min(i, len(a) - 1)][min(j, len(a[0]) - 1)]
            for i in range(3)
            for j in range(3)
        ],
        *[
            b[min(i, len(b) - 1)][min(j, len(b[0]) - 1)]
            for i in range(3)
            for j in range(3)
        ],
        *[
            c[min(i, len(c) - 1)][min(j, len(c[0]) - 1)]
            for i in range(3)
            for j in range(3)
        ],
        *[ms[i] if len(ms) > i else 0 for i in range(3)],
        *[ns[i] if len(ns) > i else 0 for i in range(3)],
        *[ks[i] if len(ks) > i else 0 for i in range(3)],
        *strides_am,
        *strides_ak,
        *strides_bk,
        *strides_bn,
        *strides_cm,
        *strides_cn,
        ACC_TYPE=ACC_TYPE,
    )
