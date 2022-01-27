# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import triton
import triton.language as tl

# CREDITS: Initially inspired by the Triton tutorial on matrix multiplications


# fmt: off
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_ROW": 16, "BLOCK_COL": 16}, num_stages=5, num_warps=1),
        triton.Config({"BLOCK_ROW": 32, "BLOCK_COL": 32}, num_stages=5, num_warps=1),
        triton.Config({"BLOCK_ROW": 64, "BLOCK_COL": 32}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_ROW": 32, "BLOCK_COL": 64}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_ROW": 128, "BLOCK_COL": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_ROW": 64, "BLOCK_COL": 128}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_ROW": 128, "BLOCK_COL": 128}, num_stages=4, num_warps=4),
        # triton.Config({"BLOCK_ROW": 32, "BLOCK_COL": 256}, num_stages=3, num_warps=4),
        # triton.Config({"BLOCK_ROW": 256, "BLOCK_COL": 32}, num_stages=3, num_warps=4),
        # triton.Config({"BLOCK_ROW": 64, "BLOCK_COL": 256}, num_stages=3, num_warps=8),
        # triton.Config({"BLOCK_ROW": 256, "BLOCK_COL": 64}, num_stages=3, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def kernel_fma(
    # Pointers to matrices
    OUT, ACT_INPUTS, INPUT, WEIGHT, BIAS,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_om, stride_im,
    stride_wn, stride_wk,
    # Meta-parameters
    **META,
):
    # fmt: on

    """
    Kernel for computing Out = activation(A x W + C)

    - Input has shape (M, K)
    - Weight has shape (K, N)
    - Bias has shape (N,)
    - Output has shape (M, N)
    - ActInputs (optional) has shape (M, N)

    'ActInputs' optionally saves the A x W + C intermediate for backward computations

    This kernel will consolidate over K
    """

    # extract metaparameters
    BLOCK_M, GROUP_M = META["BLOCK_ROW"], META["GROUP_ROW"]
    BLOCK_N, BLOCK_K = META["BLOCK_COL"], META["BLOCK_K"]

    # programs are grouped together to improve L2 hit rate
    # the logic is that we'll consolidate over K. If the programs were not grouped,
    # then multiple cols/rows in the result would end up pulling in the same row and lines
    # from the inputs. By grouping the computation we ensure some data reuse, which the hardware
    # covers via the L2 cache
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)  # number of program ids along the M axis
    num_pid_n = tl.cdiv(N, BLOCK_N)  # number of programs ids along the N axis
    num_pid_in_group = GROUP_M * num_pid_n  # number of programs in group
    group_id = pid // num_pid_in_group  # id of the group this program is in
    first_pid_m = group_id * GROUP_M  # row-id of the first program in the group
    GROUP_M = min(
        num_pid_m - first_pid_m, GROUP_M
    )  # if `num_pid_m` isn't divisible by `GROUP_M`, the last group is smaller

    # *within groups*, programs are ordered in a column-major order
    # row-id /col-id of the program in the *launch grid*
    pid_m = first_pid_m + (pid % GROUP_M)
    pid_n = (pid % num_pid_in_group) // GROUP_M

    # now compute the block that each program will go through
    # rm (resp. rn) denotes a range of indices
    # for rows (resp. col) of C
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # the memory addresses of elements can follow numpy broadcasting
    input_ptrs = INPUT + rm[:, None] * stride_im + rk[None, :]
    weight_ptrs = WEIGHT + rk[:, None] * stride_wk + rn[None, :] * stride_wn

    # initialize and iteratively update accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if META["BIAS"]:
        bias = tl.load(BIAS + rn, mask=rn < N, other=0.0).to(tl.float32)
        acc += bias[None, :]

    # block level matrix multiplication.
    # We fetch a block memory block from both inputs, matmul and accumulate, then repeat
    for _ in range(K, 0, -BLOCK_K):
        a = tl.load(input_ptrs, mask=((rk[None, :] < K) & (rm[:, None] < M)), other=0.0)
        w = tl.load(weight_ptrs, mask=((rk[:, None] < K) & (rn[None, :] < N)), other=0.0)

        acc += tl.dot(a, w).to(tl.float32)

        input_ptrs += BLOCK_K
        weight_ptrs += BLOCK_K * stride_wk

    # optional: save the activation inputs
    if META["SAVE_ACT_INPUTS"]:
        act_in_ptrs = ACT_INPUTS + rm[:, None] * stride_om + rn[None, :]
        tl.store(act_in_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))

    # optional: fused activation (while the data is in shared memory)
    if META["ACTIVATION"]:
        acc = META["ACTIVATION"](acc)

    # write back result
    out_ptrs = OUT + rm[:, None] * stride_om + rn[None, :]
    tl.store(out_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


# Activation needs to be a triton kernel
def fused_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    activation=None,
    save_act_inputs: bool = False
):
    """
    Compute e = activation(x @ weight + bias).
    This wrapper kicks the `kernel_fma` Triton kernel
    """

    if not x.is_contiguous():
        x = x.contiguous()

    x_ = x if x.ndim == 2 else x.flatten(0, 1)

    assert (
        x_.shape[1] == weight.shape[1]
    ), f"Incompatible dimensions in between inputs and weight, {x_.shape} - {weight.shape}"
    assert bias is None or bias.is_contiguous()
    assert (
        bias is None or bias.shape[0] == weight.shape[0]
    ), "Incompatible dimensions in between weight and bias"

    M, K = x_.shape
    N, K = weight.shape

    outputs = torch.empty((M, N), device=x.device, dtype=x.dtype)
    act_inputs = torch.empty_like(outputs) if save_act_inputs else x  # will not be used in that case

    # 1D launch kernel where each block gets its own program.
    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_ROW"]) * triton.cdiv(N, META["BLOCK_COL"]),
        )

    # fmt: off
    kernel_fma[grid](
        # data ptrs
        outputs, act_inputs, x_, weight,
        bias if bias is not None else x,  # auto skip bias if not present
        # shapes
        M, N, K,
        # strides
        outputs.stride(0), x_.stride(0),
        weight.stride(0), weight.stride(1),
        # optional fused activation
        ACTIVATION=activation,
        # optional fused bias
        BIAS=bias is not None,
        # speed optimization: group the programs
        # improve on data reuse in L2 cache
        GROUP_ROW=8,
        BLOCK_K=32,
        SAVE_ACT_INPUTS=save_act_inputs
    )
    # fmt: on

    outputs = outputs if x.ndim == 2 else outputs.reshape(x.shape[0], -1, N)

    return outputs, act_inputs if save_act_inputs else None
