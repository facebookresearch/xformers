# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import triton
import triton.language as tl

from xformers.triton.configs_matmul import kernel_config

# Credits: Initially inspired by the Triton tutorial on matrix multiplications


@triton.jit
def cdiv(x, y):
    return (x + y - 1) // y


# fmt: off
@triton.autotune(
    configs=kernel_config,
    key=["M", "N", "K"],
)
@triton.jit
def kernel_fma(
    # Pointers to matrices
    D, A, W, C, In,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_db,  stride_dm, stride_dn,
    stride_ab, stride_am, stride_ak,
    stride_wk, stride_wn,
    stride_ib,  stride_im, stride_in,
    # Meta-parameters
    **META,
):
    # fmt: on

    """
    Kernel for computing D = activation(A x W + C)

    - A has shape (B, M, K)
    - W has shape (K, N)
    - C has shape (N,)
    - D has shape (B, M, N)
    - In (optional) has shape (B, M, N)

    'In' optionally saves the A x W + C intermediate for backward computations
    """

    # extract metaparameters
    BLOCK_M, GROUP_M = META["BLOCK_ROW"], META["GROUP_M"]
    BLOCK_N, BLOCK_K = META["BLOCK_COL"], META["BLOCK_K"]

    # programs are grouped together to improve L2 hit rate
    pid, batch_id = tl.program_id(axis=0), tl.program_id(axis=1)

    num_pid_m = cdiv(M, BLOCK_M)  # number of program ids along the M axis
    num_pid_n = cdiv(N, BLOCK_N)  # number of programs ids along the N axis
    num_pid_in_group = GROUP_M * num_pid_n  # number of programs in group
    group_id = pid // num_pid_in_group  # id of the group this program is in
    first_pid_m = group_id * GROUP_M  # row-id of the first program in the group
    GROUP_M = min(
        num_pid_m - first_pid_m, GROUP_M
    )  # if `num_pid_m` isn't divisible by `GROUP_M`, the last group is smaller

    # *within groups*, programs are ordered in a column-major order
    # row-id of the program in the *launch grid*
    pid_m = first_pid_m + (pid % GROUP_M)

    # col-id of the program in the *launch grid*
    pid_n = (pid % num_pid_in_group) // GROUP_M

    # rm (resp. rn) denotes a range of indices
    # for rows (resp. col) of C
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # rk denotes a range of indices for columns
    # (resp. rows) of A (resp. B)
    rk = tl.arange(0, BLOCK_K)

    # the memory addresses of elements in the first block of
    # A and W can be computed using numpy-style broadcasting
    D += rm[:, None] * stride_dm + rn[None, :] * stride_dn + batch_id * stride_db
    A += rm[:, None] * stride_am + rk[None, :] * stride_ak + batch_id * stride_ab
    W += rk[:, None] * stride_wk + rn[None, :] * stride_wn

    # initialize and iteratively update accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(K, 0, -BLOCK_K):
        # block level matrix multiplication
        a = tl.load(A)
        w = tl.load(W)
        acc += tl.dot(a, w).to(tl.float32)

        # increment pointers so that the next blocks of A and B
        # are loaded during the next iteration
        A += BLOCK_K * stride_ak
        W += BLOCK_K * stride_wk

    # optional: add the bias while the data is already in shared memory
    if META["BIAS"]:
        bias = tl.load(C + rn, mask=rn < N, other=0.0)
        bias = bias.to(tl.float32)
        acc += bias[None, :]

    # optional: save the activation inputs
    if META["SAVE_ACT_INPUTS"]:
        In += rm[:, None] * stride_im + rn[None, :] * stride_in + batch_id * stride_ib
        tl.store(In, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))

    # optional: fused activation (while the data is in shared memory)
    if META["ACTIVATION"]:
        acc = META["ACTIVATION"](acc)

    # write back result
    acc = acc.to(D.dtype.element_ty)
    tl.store(D, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


# Activation needs to be a triton kernel
def fused_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    activation=None,
    save_inputs: bool = False
):
    """
    Compute e = activation(x @ weight + bias).
    This wrapper kicks the `kernel_fma` Triton kernel
    """

    if x.ndim == 2:
        # Add the batch dimension
        # This is inelegant and maybe slow, but never really used in the xformers context
        x = x.unsqueeze(0)
        _should_squeeze = True
    else:
        _should_squeeze = False

    assert (
        x.shape[2] == weight.shape[0]
    ), "Incompatible dimensions in between inputs and weight"
    assert bias is None or bias.is_contiguous()
    assert (
        bias is None or bias.shape[0] == weight.shape[1]
    ), "Incompatible dimensions in between weight and bias"

    B, M, K = x.shape
    K, N = weight.shape

    # FIXME: @lefaudeux
    assert (
        K % 32 == 0
    ), "We don't check memory-out-of-bounds with K so K must be divisible by BLOCK_K"

    outputs = torch.empty((B, M, N), device=x.device, dtype=x.dtype)
    act_inputs = torch.empty_like(outputs) if save_inputs else outputs

    # 1D launch kernel where each block gets its own program.
    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_ROW"]) * triton.cdiv(N, META["BLOCK_COL"]),
            B,
        )

    # fmt: off
    kernel_fma[grid](
        # data ptrs
        outputs, x, weight,
        bias if bias is not None else x,  # auto skip bias if not present
        act_inputs,
        # shapes
        M, N, K,
        # strides
        outputs.stride(0), outputs.stride(1), outputs.stride(2),
        x.stride(0), x.stride(1), x.stride(2),
        weight.stride(0), weight.stride(1),
        act_inputs.stride(0), act_inputs.stride(1), act_inputs.stride(2),
        # optional fused activation
        ACTIVATION=activation,
        # optional fused bias
        BIAS=bias is not None,
        # speed optimization: group the programs
        # improve on data reuse in L2 cache
        GROUP_M=8,
        BLOCK_K=32,
        SAVE_ACT_INPUTS=save_inputs
    )
    # fmt: on

    if _should_squeeze:
        outputs.squeeze_()
        if save_inputs:
            act_inputs.squeeze_()

    return (outputs, act_inputs) if save_inputs else (outputs, None)
