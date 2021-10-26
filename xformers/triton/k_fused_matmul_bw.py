# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import Optional

import torch
import triton
import triton.language as tl


# fmt: off
@triton.heuristics({
    'EVEN_K': lambda *args, **meta: args[7] % (meta['BLOCK_K']) == 0,
})
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_ROW": 64, "BLOCK_COL": 32}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_ROW": 32, "BLOCK_COL": 64}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_ROW": 32, "BLOCK_COL": 256}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_ROW": 128, "BLOCK_COL": 256}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_ROW": 256, "BLOCK_COL": 128}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_ROW": 64, "BLOCK_COL": 256}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_ROW": 256, "BLOCK_COL": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_ROW": 128, "BLOCK_COL": 128}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_ROW": 128, "BLOCK_COL": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_ROW": 64, "BLOCK_COL": 128}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_ROW": 128, "BLOCK_COL": 32}, num_stages=4, num_warps=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def kernel_bw(
    # Pointers to matrices
    OUT, GRAD_OUT, INPUT, WEIGHT, BIAS,
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

    # rm (resp. rn) denotes a range of indices
    # for rows (resp. col) of C
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # the memory addresses of elements in the first block of
    # A and W can be computed using numpy-style broadcasting
    input_ptrs = INPUT + rm[:, None] * stride_im + rk[None, :]
    weight_ptrs = WEIGHT + rk[:, None] * stride_wk + rn[None, :] * stride_wn

    # initialize and iteratively update accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if META["BIAS"]:
        bias = tl.load(BIAS + rn, mask=rn < N, other=0.0).to(tl.float32)
        acc += bias[None, :]

    # block level matrix multiplication
    for _ in range(K, 0, -BLOCK_K):
        if META['EVEN_K']:
            a = tl.load(input_ptrs)
            w = tl.load(weight_ptrs)
        else:
            a = tl.load(input_ptrs, mask=(rk[None, :] < K), other=0.0)
            w = tl.load(weight_ptrs, mask=(rk[:, None] < K), other=0.0)

        acc += tl.dot(a, w).to(tl.float32)

        input_ptrs += BLOCK_K
        weight_ptrs += BLOCK_K * stride_wk

    # optional: fused activation (while the data is in shared memory)
    if META["ACTIVATION"]:
        acc = META["ACTIVATION"](acc)

    # write back result (multiply-accumulate)
    mask_mn = (rm[:, None] < M) & (rn[None, :] < N)
    grad_out_ptrs = GRAD_OUT + rm[:, None] * stride_om + rn[None, :]
    grad_out = tl.load(grad_out_ptrs, mask=mask_mn)
    grad_out *= acc
    out_ptrs = OUT + rm[:, None] * stride_om + rn[None, :]
    tl.store(out_ptrs, grad_out, mask=mask_mn)


# Activation needs to be a triton kernel
def fused_matmul_backward(
    grad_out: torch.Tensor,
    inputs: torch.Tensor,
    bias : Optional[torch.Tensor],
    weight: torch.Tensor,
    trainable_weight: bool,
    trainable_bias: bool,
    activation_grad=None,
):
    """
    Compute grad_in = activation^-1(grad_out) @ weight.transpose()

    .. note: The weight buffer is transposed on the fly
    """

    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()

    grad_out_ = grad_out if grad_out.ndim == 2 else grad_out.flatten(0, 1)
    inputs_ = inputs if inputs.ndim == 2 else inputs.flatten(0, 1)

    assert grad_out_.shape[1] == weight.shape[0], "Incompatible dimensions in between grad_out and weight"

    M, N = grad_out_.shape
    N, _ = weight.shape
    _, K = inputs_.shape

    # Compute the gradient for the activation
    if activation_grad is not None:
        # The strategy is as follows:
        # instead of saving intermediates during the forward pass, we recompute everything that is needed
        # from scratch here.
        # This means that the kernel which was being used in the FW pass can be reused as is,
        # but the activation function becomes the activation gradient instead
        grad_act = torch.empty_like(grad_out_)

        def grid(META):
            return (
                triton.cdiv(M, META["BLOCK_ROW"]) * triton.cdiv(N, META["BLOCK_COL"]),
            )

        # fmt: off
        kernel_bw[grid](
            # data ptrs
            grad_act, grad_out_, inputs_, weight,
            bias if bias is not None else inputs_,  # auto skip bias if not needed
            # shapes
            M, N, K,
            # strides
            grad_act.stride(0), inputs_.stride(0),
            weight.stride(0), weight.stride(1),
            # optional fused activation
            ACTIVATION=activation_grad,
            # optional fused bias
            BIAS=bias is not None,
            # speed optimization: group the programs
            # improve on data reuse in L2 cache
            GROUP_ROW=8,
            BLOCK_K=32,
        )

        # Backpropagation going up, the reference gradient is now
        # just before the activation
        grad_out_ = grad_act

    grad_in = triton.ops.matmul(grad_out_, weight)
    grad_weight = triton.ops.matmul(grad_out_.transpose(1, 0), inputs_) if trainable_weight else None
    grad_bias = torch.sum(grad_out_, 0) if trainable_bias else None

    return grad_in.reshape_as(inputs), grad_weight, grad_bias
