# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import triton
import triton.language as tl

from xformers.triton.configs_matmul import kernel_config


# fmt: off
@triton.autotune(
    configs=kernel_config,
    key=["M", "N", "K"],
)
@triton.jit
def kernel_fma_grad_in(
    # Pointers to all the tensors
    GRAD_IN, GRAD_ACT, ACT_IN, GRAD_OUT, W,
    # Tensor dimensions
    M, N, K,
    # strides for all the gradients
    stride_gib, stride_gim, stride_gik,
    stride_gab, stride_gam, stride_gan,
    stride_gob, stride_gom, stride_gon,
    # strides for the extra data
    stride_aib, stride_aim, stride_ain,
    stride_wn, stride_wk,
    # Meta-parameters
    **META,
):
    # fmt: on
    """
    Kernel for computing `grad_out = grad_in * activation_grad(inputs) @ W^T`
    - grad_out has shape (B, M, N)
    - W has shape (K, N)
    - grad_in has shape (B, M, K)
    - X has shape (B, M, K)
    """
    # extract metaparameters
    BLOCK_M, GROUP_M = META["BLOCK_ROW"], META["GROUP_M"]
    BLOCK_N, BLOCK_K = META["BLOCK_N"], META["BLOCK_COL"]

    # programs are grouped together to improve L2 hit rate
    pid, batch_id = tl.program_id(axis=0), tl.program_id(axis=1)

    num_pid_m = tl.cdiv(M, BLOCK_M)  # number of program ids along the M axis
    num_pid_k = tl.cdiv(K, BLOCK_K)  # number of programs ids along the N axis
    num_pid_in_group = GROUP_M * num_pid_k  # number of programs in group
    group_id = pid // num_pid_in_group  # id of the group this program is in
    first_pid_m = group_id * GROUP_M  # row-id of the first program in the group
    GROUP_M = min(
        num_pid_m - first_pid_m, GROUP_M
    )  # if `num_pid_m` isn't divisible by `GROUP_M`, the last group is smaller

    # *within groups*, programs are ordered in a column-major order
    pid_m = first_pid_m + (pid % GROUP_M)  # row-id of the program in the *launch grid*
    pid_k = (
        pid % num_pid_in_group
    ) // GROUP_M  # col-id of the program in the *launch grid*

    # memory ranges
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    rn = tl.arange(0, BLOCK_N)

    # memory blocks can be computed using numpy-style broadcasting
    GRAD_OUT += rm[:, None] * stride_gom + rn[None, :] * stride_gon + batch_id * stride_gob
    GRAD_ACT += rm[:, None] * stride_gam + rn[None, :] * stride_gan + batch_id * stride_gab
    ACT_IN += rm[:, None] * stride_aim + rn[None, :] * stride_ain + batch_id * stride_aib
    GRAD_IN += rm[:, None] * stride_gim + rk[None, :] * stride_gik + batch_id * stride_gib
    W += rn[:, None] * stride_wn + rk[None, :] * stride_wk

    # initialize and iteratively update accumulator
    grad_in = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    act_grad_fn = META["ACTIVATION_GRAD"]

    for _ in range(N, 0, -BLOCK_N):
        grad_out = tl.load(GRAD_OUT)  # BLOCK_M x BLOCK_N
        w = tl.load(W)  # BLOCK_N x BLOCK_K

        # optional fused activation gradient (while the data is in shared memory)
        if META["ACTIVATION_GRAD"]:
            if META["ACTIVATION_GRAD_REQ_INPUTS"]:
                # This activation requires its inputs
                act_input = tl.load(ACT_IN)
                grad_act = act_grad_fn(act_input)
                ACT_IN += BLOCK_N * stride_ain
            else:
                # Save some time, we can reuse the outputs to know about the grad
                grad_act = act_grad_fn(grad_out)

            grad_out *= grad_act.to(grad_out.dtype)

        # store grad_act as an intermediate, will be used for grad/weight and grad/bias
        if META["SAVE_ACT_GRAD"]:
            tl.store(GRAD_ACT, grad_out)

        # gradient #1: input with respect to outputs
        # grad_in is grad_out scaled by the (transposed) weight
        grad_in += tl.dot(grad_out, w)

        # increment pointers so that the next blocks of A and B are loaded during the next iteration
        GRAD_OUT += BLOCK_N * stride_gon
        GRAD_ACT += BLOCK_N * stride_gan
        W += BLOCK_N * stride_wn

    # write back result
    tl.store(GRAD_IN, grad_in, mask=(rm[:, None] < M) & (rk[None, :] < K))  # type promotion or downgrade is automatic


# Activation needs to be a triton kernel
def fused_matmul_backward(
    grad_out: torch.Tensor,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    trainable_weight: bool,
    trainable_bias: bool,
    activation_inputs: Optional[torch.Tensor],
    activation_grad=None,
    activation_grad_req_inputs: bool = False,
):
    """
    Compute grad_in = activation^-1(grad_out) @ weight.transpose()

    .. note: The weight buffer is transposed on the fly
    """

    if grad_out.ndim == 2:
        # Add the batch dimension
        # This is inelegant and maybe slow, but never really used in the xformers context
        grad_out = grad_out.unsqueeze(0)
        _should_squeeze = True
    else:
        _should_squeeze = False

    assert (
        grad_out.shape[2] == weight.shape[0]
    ), "Incompatible dimensions in between grad_out and weight"

    B, M, N = grad_out.shape
    N, K = weight.shape

    grad_in = torch.empty((B, M, K), device=grad_out.device, dtype=grad_out.dtype)
    grad_act = torch.empty_like(grad_out)

    # Compute the gradient for the inputs
    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_ROW"]) * triton.cdiv(K, META["BLOCK_COL"]),
            B,
        )

    if activation_inputs is None:
        # place holder, this will not be used really
        activation_inputs = grad_out

    # fmt: off
    kernel_fma_grad_in[grid](
        # data ptrs
        grad_in, grad_act, activation_inputs, grad_out, weight,
        # shapes
        M, N, K,
        # strides
        grad_in.stride(0), grad_in.stride(1), grad_in.stride(2),
        grad_act.stride(0), grad_act.stride(1), grad_act.stride(2),
        grad_out.stride(0), grad_out.stride(1), grad_out.stride(2),
        activation_inputs.stride(0), activation_inputs.stride(1), activation_inputs.stride(2),
        weight.stride(0), weight.stride(1),
        # optional fused activation
        ACTIVATION_GRAD=activation_grad,
        # data reuse optimization
        GROUP_M=16,
        BLOCK_N=32,
        ACTIVATION_GRAD_REQ_INPUTS=activation_grad_req_inputs,
        SAVE_ACT_GRAD=trainable_weight or trainable_bias
    )
    # fmt: on

    grad_bias = torch.sum(grad_act, dim=[0, 1]) if trainable_bias else None

    # Reuse Triton optimized matmul
    grad_weight = None
    if trainable_weight:
        grad_act_ = torch.reshape(grad_act, (grad_act.shape[0]*grad_act.shape[1], grad_act.shape[2])).transpose(1, 0)
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(0)
        inputs_ = torch.reshape(inputs, (inputs.shape[0]*inputs.shape[1], inputs.shape[2]))
        grad_weight = triton.ops.matmul(grad_act_, inputs_)

    if _should_squeeze:
        grad_in = grad_in.squeeze_()

    del grad_act

    return grad_in, grad_weight, grad_bias
