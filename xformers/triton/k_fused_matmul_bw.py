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
    'EVEN_N': lambda args: args["N"] % (args['BLOCK_N']) == 0,
})
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 32}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_N": 64}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_N": 128}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_N": 256}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_N": 512}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_N": 1024}, num_stages=3, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def kernel_bw_act(
    # Pointers to tensors
    D_ACT, D_OUT, ACT_INPUTS,
    # Matrix dimensions
    N,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_gom, stride_aim,
    # Meta-parameters
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
    ACTIVATION_GRAD: tl.constexpr,
):
    # fmt: on

    """
    Go over all the activation inputs, compute the corresponding gradient
    """

    # this kernel is relatively simple in terms of scheduling:
    # - per row (pid_m)
    # - each program a given chunk on the col axis,
    # since it's more effective memory and occupancy wise
    pid_m, pid_n = tl.program_id(axis=0), tl.program_id(axis=1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # the memory addresses of elements in the first block of
    # A and W can be computed using numpy-style broadcasting
    act_input_ptrs = ACT_INPUTS + pid_m * stride_aim + rn

    # compute the gradient which is related to this activation
    if EVEN_N:
        act_in = tl.load(act_input_ptrs)
    else:
        act_in = tl.load(act_input_ptrs, mask=rn < N, other=0.0)

    grad_act = ACTIVATION_GRAD(act_in)

    # now read the incoming gradient, the backpropagated one is the multiple of both
    grad_out_ptrs = D_OUT + pid_m * stride_gom + rn
    if EVEN_N:
        grad_out = tl.load(grad_out_ptrs)
    else:
        grad_out = tl.load(grad_out_ptrs, mask=rn < N)

    grad_act *= grad_out

    # write back result
    grad_act_ptrs = D_ACT + pid_m * stride_gom + rn
    tl.store(grad_act_ptrs, grad_act, mask=rn < N)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 16}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 32}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 32}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 64}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 64}, num_stages=4, num_warps=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 128}, num_stages=4, num_warps=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 128}, num_stages=3, num_warps=2),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def kernel_bw_epilogue(
    # Pointers to tensors
    D_WEIGHT, D_BIAS, D_IN, D_ACT, WEIGHT,
    # Tensor dimensions
    M, N, K,
    # Strides which are not == 1
    stride_im, stride_am, stride_wn,
    # Meta-parameters
    BLOCK_M: tl.constexpr, GROUP_M: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    COMPUTE_D_WEIGHT: tl.constexpr,
    COMPUTE_D_BIAS: tl.constexpr,
):
    # fmt: on

    """
    Given the gradient pre-activation, compute the input gradient,
    and optionally the bias and weight gradients

    Shapes:
    d_in    - M x K
    d_act   - M x N
    weight  - N x K
    bias    - N

    The main computation in this kernel is
    d_in = d_act @ weight

    The consolidation is over N
    """

    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)  # number of program ids along the M axis
    num_pid_k = tl.cdiv(K, BLOCK_K)  # number of programs ids along the N axis
    num_pid_in_group = GROUP_M * num_pid_k  # number of programs in group
    group_id = pid // num_pid_in_group  # id of the group this program is in
    first_pid_m = group_id * GROUP_M  # row-id of the first program in the group
    GROUP_M = min(
        num_pid_m - first_pid_m, GROUP_M
    )  # if `num_pid_m` isn't divisible by `GROUP_M`, the last group is smaller

    # *within groups*, programs are ordered in a column-major order
    # row-id /col-id of the program in the *launch grid*
    pid_m = first_pid_m + (pid % GROUP_M)
    pid_k = (pid % num_pid_in_group) // GROUP_M

    # now compute the block that each program will go through
    # rm (resp. rn) denotes a range of indices
    # for rows (resp. col) of C
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    # Compute the input derivative
    # ```
    #   d_in = d_act @ weight
    # ```

    # We fetch a block memory block from both inputs, matmul and accumulate, then repeat
    act_ptrs = D_ACT + rm[:, None] * stride_am
    weight_ptrs = WEIGHT + rk[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    mask_rm = rm < M
    mask_rk = rk < K

    for step_n in range(0, N, BLOCK_N):
        rn = tl.arange(0, BLOCK_N) + step_n
        a = tl.load(act_ptrs + rn[None, :], mask=(mask_rm[:, None] & (rn[None, :] < N)), other=0.0)
        w = tl.load(weight_ptrs + rn[:, None] * stride_wn, mask=((rn[:, None] < N) & mask_rk[None, :]), other=0.0)
        acc += tl.dot(a, w)

        # Optionally compute the bias gradient
        if COMPUTE_D_BIAS:
            # ```
            #   grad_bias = sum_2d_dim_0(grad_act_)
            # ````
            # Benefit here is that we've already loaded the grad_act data, so we can compute the accumulation on the fly
            db = tl.sum(a, axis=0)
            tl.atomic_add(D_BIAS + rn, db, mask=rn < N)

    # write back result
    grad_in_ptrs = D_IN + rm[:, None] * stride_im + rk[None, :]
    tl.store(grad_in_ptrs, acc, mask=mask_rm[:, None] & mask_rk[None, :])

    # Optionally compute the weight gradient
    if COMPUTE_D_WEIGHT:
        # grad_weight = grad_out_.transpose(1, 0) @ inputs_
        # not sure that this can be fused really
        pass


def fused_matmul_backward(
    grad_out: torch.Tensor,
    inputs: torch.Tensor,
    act_in: Optional[torch.Tensor],
    weight: torch.Tensor,
    trainable_weight: bool,
    trainable_bias: bool,
    activation_grad=None,
):
    """
    Compute grad_in = activation^-1(grad_out) @ weight.transpose()

    .. note: The weight buffer is transposed on the fly
    .. note: Activation gradient needs to be a Triton kernel
    """

    # Make sure that we don't have to handle the stride over cols
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()

    grad_out_ = grad_out if grad_out.ndim == 2 else grad_out.flatten(0, 1)
    inputs_ = inputs if inputs.ndim == 2 else inputs.flatten(0, 1)

    assert grad_out_.shape[1] == weight.shape[0], "Incompatible dimensions in between grad_out and weight"

    M, N = grad_out_.shape
    N, K = weight.shape

    # Compute the gradient for the activation
    if activation_grad is not None:
        grad_act = torch.empty_like(grad_out_)

        # Some activations do not require their inputs to
        # know of their grad, the downstream grad is enough
        if act_in is None:
            act_in = grad_out_

        grid = lambda META: (M, triton.cdiv(N, META["BLOCK_N"])) # noqa

        # fmt: off
        kernel_bw_act[grid](
            grad_act, grad_out_, act_in,            # data ptrs
            N,                                      # shapes
            grad_act.stride(0), act_in.stride(0),   # strides
            ACTIVATION_GRAD=activation_grad,        # optional fused activation
        )
        # fmt: on

        # Backpropagation going up, the reference gradient is now
        # just before the activation
        grad_out_ = grad_act

    # Now compute the input and bias gradients
    grid_e = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(K, META["BLOCK_K"]),) # noqa

    grad_weight : Optional[torch.Tensor] = torch.empty_like(weight) if trainable_weight else grad_out
    grad_bias = torch.empty((N,), dtype=inputs_.dtype, device=inputs_.device) if trainable_bias else grad_out
    grad_in = torch.empty_like(inputs_)

    # NOTE: We're using atomic add in the kernel, over a block of size BLOCK_N
    # We should not have more threads than the number of elements in the block
    # else atomic_add is not guarded (multiple commits per element)
    # hence ```num_warps == BLOCK_N // 32```

    # fmt: off
    kernel_bw_epilogue[grid_e](
        grad_weight, grad_bias, grad_in, grad_out_, weight,
        M, N, K,
        grad_in.stride(0), grad_out_.stride(0), weight.stride(0),
        # Meta-parameters
        GROUP_M=8, BLOCK_N=64,
        COMPUTE_D_WEIGHT=False,
        COMPUTE_D_BIAS=trainable_bias,
    )
    # fmt: on

    # Final computation, not fused yet..
    grad_weight = grad_out_.transpose(1, 0) @ inputs_ if trainable_weight else None

    return grad_in.reshape_as(inputs), grad_weight, grad_bias if trainable_bias else None
