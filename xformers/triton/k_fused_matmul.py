# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import triton
import triton.language as tl

# CREDITS: Initially inspired by the Triton tutorial on matrix multiplications

_fuse_bias_gradient_computation = (
    False  # depending on the matrix sizes, this is not always beneficial
)


# fmt: off
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64 , 'BLOCK_N': 64}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 64 , 'BLOCK_N': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 64}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 64 , 'BLOCK_N': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64 , 'BLOCK_N': 256}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128 , 'BLOCK_N': 256}, num_stages=3, num_warps=8)
    ],
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
    stride_dm, stride_am,
    stride_wn, stride_wk,
    stride_im,
    # Meta-parameters
    **META,
):
    # fmt: on

    """
    Kernel for computing D = activation(A x W + C)

    - A has shape (M, K)
    - W has shape (K, N)
    - C has shape (N,)
    - D has shape (M, N)
    - In (optional) has shape (M, N)

    'In' optionally saves the A x W + C intermediate for backward computations
    """

    # extract metaparameters
    BLOCK_M, GROUP_M = META["BLOCK_M"], META["GROUP_M"]
    BLOCK_N, BLOCK_K = META["BLOCK_N"], META["BLOCK_K"]

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
    D += rm[:, None] * stride_dm + rn[None, :]
    A += rm[:, None] * stride_am + rk[None, :]
    W += rn[None, :] * stride_wn + rk[:, None] * stride_wk

    # initialize and iteratively update accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if META["BIAS"]:
        bias = tl.load(C + rn, mask=rn < N, other=0.0).to(tl.float32)
        acc += bias[None, :]

    for _ in range(K, 0, -BLOCK_K):
        # load then increment pointers so that the next blocks of A and B
        # are loaded during the next iteration
        a = tl.load(A, mask=(rm[:, None] < M) & (rk[None, :] < K))
        A += BLOCK_K

        w = tl.load(W, mask=(rn[None, :] < N) & (rk[:, None] < K))
        W += BLOCK_K * stride_wk

        # block level matrix multiplication
        acc += tl.dot(a, w).to(tl.float32)

    # optional: save the activation inputs
    if META["SAVE_ACT_INPUTS"]:
        In += rm[:, None] * stride_im + rn[None, :]
        tl.store(In, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))

    # optional: fused activation (while the data is in shared memory)
    if META["ACTIVATION"]:
        acc = META["ACTIVATION"](acc)

    # write back result
    tl.store(D, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def _sanitize_inputs(x, weight, bias):
    assert (
        x.shape[1] == weight.shape[1]
    ), f"Incompatible dimensions in between inputs and weight, {x.shape} - {weight.shape}"
    assert bias is None or bias.is_contiguous()
    assert (
        bias is None or bias.shape[0] == weight.shape[0]
    ), "Incompatible dimensions in between weight and bias"

    assert x.is_contiguous(), "A contiguous input is required"


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

    # Fold the batch dimension, if any
    x_ = x.flatten(0, 1) if x.ndim == 3 else x
    _sanitize_inputs(x_, weight, bias)

    M, K = x_.shape
    N, K = weight.shape

    outputs = torch.empty((M, N), device=x.device, dtype=x.dtype)
    act_inputs = torch.empty_like(outputs) if save_inputs else outputs

    # 1D launch kernel where each block gets its own program.
    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        )

    # fmt: off
    kernel_fma[grid](
        # data ptrs
        outputs, x_, weight,
        bias if bias is not None else x,  # auto skip bias if not present
        act_inputs,
        # shapes
        M, N, K,
        # strides
        outputs.stride(0),
        x_.stride(0),
        weight.stride(0), weight.stride(1),
        act_inputs.stride(0),
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

    if x.ndim == 3:
        outputs = outputs.reshape(x.shape[0], x.shape[1], N)

    return (outputs, act_inputs) if save_inputs else (outputs, None)


# fmt: off
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64 , 'BLOCK_K': 256}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64 , 'BLOCK_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64 , 'BLOCK_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_K': 64}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def kernel_grad_inputs(
    # Pointers to all the tensors
    GRAD_IN, GRAD_ACT, GRAD_OUT, ACT_IN, WEIGHT,
    # Tensor dimensions
    M, N, K,
    # strides for all the gradients
    stride_gim, stride_gam, stride_gom,
    # strides for the extra data
    stride_aim, stride_wn, stride_wk,
    # Meta-parameters
    **META,
):
    # fmt: on
    """
    Compute the input gradients.

    Shapes:
        GRAD_IN     (M, K)
        GRAD_ACT    (M, N)
        GRAD_OUT    (M, N)
        WEIGHT      (K, N)
        ACT_IN      (M, N)

    The main matmul is
        GRAD_IN <- GRAD_OUT x WEIGHT^T
        (M, K)  <- (M, N) x (N, K)
        we're accumulating on N
    """

    # extract metaparameters
    BLOCK_M, GROUP_M = META["BLOCK_M"], META["GROUP_M"]
    BLOCK_N, BLOCK_K = META["BLOCK_N"], META["BLOCK_K"]

    # programs are grouped together to improve L2 hit rate
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_k = tl.cdiv(K, BLOCK_K)
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
    grad_out_ptrs = GRAD_OUT + rm[:, None] * stride_gom + rn[None, :]
    grad_act_ptrs = GRAD_ACT + rm[:, None] * stride_gam + rn[None, :]
    act_in_ptrs = ACT_IN + rm[:, None] * stride_aim + rn[None, :]
    grad_in_ptrs = GRAD_IN + rm[:, None] * stride_gim + rk[None, :]
    w_ptrs = WEIGHT + rn[:, None] * stride_wn + rk[None, :] * stride_wk

    # initialize and iteratively update accumulator
    grad_in_acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    act_grad_fn = META["ACTIVATION_GRAD"]

    mask_mk = (rm[:, None] < M) & (rk[None, :] < K)

    for _ in range(N, 0, -BLOCK_N):
        grad_out = tl.load(grad_out_ptrs)
        grad_out_ptrs += BLOCK_N

        w = tl.load(w_ptrs)
        w_ptrs += BLOCK_N * stride_wn

        # optional fused activation gradient (while the data is in shared memory)
        if META["ACTIVATION_GRAD"]:
            if META["ACTIVATION_GRAD_REQ_INPUTS"]:
                # This activation requires its inputs
                act_input = tl.load(act_in_ptrs)
                act_in_ptrs += BLOCK_N

                grad_act = act_grad_fn(act_input)
            else:
                # Save some time, we can reuse the outputs to know about the grad
                grad_act = act_grad_fn(grad_out)

            grad_out *= grad_act.to(grad_out.dtype)

        # store grad_act as an intermediate, will be used for grad/weight
        if META["SAVE_ACT_GRAD"]:
            tl.store(grad_act_ptrs, grad_out)
            grad_act_ptrs += BLOCK_N

        # gradient #1: input with respect to outputs
        # grad_in is grad_out scaled by the (transposed) weight
        grad_in_acc += tl.dot(grad_out, w)

    # write back result
    # automatic type promotion/downgrade
    tl.store(grad_in_ptrs, grad_in_acc, mask=mask_mk)


# fmt: off
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_N': 64 , 'BLOCK_K': 256}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_N': 64 , 'BLOCK_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_N': 64 , 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_N': 64 , 'BLOCK_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_N': 32 , 'BLOCK_K': 64}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_N': 32 , 'BLOCK_K': 32}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def kernel_grad_bias_weight(
    # Pointers to all the tensors
    GRAD_BIAS, GRAD_WEIGHT, GRAD_ACT, INPUTS,
    # Tensor dimensions
    M, N, K,
    # strides for all the gradients & inputs
    stride_gwn, stride_gam, stride_im,
    # Meta-parameters
    **META,
):
    # fmt: on
    """
    Compute the grad and weight gradients.

    Shapes:
        INPUTS      (M, K)
        GRAD_ACT    (M, N)
        GRAD_BIAS   (N)
        GRAD_WEIGHT (N, K)

    The main matmul is
        GRAD_WEIGHT <- GRAD_ACT^T x INPUTS
        (N, K)      <- (N, M)     x (M, K)
        meaning that we consolidate over M
    """

    # extract metaparameters
    BLOCK_M, GROUP_N = META["BLOCK_M"], META["GROUP_N"]
    BLOCK_N, BLOCK_K = META["BLOCK_N"], META["BLOCK_K"]

    # get the kernel ids, aka where are we ?
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_k = tl.cdiv(K, BLOCK_K)
    num_pid_in_group = GROUP_N * num_pid_k  # number of programs in group
    group_id = pid // num_pid_in_group  # id of the group this program is in
    first_pid_n = group_id * GROUP_N  # row-id of the first program in the group
    GROUP_N = min(
        num_pid_n - first_pid_n, GROUP_N
    )  # if `num_pid_m` isn't divisible by `GROUP_M`, the last group is smaller

    # *within groups*, programs are ordered in a column-major order
    pid_n = first_pid_n + (pid % GROUP_N)  # row-id of the program in the *launch grid*
    pid_k = (
        pid % num_pid_in_group
    ) // GROUP_N  # col-id of the program in the *launch grid*

    # memory ranges
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    rm = tl.arange(0, BLOCK_M)

    # set the starting points for all the pointers
    grad_act_ptrs = GRAD_ACT + rn[:, None] + rm[None, :] * stride_gam    # N x M
    inputs_ptrs = INPUTS + rm[:, None] * stride_im + rk[None, :]         # M x K

    if META["COMPUTE_GRAD_BIAS"]:
        grad_bias = tl.zeros((BLOCK_N,), tl.float16)
        if META["FP_32"]:
            grad_bias = grad_bias.to(tl.float32)

    grad_weight_acc = tl.zeros((BLOCK_N, BLOCK_K), tl.float32)

    # matmul + fused grad bias.
    # consolidate over M
    mask_nm = ((rn[:, None] < N) & (rm[None, :] < M))
    mask_mk = (rm[:, None] < M) & (rk[None, :] < K)

    for _ in range(M, 0, -BLOCK_M):
        grad_act = tl.load(grad_act_ptrs, mask=mask_nm, other=0.)
        grad_act_ptrs += BLOCK_M * stride_gam

        if META["COMPUTE_GRAD_WEIGHT"]:
            # GRAD_WEIGHT <- GRAD_ACT^T x INPUTS
            inputs = tl.load(inputs_ptrs, mask=mask_mk, other=0.)
            grad_weight_acc += tl.dot(grad_act, inputs)
            inputs_ptrs += BLOCK_M * stride_im

        if META["COMPUTE_GRAD_BIAS"]:
            # grad bias is just the accumulation of grad_act over M
            grad_bias += tl.sum(grad_act, axis=1)

    # epilogue, save whatever is needed
    if META["COMPUTE_GRAD_BIAS"]:
        grad_bias_ptrs = GRAD_BIAS + rn
        tl.store(grad_bias_ptrs, grad_bias, mask=rn < N)

    if META["COMPUTE_GRAD_WEIGHT"]:
        grad_weight_ptrs = GRAD_WEIGHT + rn[:, None] * stride_gwn + rk[None, :]
        tl.store(grad_weight_ptrs, grad_weight_acc, mask=((rn[:, None] < N) & (rk[None, :] < K)))


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

    grad_out = grad_out.flatten(0, 1) if grad_out.ndim == 3 else grad_out
    if grad_out.stride(1) != 1:
        grad_out.contiguous()

    assert (
        grad_out.shape[1] == weight.shape[0]
    ), "Incompatible dimensions in between grad_out and weight"

    M, N = grad_out.shape
    N, K = weight.shape

    grad_in = torch.empty((M, K), device=grad_out.device, dtype=grad_out.dtype)
    grad_act = torch.empty_like(grad_out)
    activation_inputs = grad_out if activation_inputs is None else activation_inputs

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(K, META["BLOCK_K"]),
        )

    # Compute the gradient for the inputs + partial reduction for grad bias
    # fmt: off
    # (M, K) <- (M, N) x (N, K)
    kernel_grad_inputs[grid](
        # data ptrs
        grad_in, grad_act, grad_out,
        activation_inputs, weight,
        # shapes
        M, N, K,
        # strides
        grad_in.stride(0), grad_act.stride(0), grad_out.stride(0),
        activation_inputs.stride(0), weight.stride(0), weight.stride(1),
        # optional fused activation
        ACTIVATION_GRAD=activation_grad,
        GROUP_M=8,  # L2 data reuse optimization
        BLOCK_N=32,
        ACTIVATION_GRAD_REQ_INPUTS=activation_grad_req_inputs,
        SAVE_ACT_GRAD=trainable_weight or trainable_bias,
        GRAD_BIAS=trainable_bias
    )
    # fmt: on

    grad_weight = None
    grad_bias = None

    _fuse_bias_gradient_computation = True

    if trainable_bias or trainable_weight:
        # NOTE: the bias gradient computation is not fused here, as it currently leads to worse performance

        # placeholders if in need
        grad_weight = torch.empty((N, K), device=grad_out.device, dtype=grad_out.dtype) \
            if trainable_weight else grad_act

        grad_bias = torch.empty((N), device=grad_out.device, dtype=grad_out.dtype) if trainable_bias else grad_act

        def grid(META):
            return (
                triton.cdiv(N, META["BLOCK_N"]) * triton.cdiv(K, META["BLOCK_K"]),
            )

        # fmt: off
        inputs_ = inputs.flatten(0, 1) if inputs.ndim == 3 else inputs

        # (N, K) <- (N, M) x (M, K)
        kernel_grad_bias_weight[grid](
            # data ptrs
            grad_bias, grad_weight, grad_act, inputs_,
            # shapes
            M, N, K,
            # strides
            grad_weight.stride(0), grad_act.stride(0), inputs_.stride(0),
            COMPUTE_GRAD_BIAS=_fuse_bias_gradient_computation and trainable_bias,
            COMPUTE_GRAD_WEIGHT=trainable_weight,
            GROUP_N=8,  # L2 data reuse optimization
            BLOCK_M=32,
            FP_32=(inputs.dtype == torch.float32)
        )

        if trainable_bias and not _fuse_bias_gradient_computation:
            grad_bias = torch.sum(grad_act, dim=0)

        if not trainable_weight:
            grad_weight = None

        if not trainable_bias:
            grad_bias = None

    del grad_act

    return (
            grad_in.reshape(inputs.shape),
            grad_weight,
            grad_bias
        )
