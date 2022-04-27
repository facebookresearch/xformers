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
    'EVEN_BLOCKS': lambda args:
        args["M"] % (args['BLOCK_M']) == 0
        and args["N"] % (args['BLOCK_N']) == 0
})
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 32}, num_warps=2),
        triton.Config({"BLOCK_N": 64}, num_warps=2),
        triton.Config({"BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_N": 512}, num_warps=8),
    ],
    key=["M", "N", "R"],
)
@triton.jit
def kernel_bw_act(
    # Pointers to matrices
    GRAD_ACT, GRAD_BIAS,
    GRAD_OUT, ACT_INPUTS,
    # Matrix dimensions
    M, N, R,  # R is used for autotune, are we reducing or not ?
    stride_gom, stride_aim, stride_bm,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ACTIVATION_GRAD: tl.constexpr,
    COMPUTE_D_BIAS: tl.constexpr,
    EVEN_BLOCKS: tl.constexpr,
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
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # the memory addresses of elements in the first block of
    # A and W can be computed using numpy-style broadcasting
    act_input_ptrs = ACT_INPUTS + rm[:, None] * stride_aim + rn[None, :]

    # compute the gradient which is related to this activation
    if EVEN_BLOCKS:
        act_in = tl.load(act_input_ptrs)
    else:
        mask_rn = rn < N
        mask_rm = rm < M
        act_in = tl.load(act_input_ptrs, mask=mask_rn[None, :] & mask_rm[:, None], other=0.0)

    grad_act = ACTIVATION_GRAD(act_in)

    # now read the incoming gradient, the backpropagated one is the multiple of both
    grad_out_ptrs = GRAD_OUT + rm[:, None] * stride_gom + rn[None, :]
    if EVEN_BLOCKS:
        grad_out = tl.load(grad_out_ptrs)
    else:
        grad_out = tl.load(grad_out_ptrs, mask=mask_rn[None, :] & mask_rm[:, None], other=0.0)

    grad_act *= grad_out

    # write back result
    grad_act_ptrs = GRAD_ACT + rm[:, None] * stride_gom + rn[None, :]

    if EVEN_BLOCKS:
        tl.store(grad_act_ptrs, grad_act)
    else:
        mask_rn = rn < N
        mask_rm = rm < M
        tl.store(grad_act_ptrs, grad_act, mask=mask_rn[None, :] & mask_rm[:, None])

    # opportunistically partially fuse the d_bias computation
    if COMPUTE_D_BIAS:
        if EVEN_BLOCKS:
            tl.store(GRAD_BIAS + pid_m * stride_bm + rn, tl.sum(grad_act, axis=0))
        else:
            d_out = tl.where(mask_rn[None, :] & mask_rm[:, None], grad_act, 0.0)
            d_bias = tl.sum(d_out, axis=0)
            tl.store(GRAD_BIAS + pid_m * stride_bm + rn, d_bias, mask=rn < N)


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

    Contrary to the forward pass, not all the computations are fused here.
    We fuse the

    .. note: The weight buffer is transposed on the fly
    .. note: Activation gradient needs to be a Triton kernel
    """

    # Make sure that we don't have to handle the stride over cols
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()

    grad_out_ = grad_out if grad_out.ndim == 2 else grad_out.flatten(0, 1)
    grad_bias : Optional[torch.Tensor] = None
    grad_weight : Optional[torch.Tensor] = None

    assert grad_out_.shape[1] == weight.shape[0], "Incompatible dimensions in between grad_out and weight"

    M, N = grad_out_.shape
    N, _ = weight.shape

    # Compute the gradient for the activation + bias
    if activation_grad is not None:
        grad_act = torch.empty_like(grad_out_)

        # Some activations do not require their inputs to compute their gradient
        if act_in is None:
            act_in = grad_out_

        if trainable_bias:
            # Bigger BLOCK_M to partially reduce on the fly
            m_cap = 256 if N > 1024 else 1024
            BLOCK_M = min(triton.next_power_of_2(M), m_cap)
        else:
            # We don´t need to reduce anything here
            BLOCK_M = 64

        # Opportunistically compute grad bias if required
        # Note that we won't completely reduce grad_bias, but some of it will be fused with the main backward kernel
        if trainable_bias:
            grad_bias = torch.zeros((triton.cdiv(M, BLOCK_M), N), dtype=grad_act.dtype, device=grad_act.device)
        else:
            grad_bias = grad_out    # not used

        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"])) # noqa

        # fmt: off
        kernel_bw_act[grid](
            grad_act, grad_bias,
            grad_out_, act_in,                      # data ptrs
            M, N, 1 if trainable_bias else 0,       # shapes
            grad_act.stride(0),                     # strides
            act_in.stride(0),
            grad_bias.stride(0),
            ACTIVATION_GRAD=activation_grad,        # optional fused activation
            COMPUTE_D_BIAS=trainable_bias,
            BLOCK_M=BLOCK_M,
        )
        # fmt: on

        if trainable_bias:
            if grad_bias.squeeze_().ndim == 2:
                grad_bias = torch.sum(grad_bias, dim=0)
        else:
            grad_bias = None

        # Backpropagation going up, the reference gradient is now
        # just before the activation
        grad_out_ = grad_act

    # Compute the gradient for the weight.
    # [This is not fused, requires follow up]
    if trainable_weight:
        inputs_ = inputs if inputs.ndim == 2 else inputs.flatten(0, 1)
        grad_weight = grad_out_.transpose(0, 1) @ inputs_

    # Epilogue, could probably be better handled. Both this and the above could be fused
    # [This is not fused either, requires follow up]
    grad_in = grad_out_ @ weight

    if grad_bias is None and trainable_bias:  # If there was no activation, fallback
        grad_bias = torch.sum(grad_out_, dim=0)

    return grad_in.reshape_as(inputs), grad_weight, grad_bias
