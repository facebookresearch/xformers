# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import os
from typing import List, Optional

import torch
import torch.multiprocessing.reductions
from torch.utils._pytree import tree_flatten, tree_unflatten
from typing_extensions import Annotated

from .. import _is_triton_available
from .common import Alias, make_pytorch_operator_for_dispatch_key

if _is_triton_available():
    from ._triton.tiled_matmul_kernels import _launch_triton_matmul

    TRITON_IS_AVAILABLE = True
else:
    TRITON_IS_AVAILABLE = False


# Copied over from the sequence parallel fused ops.
def _should_use_triton(device: torch.device, dtype: torch.dtype) -> bool:
    if not int(os.getenv("XFORMERS_TILED_MATMUL_ENABLE_TRITON", "1")):
        return False
    if not TRITON_IS_AVAILABLE:
        return False
    device_capability = torch.cuda.get_device_capability(device)
    # Triton seems to be having issues on P100 and V100 GPUs, such as
    # https://github.com/openai/triton/issues/1609
    # https://github.com/openai/triton/issues/1610
    # https://github.com/openai/triton/issues/1257#issuecomment-1532616965
    # and, in recent Triton versions (Jan 2024), returning wrong values.
    if device_capability < (8, 0):
        return False
    return True


# We can't use make_pytorch_cuda_operator because PyTorch isn't able to inspect
# Tensor[][] args to detect they contain CUDA args. Thus we need to register
# this as a fallback implementation, so it gets invoked regardless of the args.
# See: https://github.com/pytorch/pytorch/issues/113022
@make_pytorch_operator_for_dispatch_key("")
def tiled_matmul_fwd(
    a: List[List[torch.Tensor]],
    b: List[List[torch.Tensor]],
    out: Optional[List[List[Annotated[torch.Tensor, Alias("a", write=True)]]]] = None,
) -> List[List[Annotated[torch.Tensor, Alias("a", write=True)]]]:
    assert len(a) >= 1 and len(a[0]) >= 1 and all(len(row) == len(a[0]) for row in a), (
        "the first operand must be a non-empty two-dimensional regular list of lists "
        "of tenors"
    )
    assert len(b) >= 1 and len(b[0]) >= 1 and all(len(row) == len(b[0]) for row in b), (
        "the second operand must be a non-empty two-dimensional regular list of lists "
        "of tenors"
    )

    m_tiles = len(a)
    k_tiles = len(a[0])
    assert len(b) == k_tiles, (
        "the first operand's inner dimension must match the second operand's outer "
        f"dimension, got {k_tiles} and {len(b)}"
    )
    n_tiles = len(b[0])

    ms = [a[tile_m][0].shape[0] for tile_m in range(m_tiles)]
    ns = [b[0][tile_n].shape[1] for tile_n in range(n_tiles)]
    aks = [a[0][tile_k].shape[1] for tile_k in range(k_tiles)]
    bks = [b[tile_k][0].shape[0] for tile_k in range(k_tiles)]

    for tile_m in range(m_tiles):
        for tile_k in range(k_tiles):
            assert a[tile_m][tile_k].shape[0] == ms[tile_m], (
                f"the tensors on row {tile_m} of the first operand must all have the "
                f"same size along the m dimension, got {ms[tile_m]} at position 0 and "
                f"{a[tile_m][tile_k].shape[0]} at position {tile_k}"
            )
            assert a[tile_m][tile_k].shape[1] == aks[tile_k], (
                f"the tensors on column {tile_k} of the first operand must all have "
                f"the same size along the k dimension, got {aks[tile_k]} at position 0 "
                f"and {a[tile_m][tile_k].shape[1]} at position {tile_m}"
            )

    for tile_n in range(n_tiles):
        for tile_k in range(k_tiles):
            assert b[tile_k][tile_n].shape[0] == bks[tile_k], (
                f"the tensors on row {tile_k} of the second operand must all have the "
                f"same size along the k dimension, got {bks[tile_k]} at position 0 and "
                f"{b[tile_k][tile_n].shape[0]} at position {tile_n}"
            )
            assert b[tile_k][tile_n].shape[1] == ns[tile_n], (
                f"the tensors on column {tile_n} of the second operand must all have "
                f"the same size along the n dimension, got {ns[tile_n]} at position 0 "
                f"and {b[tile_k][tile_n].shape[1]} at position {tile_k}"
            )

    for tile_k in range(k_tiles):
        assert aks[tile_k] == bks[tile_k], (
            f"the tensors on column {tile_k} of the first operand and those on row "
            f"{tile_k} of the second operand must have the same size along the k "
            f"dimension, got {aks[tile_k]} and {bks[tile_k]}"
        )
    ks = aks

    if out is not None:
        assert (
            len(out) >= 1
            and len(out[0]) >= 1
            and all(len(row) == len(out[0]) for row in out)
        ), "out must be a non-empty two-dimensional regular list of lists of tenors"
        assert len(out) == m_tiles
        assert len(out[0]) == n_tiles
        cms = [out[tile_m][0].shape[0] for tile_m in range(m_tiles)]
        cns = [out[0][tile_n].shape[1] for tile_n in range(n_tiles)]
        for tile_m in range(m_tiles):
            for tile_n in range(n_tiles):
                assert out[tile_m][tile_n].shape[0] == cms[tile_m], (
                    f"the tensors on row {tile_m} of out must all have the same size "
                    f"along the m dimension, got {cms[tile_m]} at position 0 and "
                    f"{out[tile_m][tile_n].shape[0]} at position {tile_n}"
                )
                assert out[tile_m][tile_n].shape[1] == cns[tile_n], (
                    f"the tensors on column {tile_n} of out must all have the same "
                    f"size along the k dimension, got {cns[tile_n]} at position 0 and "
                    f"{out[tile_m][tile_n].shape[1]} at position {tile_m}"
                )
        for tile_m in range(m_tiles):
            assert cms[tile_m] == ms[tile_m], (
                f"the tensors on row {tile_m} of out and those on row {tile_m} of the "
                f"first operand must have the same size along the m dimension, got "
                f"{cms[tile_m]} and {ms[tile_m]}"
            )
        for tile_n in range(n_tiles):
            assert cns[tile_n] == ns[tile_n], (
                f"the tensors on column {tile_n} of out and those on column {tile_n} "
                f"of the second operand must have the same size along the n dimension, "
                f"got {cns[tile_n]} and {ns[tile_n]}"
            )
        c = out
    else:
        c = [[a[0][0].new_empty((m, n)) for n in ns] for m in ms]

    # TODO We can try merging tiles that come from contiguous memory, using
    # stack_or_none, to further improve performance.

    # Because the Triton kernel is hardcoded for maximum three tiles.
    # Because, in turn, we aimed this at the fusion of wq/wk/wv.
    if (
        m_tiles <= 3
        and k_tiles <= 3
        and n_tiles <= 3
        and _should_use_triton(a[0][0].device, a[0][0].dtype)
    ):
        _launch_triton_matmul(a, b, c, ms, ns, ks)
    else:
        for tile_m in range(len(ms)):
            for tile_n in range(len(ns)):
                torch.mm(a[tile_m][0], b[0][tile_n], out=c[tile_m][tile_n])
                for tile_k in range(1, len(ks)):
                    c[tile_m][tile_n].addmm_(a[tile_m][tile_k], b[tile_k][tile_n])

    return c


def _transpose(x: List[List[torch.Tensor]]) -> List[List[torch.Tensor]]:
    return [[t.t() for t in y] for y in zip(*x)]


class _TiledMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ab_tree_spec, *ab_tree_values):
        ctx.ab_tree_spec = ab_tree_spec
        ctx.save_for_backward(*ab_tree_values)
        a, b = tree_unflatten(list(ab_tree_values), ab_tree_spec)

        c = tiled_matmul_fwd(a, b)

        c_tree_values, c_tree_spec = tree_flatten(c)
        ctx.c_tree_spec = c_tree_spec
        return (c_tree_spec,) + tuple(c_tree_values)

    @staticmethod
    def backward(ctx, _none, *grad_c_tree_values):
        a, b = tree_unflatten(list(ctx.saved_tensors), ctx.ab_tree_spec)
        grad_c = tree_unflatten(list(grad_c_tree_values), ctx.c_tree_spec)

        grad_a = tiled_matmul_fwd(grad_c, _transpose(b))
        grad_b = tiled_matmul_fwd(_transpose(a), grad_c)

        grad_ab_tree_values, grad_ab_tree_spec = tree_flatten((grad_a, grad_b))
        return (None,) + tuple(grad_ab_tree_values)


def tiled_matmul(
    a: List[List[torch.Tensor]],
    b: List[List[torch.Tensor]],
) -> List[List[torch.Tensor]]:
    """Multiply two matrices given as grids of tiles

    It performs the matmul between A and B, which are given as two-dimensional
    grids of tiles (i.e., blocks), represented as lists of lists of tensors.
    The output will itself be a matrix in such a form. Formally:

        out[m][n] = sum(a[m][k] @ b[k][n] for k in range(...))

    with the obvious constraints needed to make it work, in terms of number of
    tiles and sizes of each tile.

    The interest of this operator is to improve performance by avoding wave
    quantization effects when doing independent matrix multiplications in
    series. Sometimes, when these matmuls have one operand in common, this can
    also be addressed by concatenating the other operands into a single matrix,
    and issuing a single matmul. However this isn't always possible (e.g., might
    break the checkpoint format) and it's an anti-pattern, as it obscures the
    logic (e.g., changing the modelling code out of performance reasons). This
    tiled matmul performs the same computation as if the matrices were merged,
    without merging them, simply through a smarter memory addressing scheme.

    The tiled matmul is less generic than a grouped matmul, which can also help
    with wave quantization, and doesn't need the matmuls to have the same lhs
    or rhs operand. However, a grouped matmul will write the result of each
    matmul to a separate output matrix, whereas the tiled matmul allows to add
    them together into a single output. This is needed during the backward pass
    of a linear layer, and it's the reason we wrote this instead of using a
    grouped matmul.

    The tiled matmul is implemented using a custom Triton kernel, which puts
    constraints on the strides of the tiles. All rows of A must have the same
    K stride, all columns of A must have the same M stride, and so on.

    Currently the tiled matmul supports at most three tiles on each dimension,
    although fewer can also be given. This is because we needed it to fuse the
    query, key and value weights of an attention layer. This limit can be
    increased if needed.

    This operator is differentiable.

    """
    ab_tree_values, ab_tree_spec = tree_flatten((a, b))
    c_tree_spec, *c_tree_values = _TiledMatmul.apply(ab_tree_spec, *ab_tree_values)
    c = tree_unflatten(list(c_tree_values), c_tree_spec)

    return c
