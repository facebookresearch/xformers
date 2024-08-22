# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import os
from typing import List, Tuple

import torch

from .. import _is_triton_available


# Copied over from the sequence parallel fused ops.
def _should_use_triton(device: torch.device, dtype: torch.dtype) -> bool:
    if not int(os.getenv("XFORMERS_TILED_MATMUL_ENABLE_TRITON", "1")):
        return False
    if not _is_triton_available():
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


def check_inputs(
    a: List[List[torch.Tensor]],
    b: List[List[torch.Tensor]],
) -> Tuple[List[int], List[int], List[int]]:
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

    return ms, ns, ks


def check_output(out: List[List[torch.Tensor]], ms: List[int], ns: List[int]) -> None:
    m_tiles, n_tiles = len(ms), len(ns)
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


# Using out= args in PyTorch is complicated, especially with custom_ops and
# torch.compile (we need to declare our side-effects with mutates_args, which
# then requires a functionalization step, ...). Thus this out= variant of the
# operator is exposed as a PLAIN PYTHON function, and is not compilable nor
# differentiable. It needs to be invoked from within a custom_op elsewhere.
def tiled_matmul_out(
    a: List[List[torch.Tensor]],
    b: List[List[torch.Tensor]],
    out: List[List[torch.Tensor]],
) -> None:
    ms, ns, ks = check_inputs(a, b)
    check_output(out, ms, ns)

    # TODO We can try merging tiles that come from contiguous memory, using
    # stack_or_none, to further improve performance.

    # Because the Triton kernel is hardcoded for maximum three tiles.
    # Because, in turn, we aimed this at the fusion of wq/wk/wv.
    if (
        len(ms) <= 3
        and len(ks) <= 3
        and len(ns) <= 3
        and _should_use_triton(a[0][0].device, a[0][0].dtype)
    ):
        from ._triton.tiled_matmul_kernels import _launch_triton_matmul

        _launch_triton_matmul(a, b, out, ms, ns, ks)
    else:
        for tile_m in range(len(ms)):
            for tile_n in range(len(ns)):
                torch.mm(a[tile_m][0], b[0][tile_n], out=out[tile_m][tile_n])
                for tile_k in range(1, len(ks)):
                    out[tile_m][tile_n].addmm_(a[tile_m][tile_k], b[tile_k][tile_n])


def _flatten(x: List[List[torch.Tensor]], rows: int, cols: int) -> List[torch.Tensor]:
    assert len(x) == rows
    assert all(len(row) == cols for row in x)
    flat_x = [elem for row in x for elem in row]
    assert len(flat_x) == rows * cols
    return flat_x


def _unflatten(
    flat_x: List[torch.Tensor], rows: int, cols: int
) -> List[List[torch.Tensor]]:
    assert len(flat_x) == cols * rows
    x = [
        flat_x[row_offset : row_offset + cols]
        for row_offset in range(0, rows * cols, cols)
    ]
    assert len(x) == rows
    assert all(len(row) == cols for row in x)
    return x


def _flattened_transpose(
    flat_x: List[torch.Tensor], rows: int, cols: int
) -> List[torch.Tensor]:
    x = _unflatten(flat_x, rows, cols)
    transposed_x = [[elem.t() for elem in col] for col in zip(*x)]
    flat_transposed_x = _flatten(transposed_x, cols, rows)
    return flat_transposed_x


# PyTorch (custom_op and torch.compile, but also the dispatcher in general)
# have a hard time with Tensor[][] args. Thus we flatten them into Tensor[] to
# pass them into and out of this operator.
# See: https://github.com/pytorch/pytorch/issues/113022
@torch.library.custom_op(
    "xformers_python::tiled_matmul_fwd",
    mutates_args=(),
    device_types="cuda",
)
def tiled_matmul_fwd(
    flat_a: List[torch.Tensor],
    flat_b: List[torch.Tensor],
    ms: List[int],
    ns: List[int],
    ks: List[int],
) -> List[torch.Tensor]:
    a = _unflatten(flat_a, len(ms), len(ks))
    b = _unflatten(flat_b, len(ks), len(ns))

    c = [[a[0][0].new_empty((m, n)) for n in ns] for m in ms]
    tiled_matmul_out(a, b, out=c)

    return _flatten(c, len(ms), len(ns))


@torch.library.register_fake("xformers_python::tiled_matmul_fwd")
def tiled_matmul_fwd_fake(
    flat_a: List[torch.Tensor],
    flat_b: List[torch.Tensor],
    ms: List[int],
    ns: List[int],
    ks: List[int],
) -> List[torch.Tensor]:
    c = [[flat_a[0][0].new_empty((m, n)) for n in ns] for m in ms]
    return _flatten(c, len(ms), len(ks))


def tiled_matmul_setup_context(ctx, inputs, output):
    flat_a, flat_b, ctx.ms, ctx.ns, ctx.ks = inputs
    ctx.save_for_backward(*flat_a, *flat_b)


def tiled_matmul_bwd(ctx, flat_grad_c):
    assert len(ctx.saved_tensors) == len(ctx.ms) * len(ctx.ks) + len(ctx.ks) * len(
        ctx.ns
    )
    flat_a = ctx.saved_tensors[: len(ctx.ms) * len(ctx.ks)]
    flat_b = ctx.saved_tensors[-len(ctx.ks) * len(ctx.ns) :]

    flat_transposed_a = _flattened_transpose(flat_a, len(ctx.ms), len(ctx.ks))
    flat_transposed_b = _flattened_transpose(flat_b, len(ctx.ks), len(ctx.ns))

    flat_grad_a = tiled_matmul_fwd(
        flat_grad_c, flat_transposed_b, ctx.ms, ctx.ks, ctx.ns
    )
    flat_grad_b = tiled_matmul_fwd(
        flat_transposed_a, flat_grad_c, ctx.ks, ctx.ns, ctx.ms
    )

    return flat_grad_a, flat_grad_b, None, None, None


torch.library.register_autograd(
    "xformers_python::tiled_matmul_fwd",
    tiled_matmul_bwd,
    setup_context=tiled_matmul_setup_context,
)


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
    # Inputs are checked inside the op, but we check them as well to make sure
    # that they are "regular" and can be flattened.
    ms, ns, ks = check_inputs(a, b)
    flat_a = _flatten(a, len(ms), len(ks))
    flat_b = _flatten(b, len(ks), len(ns))
    flat_c = tiled_matmul_fwd(flat_a, flat_b, ms, ns, ks)
    c = _unflatten(flat_c, len(ms), len(ns))
    return c
