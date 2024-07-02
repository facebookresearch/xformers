# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional, Tuple

import torch

from xformers.ops.fmha.attn_bias import (  # type: ignore
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
)

from .. import _is_triton_available


def rope_padded(
    xq: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    attn_bias: BlockDiagonalCausalWithOffsetPaddedKeysMask,
    *,
    theta: float = 10000.0,
    linear_scale: float = 1.0,
    out_q: Optional[torch.Tensor] = None,
    first_seqpos: Optional[torch.Tensor] = None,
    seqpos: Optional[torch.Tensor] = None,
    adjacents: bool = True,
    internal_dtype: str = "",
):
    """
    Performs RoPE (rotary embeddings) and kv-cache emplacement for a heterogeneous
    batch for inference in the style given by
    BlockDiagonalCausalWithOffsetPaddedKeysMask.
    The batch is concatenated along the sequence dimension, so the
    actual dim-0 length of all tensors is 1.

    xq, xk and xv should be (1, slen, n_heads, dim), where
    xq's n_heads can differ from xk and xv.

    This function places the roped xk in the right place in cache_k, and
    xv (unmodified) in the right place in cache_v, and returns out_q
    (the roped xq) such that things are ready to call

    xformers.ops.memory_efficient_attention(
        out_q, cache_k, cache_v, attn_bias=attn_bias
    )

    This functionality is experimental. Its API might be changed without warnings.
    Use it at your own risk.

    Arguments:
        xq: tensor of queries to apply rope to
        xk: tensor of keys to apply rope to
        xv: tensor of values to copy into cache_v
        cache_k: cache of keys, MODIFIED IN PLACE
        cache_v: cache of values, MODIFIED IN PLACE
        attn_bias: details the layout of caches.
                Used to determine frequencies for the
                RoPE calculation as well as the locations in cache_k and cache_v
                to write to. Must be on the device.
        first_seqpos: Optionally a tensor containing the sequence position of the
                    beginning of the cache for each batch element.
                    Providing a tensor of zeros is the same as providing None.
                    This affects the numerical calculation but not which memory
                    locations are read or written.
        seqpos: Optionally a 1D tensor containing the sequence position of each
                    query. This should have length equal to xq.shape[1] .
                    This affects the numerical calculation but not which memory
                    locations are read or written.
        adjacents: If True, the inputs are in adjacent pairs along the final dim axis.
                  This is like the released LLaMA model.
                  If False, the dim axis is split in two equal pieces.
                   I.e. the features are ordered with all the real parts before all
                   the imaginary parts. This matches HuggingFace, e.g.
                   https://github.com/huggingface/transformers/blob/
                   f143037789288ba532dada934a118e648e715738/
                   src/transformers/models/llama/modeling_llama.py#L126-L130
        linear_scale: A scaling factor to apply to the sequence ids when computing
                      the RoPE frequencies.  When set to K, all sequence indices
                      are divided by K.
        internal_dtype: set to "f32" or "f64" to enforce dtype in the calculation
    """
    if torch.is_grad_enabled() and (
        xq.requires_grad
        or xk.requires_grad
        or xv.requires_grad
        or cache_k.requires_grad
        or cache_v.requires_grad
        or out_q is not None
    ):
        raise ValueError("Gradients not supported.")
    assert _is_triton_available()
    import triton

    from ._triton.rope_padded_kernels import _rope_padded_kernel

    n_total_queries = attn_bias.q_seqinfo.seqstart_py[-1]
    cache_length = attn_bias.k_seqinfo.seqstart_py[-1]
    ndim = xq.ndim
    if ndim not in [4, 5]:
        raise ValueError("Unexpected xq dimension")
    xq_stride = xq.stride()
    xk_stride = xk.stride()
    xv_stride = xv.stride()
    cache_k_stride = cache_k.stride()
    cache_v_stride = cache_v.stride()
    cache_k_shape = cache_k.shape
    xk_shape = xk.shape
    n_kv_heads = xk_shape[-2]
    expected_kv_heads = n_kv_heads
    if xk_stride[-2] == 0:
        n_kv_heads = 1
    expected_cache_heads = n_kv_heads
    if n_kv_heads == 1 and cache_k_stride[-2] == 0:
        # If there's 1 kv head, don't care how expanded
        # cache_k is. User might expand before or after rope.
        expected_cache_heads = cache_k_shape[-2]

    if ndim == 4:
        bsz, q_len, n_q_heads, dim = xq.shape
        assert q_len == n_total_queries
        if xk_shape != (1, n_total_queries, expected_kv_heads, dim):
            raise ValueError(
                f"unexpected k shape {xk_shape}: expected {(1, n_total_queries, expected_kv_heads, dim)}"
            )
        if xv.shape != (1, n_total_queries, expected_kv_heads, dim):
            raise ValueError(
                f"unexpected v shape {xv.shape}: expected {(1, n_total_queries, expected_kv_heads, dim)}"
            )
        if cache_k_shape != (1, cache_length, expected_cache_heads, dim):
            raise ValueError("unexpected cache_k shape")
        if cache_v.shape != (1, cache_length, expected_cache_heads, dim):
            raise ValueError("unexpected cache_v shape")
        n_groups = 1
        out_q_stride: Tuple[int, ...] = (0, n_q_heads * dim, dim, 1)

    else:
        bsz, q_len, n_groups, n_q_heads, dim = xq.shape
        assert q_len == n_total_queries
        if xk_shape != (1, n_total_queries, n_groups, expected_kv_heads, dim):
            raise ValueError(
                f"unexpected k shape {xk_shape}: expected {(1, n_total_queries, n_groups, expected_kv_heads, dim)}"
            )
        if xv.shape != (1, n_total_queries, n_groups, expected_kv_heads, dim):
            raise ValueError(
                f"unexpected v shape {xv.shape}: expected {(1, n_total_queries, n_groups, expected_kv_heads, dim)}"
            )
        if cache_k_shape != (1, cache_length, n_groups, expected_cache_heads, dim):
            raise ValueError(
                f"unexpected cache_k shape {cache_k_shape}: "
                f"expected {(1, cache_length, n_groups, expected_cache_heads, dim)}"
            )
        if cache_v.shape != (1, cache_length, n_groups, expected_cache_heads, dim):
            raise ValueError(
                f"unexpected cache_v shape {cache_v.shape}: "
                f"expected {(1, cache_length, n_groups, expected_cache_heads, dim)}"
            )
        out_q_stride = (
            0,
            n_q_heads * dim * n_groups,
            n_q_heads * dim,
            dim,
            1,
        )

    if bsz != 1:
        raise ValueError(
            "Expected batch size dimension to be 1 as batches should be concatenated."
        )
    if xq_stride[-1] != 1:
        raise ValueError("Each q head must be contiguous")
    if xk_stride[-1] != 1:
        raise ValueError("Each k head must be contiguous")
    if xv_stride[-1] != 1:
        raise ValueError("Each v head must be contiguous")
    if cache_k_stride[-1] != 1:
        raise ValueError("Each cache_k head must be contiguous")
    if cache_v_stride[-1] != 1:
        raise ValueError("Each cache_v head must be contiguous")
    n_total_heads = n_q_heads + 2 * n_kv_heads
    v_start = n_total_heads - n_kv_heads
    k_start = n_q_heads
    if out_q is None:
        out_q = xq.new_empty(xq.shape)
    else:
        if out_q.shape != xq.shape:
            raise ValueError("Unexpected shape of out_q")
        out_q_stride = out_q.stride()
        if out_q_stride[-1] != 1:
            raise ValueError("Each out_q head must be contiguous")

    assert out_q is not None

    logical_bsz = len(attn_bias.q_seqinfo.seqstart_py) - 1

    if first_seqpos is not None and seqpos is not None:
        raise ValueError("seqpos and first_seqpos may not both be provided")
    stride_seqpos = 0
    if first_seqpos is not None:
        if first_seqpos.shape != (logical_bsz,):
            shape = tuple(first_seqpos.shape)
            raise ValueError(
                f"first_seqpos.shape {shape} but ({logical_bsz},) expected."
            )
        stride_seqpos = first_seqpos.stride(0)
    elif seqpos is not None:
        if seqpos.shape != (n_total_queries,):
            shape = tuple(seqpos.shape)
            raise ValueError(f"seqpos.shape {shape} but ({n_total_queries},) expected.")
        stride_seqpos = seqpos.stride(0)

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // xq.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(dim))
    BLOCK_SIZE = max(BLOCK_SIZE, 128)
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)
    # heuristics for number of warps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    device = xq.device
    seqstartq = attn_bias.q_seqinfo.seqstart
    seqstartk = attn_bias.k_seqinfo.seqstart
    seqlenk = attn_bias.k_seqinfo.seqlen
    if (
        seqstartq.device != device
        or seqstartk.device != device
        or seqlenk.device != device
    ):
        raise ValueError("`attn_bias` must be on the same device as the other inputs")
    assert internal_dtype in ["", "f32", "f64"]
    # experiment with the order of dims here.
    with torch.cuda.device(xq.device):
        _rope_padded_kernel[
            (attn_bias.q_seqinfo.max_seqlen, logical_bsz, n_total_heads * n_groups)
        ](
            xq,
            xk,
            xv,
            out_q,
            cache_k,
            cache_v,
            seqstartq,
            seqstartk,
            seqlenk,
            theta,
            linear_scale,
            first_seqpos,
            seqpos,
            k_start,
            v_start,
            n_groups,
            dim,
            xq_stride[1],
            xq_stride[2] if ndim == 5 else 0,
            xq_stride[-2],
            xk_stride[1],
            xk_stride[2] if ndim == 5 else 0,
            xk_stride[-2],
            xv_stride[1],
            xv_stride[2] if ndim == 5 else 0,
            xv_stride[-2],
            cache_k_stride[1],
            cache_k_stride[2] if ndim == 5 else 0,
            cache_k_stride[-2],
            cache_v_stride[1],
            cache_v_stride[2] if ndim == 5 else 0,
            cache_v_stride[-2],
            seqstartq.stride(0),
            seqstartk.stride(0),
            seqlenk.stride(0),
            out_q_stride[1],
            out_q_stride[2] if ndim == 5 else 0,
            out_q_stride[-2],
            stride_seqpos,
            internal_dtype,
            const_batch_strides=False,
            cache_padding_length=0,
            seqlenk_shift=0,
            BLOCK_SIZE=BLOCK_SIZE,
            adjacents=adjacents,
            num_warps=num_warps,
        )
    return out_q
