# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from xformers.ops import fmha
from xformers.ops.fmha.split_blocks_fairinternal import (
    split_blocks_for_decoding,
    split_blocks_for_prefill,
)

from .utils import cuda_only

compute_capability = (0, 0)
if torch.cuda.is_available():
    compute_capability = torch.cuda.get_device_capability("cuda")
sm80_or_better_only = pytest.mark.skipif(
    torch.version.cuda is not None and compute_capability < (8, 0),
    reason="requires sm80+",
)


def test_split_blocks_for_decoding():
    max_len_kv = 2048
    B = 64
    local_attention_len = 512
    seqlens = torch.randint(
        max_len_kv, size=(B,), device=torch._C._get_accelerator().type
    )
    attn_bias = fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
        q_seqlen=[1] * B, kv_seqlen=seqlens.tolist(), kv_padding=max_len_kv
    )
    chunked_bias = split_blocks_for_decoding(attn_bias, local_attention_len)
    assert chunked_bias.q_seqinfo.seqstart_py == list(range(B + 1))
    assert (chunked_bias.k_seqinfo.seqlen <= local_attention_len).all()
    assert (chunked_bias.k_seqinfo.seqstart >= attn_bias.k_seqinfo.seqstart).all()


@cuda_only
@sm80_or_better_only
@pytest.mark.parametrize("spec_decoding", [False, True])
@pytest.mark.parametrize("paged", [False, True])
def test_split_blocks_decoding_vs_prefill(spec_decoding, paged):
    """
    We should be able to use the prefill split-blocks algo for decoding, and get the same attention output.
    """
    if torch.version.hip:
        op = fmha.ck.FwOp
    elif fmha.flash.FwOp.VARLEN_LSE_PACKED:
        # Modern enough flash attention
        op = fmha.flash.FwOp
    elif spec_decoding:
        # spec_decoding needs variable query length support
        pytest.skip("We have no fallback kernel for spec decoding")
        assert False
    else:
        op = fmha.triton_splitk.FwOp

    AttnBias = (
        fmha.attn_bias.BlockDiagonalPaddedKeysMask
        if spec_decoding
        else fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask
    )
    PagedAttnBias = (
        fmha.attn_bias.PagedBlockDiagonalPaddedKeysMask
        if spec_decoding
        else fmha.attn_bias.PagedBlockDiagonalCausalWithOffsetPaddedKeysMask
    )
    if paged and (PagedAttnBias not in op.SUPPORTED_ATTN_BIAS_TYPES):
        pytest.skip("Not supported bias")

    dtype = torch.bfloat16
    nheads_kv = 1
    nheads_q = 8
    seq_len_q = 5 if spec_decoding else 1
    head_dim = 128
    max_len_kv = 2048
    B = 64
    local_attention_len = 512
    seqlens = torch.randint(
        low=seq_len_q,
        high=max_len_kv,
        size=(B,),
        device=torch._C._get_accelerator().type,
    )
    seqlens[3] = local_attention_len * 2  # corner cases
    # seqlens[4] = local_attention_len * 2  + 3 # reproduces cross boundary case
    seqlens[2] = local_attention_len
    if spec_decoding:
        # Ensure no cross-boundary attention
        non_compliant_mask = (seqlens >= seq_len_q) & (
            seqlens % local_attention_len < seq_len_q
        )
        adjustment = (seqlens % local_attention_len) * non_compliant_mask
        seqlens -= adjustment
    attn_bias = AttnBias.from_seqlens(
        q_seqlen=[seq_len_q] * B, kv_seqlen=seqlens.tolist(), kv_padding=max_len_kv
    )
    if paged:
        page_size = 256
        block_tables = torch.arange(
            B * max_len_kv // page_size, device="cuda", dtype=torch.int32
        ).reshape(B, -1)
    else:
        page_size = None
        block_tables = None
    chunked_bias_decoding = split_blocks_for_decoding(
        attn_bias, local_attention_len, block_tables, page_size
    )
    chunked_bias_prefill = split_blocks_for_prefill(attn_bias, local_attention_len)
    prefill_attn_to_use = chunked_bias_prefill
    if paged:
        attn_batch_size = len(chunked_bias_prefill.k_seqinfo.seqlen)
        if attn_batch_size != block_tables.shape[0]:
            block_tables = block_tables.view(attn_batch_size, -1)
        prefill_attn_to_use = chunked_bias_prefill.make_paged(
            block_tables,
            page_size,
            paged_type=PagedAttnBias,
        )
    # The only difference between attention biases should be that the bias computed
    # using split_blocks_for_prefill contains elements with query len 0.
    decoding_q_lens = [b - a for a, b in chunked_bias_decoding.q_seqinfo.intervals()]
    prefill_q_lens = [b - a for a, b in prefill_attn_to_use.q_seqinfo.intervals()]
    assert [x for x in prefill_q_lens if x > 0] == decoding_q_lens
    filtered_prefill_k_lens = [
        x
        for x, y in zip(prefill_attn_to_use.k_seqinfo.seqlen_py, prefill_q_lens)
        if y > 0
    ]
    assert chunked_bias_decoding.k_seqinfo.seqlen_py == filtered_prefill_k_lens

    q = torch.randn(
        B,
        seq_len_q,
        nheads_q,
        head_dim,
        device=torch._C._get_accelerator().type,
        dtype=dtype,
    )
    k = torch.randn(
        B,
        max_len_kv,
        nheads_kv,
        head_dim,
        device=torch._C._get_accelerator().type,
        dtype=dtype,
    )
    v = torch.randn(
        B,
        max_len_kv,
        nheads_kv,
        head_dim,
        device=torch._C._get_accelerator().type,
        dtype=dtype,
    )

    xq = q.view(1, -1, nheads_q, head_dim)
    xk = k.view(1, -1, nheads_kv, head_dim).expand(1, -1, nheads_q, -1)
    xv = v.view(1, -1, nheads_kv, head_dim).expand(1, -1, nheads_q, -1)

    out_dec, lse_dec = fmha.memory_efficient_attention_forward_requires_grad(
        xq, xk, xv, chunked_bias_decoding, op=fmha.triton_splitk.FwOp
    )

    out_prefill, lse_prefill = fmha.memory_efficient_attention_forward_requires_grad(
        xq, xk, xv, prefill_attn_to_use, op=op
    )

    torch.testing.assert_close(out_dec, out_prefill, rtol=1e-4, atol=5e-3)
    torch.testing.assert_close(lse_dec, lse_prefill, rtol=1e-4, atol=1e-4)
