# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import math
from typing import Callable, List, Optional, Tuple, Type

import pytest
import torch
from xformers.ops import fmha
from xformers.ops.fmha.common import AttentionFwOpBase
from xformers.ops.fmha.merge_training import (
    memory_efficient_attention_partial_autograd,
    merge_attentions_autograd,
)

from .utils import assert_allclose, disable_on_rocm

compute_capability = (0, 0)
if torch.cuda.is_available():
    compute_capability = torch.cuda.get_device_capability("cuda")
sm80_or_better_only = pytest.mark.skipif(
    compute_capability < (8, 0), reason="requires sm90+"
)
sm90_or_better_only = pytest.mark.skipif(
    compute_capability < (9, 0), reason="requires sm90+"
)


# This temporary working is necessary because the MTIA test collection might not happen
# on the same device as the device the tests are actually executed on. If test collection
# is done on a device without MTIA, the supported masks will contain masks that MTIA support
# and the corresponding tests will get collected. But when it comes time to actually run the
# tests, the mask won't be supported because it is run on an actual MTIA device.
def get_supported_attn_bias_types(op):
    supported_attn_bias_types = op.SUPPORTED_ATTN_BIAS_TYPES

    try:
        import mtia.host_runtime.torch_mtia.dynamic_library  # noqa

        supported_attn_bias_types = [
            b
            for b in supported_attn_bias_types
            if not issubclass(
                b,
                (
                    fmha.attn_bias.PagedBlockDiagonalGappyKeysMask,
                    fmha.attn_bias.PagedBlockDiagonalPaddedKeysMask,
                ),
            )
        ]
    except (ImportError, OSError):
        pass

    return supported_attn_bias_types


@disable_on_rocm
@sm80_or_better_only
@pytest.mark.parametrize(
    "op",
    [
        fmha.triton_splitk.FwOp,
        fmha.flash.FwOp,
        fmha.flash3.FwOp,
        None,
    ],
    ids=lambda op: "None" if op is None else op.NAME,
)
@pytest.mark.parametrize("G,H", [(1, 11), (7, 1), (1, 1), (7, 11), (None, 11)])
@pytest.mark.parametrize(
    "write_lse", (False, True), ids=lambda x: "write_lse" if x else ""
)
@pytest.mark.parametrize(
    "stack_inputs", (False, True), ids=lambda x: "stack_inputs" if x else ""
)
def test_merge_attentions_nobias(
    write_lse: bool,
    stack_inputs: bool,
    op: Type[AttentionFwOpBase],
    G: Optional[int],
    H: int,
):
    """
    Merging the same attention twice shouldn't change anything.
    This also tests the shape of the lse output of each permitted op.
    """
    if op is fmha.flash3.FwOp and not op.is_available():
        pytest.skip("Flash3 not available")
    B, Mq, K = 13, 3, 192
    if op is fmha.triton_splitk.FwOp:
        K = 128
    case_name = str((write_lse, G, H, stack_inputs)).encode("ascii")
    many_keys = hashlib.md5(case_name).digest()[0] % 2
    M = [5, 100000][many_keys]
    if op is None or torch.bfloat16 in op.SUPPORTED_DTYPES:
        dtype = torch.bfloat16
    else:
        dtype = next(iter(op.SUPPORTED_DTYPES))
    if dtype == torch.float8_e4m3fn:
        pytest.skip("float8 not supported")
    if G is None:
        q = 3 * torch.rand(B, Mq, H, K, dtype=dtype, device="cuda")
        k = (3 * torch.rand(B, M, 1, K, dtype=dtype, device="cuda")).expand(B, M, H, K)
        v = (3 * torch.rand(B, M, 1, K, dtype=dtype, device="cuda")).expand(B, M, H, K)
    else:
        q = 3 * torch.rand(B, Mq, G, H, K, dtype=dtype, device="cuda")
        k = (3 * torch.rand(B, M, G, 1, K, dtype=dtype, device="cuda")).expand(
            B, M, G, H, K
        )
        v = (3 * torch.rand(B, M, G, 1, K, dtype=dtype, device="cuda")).expand(
            B, M, G, H, K
        )
    out1, lse1 = fmha.memory_efficient_attention_partial(q, k, v, op=op)
    assert out1.shape == q.shape
    M_ceil = lse1.shape[-1]
    assert M_ceil >= Mq
    assert lse1.shape == (B, H, M_ceil) if G is None else (B, G, H, M_ceil)
    lse1 = lse1[..., :Mq]

    attn_chunks = [out1, out1]
    lse_chunks = [lse1, lse1]
    attn_chunks_ = torch.stack(attn_chunks) if stack_inputs else attn_chunks
    lse_chunks_ = torch.stack(lse_chunks) if stack_inputs else lse_chunks
    out, lse = fmha.merge_attentions(attn_chunks_, lse_chunks_, write_lse=write_lse)  # type: ignore
    assert out.shape == out1.shape
    assert_allclose(out1, out, rtol=1e-3, atol=1e-3, msg="out")
    if write_lse:
        assert lse is not None
        assert lse.shape[:-1] == lse1.shape[:-1]
        assert_allclose(
            lse1[..., :Mq] + math.log(2), lse[..., :Mq], rtol=1e-3, atol=1e-3, msg="lse"
        )
    else:
        assert lse is None


@disable_on_rocm
@sm80_or_better_only
@pytest.mark.parametrize(
    "dtype,op",
    [
        (torch.bfloat16, fmha.triton_splitk.FwOp_S1),
        # Cutlass's LSE is not consistent
        # (torch.float32, fmha.cutlass.FwOp),
        (torch.bfloat16, fmha.flash.FwOp),
    ],
    ids=lambda o: f"{o.NAME}" if hasattr(o, "NAME") else str(o),
)
@pytest.mark.parametrize("num_queries", [1])
@pytest.mark.parametrize("bmghk", [True, False], ids=lambda x: "bmghk" if x else "")
def test_partial_paged(
    dtype: torch.dtype, op: Type[AttentionFwOpBase], num_queries: int, bmghk: bool
):
    B = 128
    N_H_L = 8
    D_H = 128
    page_size = 256
    G = 2 if bmghk else 1
    block_tables = torch.zeros((B, 1), dtype=torch.int32, device="cuda")
    torch.manual_seed(1)
    output_dtype = torch.float32 if op.SUPPORTS_OUTPUT_DTYPE else None

    B_T = num_queries * B

    q = torch.randn((1, B_T, G, N_H_L, D_H), dtype=dtype, device="cuda")
    k = torch.randn((1, page_size, G, 1, D_H), dtype=dtype, device="cuda")
    v = torch.randn_like(k)
    k = k.expand(1, page_size, G, N_H_L, D_H)
    v = v.expand(1, page_size, G, N_H_L, D_H)
    if not bmghk:
        q = q[:, :, 0]
        k = k[:, :, 0]
        v = v[:, :, 0]

    attn_bias = (
        fmha.attn_bias.PagedBlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
            q_seqlen=[num_queries] * B,
            kv_seqlen=[1] + ([100] * (B - 1)),
            page_size=page_size,
            block_tables=block_tables,
        )
    )

    if attn_bias not in get_supported_attn_bias_types(op):
        pytest.skip("Not supported bias")

    attn_chunk, lse_chunk = fmha.memory_efficient_attention_partial(
        q,
        k,
        v,
        attn_bias,
        op=op,
        output_dtype=output_dtype,
    )
    if bmghk:
        assert attn_chunk.shape == (1, B_T, G, N_H_L, D_H)
        assert lse_chunk.shape == (
            1,
            G,
            N_H_L,
            B_T,
        ), f"{lse_chunk.shape=}, {(1, G, N_H_L, B_T)=}"
    else:
        assert attn_chunk.shape == (1, B_T, N_H_L, D_H)
        assert lse_chunk.shape == (
            1,
            N_H_L,
            B_T,
        ), f"{lse_chunk.shape=}, {(1, N_H_L, B_T)=}"


@disable_on_rocm
@sm80_or_better_only
@pytest.mark.parametrize(
    "dtype,op",
    [
        (torch.bfloat16, fmha.triton_splitk.FwOp_S1),
        (torch.bfloat16, fmha.triton_splitk.FwOp_S32),
        # Cutlass's LSE is not consistent
        # (torch.float32, fmha.cutlass.FwOp),
        (torch.bfloat16, fmha.flash.FwOp),
    ],
    ids=lambda o: f"{o.NAME}" if hasattr(o, "NAME") else str(o),
)
@pytest.mark.parametrize("num_queries", [1, 2])
@pytest.mark.parametrize("bmghk", [True, False], ids=lambda x: "bmghk" if x else "")
@pytest.mark.parametrize(
    "stack_inputs", (False, True), ids=lambda x: "stack_inputs" if x else ""
)
def test_merge_attentions_decoding(
    dtype: torch.dtype,
    op: Type[AttentionFwOpBase],
    num_queries: int,
    bmghk: bool,
    stack_inputs: bool,
):
    """
    Compute decoding attention on chunks of K/V and merge them together.
    Compare with computing attention on the whole K/V.
    """
    MAX_T = 8192
    B = 128
    N_H_L = 8
    D_H = 128
    G = 2 if bmghk else 1
    torch.manual_seed(1)
    output_dtype = torch.float32 if op.SUPPORTS_OUTPUT_DTYPE else None

    num_chunks = 10

    chunk_starts = sorted(
        torch.randint(low=1, high=MAX_T // 2, size=(num_chunks,)).tolist()
    )
    chunk_starts[0] = 0
    chunk_starts.append(MAX_T)

    # We construct sequences so that even the last chunk has a non-empty part of every sequence
    # as long as the number of queries.
    # Otherwise the corresponding LSE will be -inf and that'll propagate to the whole sum.
    # It is possible to teach the kernel to ignore infinite LSEs, but in practical use cases
    # of merging attention, e.g. a batch of sequences with a common prefix, this condition should be satisfied.
    k_lens = torch.randint(
        low=chunk_starts[-2] + num_queries, high=MAX_T, size=(B,)
    ).tolist()
    q_lens = [num_queries] * B
    B_T = num_queries * B

    q = torch.randn((1, B_T, G, N_H_L, D_H), dtype=dtype, device="cuda")
    k = torch.randn((B, MAX_T, G, 1, D_H), dtype=dtype, device="cuda")
    v = torch.randn_like(k)
    if not bmghk:
        q = q[:, :, 0]

    # Compute per-chunk attention
    chunks_output: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for i in range(num_chunks):
        chunk_start, chunk_end = chunk_starts[i], chunk_starts[i + 1]
        k_chunk = k[:, chunk_start:chunk_end, ...]
        v_chunk = v[:, chunk_start:chunk_end, ...]
        axk = k_chunk.reshape(-1, G, 1, D_H).expand(1, -1, G, N_H_L, D_H)
        axv = v_chunk.reshape(-1, G, 1, D_H).expand(1, -1, G, N_H_L, D_H)
        if not bmghk:
            axk = axk[:, :, 0]
            axv = axv[:, :, 0]

        bias_type = fmha.attn_bias.BlockDiagonalPaddedKeysMask
        if i + 1 == num_chunks:
            bias_type = fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask
        attn_bias = bias_type.from_seqlens(
            q_seqlen=q_lens,
            kv_padding=chunk_end - chunk_start,
            kv_seqlen=[max(min(x, chunk_end) - chunk_start, 0) for x in k_lens],
        )

        attn_chunk, lse_chunk = fmha.memory_efficient_attention_partial(
            q,
            axk,
            axv,
            attn_bias,
            op=op,
            output_dtype=output_dtype,
        )
        if bmghk:
            assert attn_chunk.shape == (1, B_T, G, N_H_L, D_H)
            assert lse_chunk.shape == (1, G, N_H_L, B_T)
        else:
            assert attn_chunk.shape == (1, B_T, N_H_L, D_H)
            assert lse_chunk.shape == (1, N_H_L, B_T)
        chunks_output.append((attn_chunk, lse_chunk))

    # Merge attention from all chunks
    attn_split = [attn_chunk for attn_chunk, _ in chunks_output]
    lse_split = [lse_chunk for _, lse_chunk in chunks_output]
    if stack_inputs:
        attn_out, lse_out = fmha.merge_attentions(
            torch.stack(attn_split), torch.stack(lse_split), output_dtype=dtype
        )
    else:
        attn_out, lse_out = fmha.merge_attentions(
            attn_split, lse_split, output_dtype=dtype
        )
    assert lse_out is not None

    # Compute attention on the full K/V
    attn_bias = fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
        q_seqlen=q_lens,
        kv_padding=MAX_T,
        kv_seqlen=k_lens,
    )
    axk = k.view(1, -1, G, 1, D_H).expand(1, -1, G, N_H_L, D_H)
    axv = v.view(1, -1, G, 1, D_H).expand(1, -1, G, N_H_L, D_H)
    if not bmghk:
        axk = axk[:, :, 0]
        axv = axv[:, :, 0]
    attn_full, lse_full = fmha.memory_efficient_attention_partial(
        q,
        axk,
        axv,
        attn_bias,
        op=op,
        output_dtype=output_dtype,
    )

    assert_allclose(
        lse_out.to(lse_full.dtype), lse_full, rtol=1e-3, atol=1e-3, msg="lse"
    )
    assert_allclose(
        attn_out.to(attn_full.dtype), attn_full, rtol=1e-3, atol=1e-3, msg="out"
    )

    attn_full2 = fmha.memory_efficient_attention_forward(
        q,
        axk,
        axv,
        attn_bias,
        op=op,
        output_dtype=output_dtype,
    )
    assert_allclose(attn_full2, attn_full, rtol=1e-3, atol=1e-3, msg="out2")


@disable_on_rocm
@sm80_or_better_only
@pytest.mark.parametrize(
    "dtype,op",
    [
        (torch.bfloat16, fmha.triton_splitk.FwOp_S1),
        (torch.bfloat16, fmha.triton_splitk.FwOp_S32),
    ],
    ids=lambda o: f"{o.NAME}" if hasattr(o, "NAME") else str(o),
)
@pytest.mark.parametrize("gqa", [False, True], ids=lambda x: "gqa" if x else "")
def test_merge_attentions_sharedinput(
    dtype: torch.dtype,
    op: Type[AttentionFwOpBase],
    gqa: bool,
):
    """
    Compute decoding attention on chunks of K/V and merge them together.
    Compare with computing attention on the whole K/V.
    """
    MAX_T = 8192
    N_H_L = 16
    D_H = 128
    G = 2
    torch.manual_seed(1)
    output_dtype = torch.float32 if op.SUPPORTS_OUTPUT_DTYPE else None

    shared_length = 20
    full_lengths = [30, 35, 40]

    attn_bias = fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
        q_seqlen=[1, 1, 1],
        kv_padding=MAX_T,
        kv_seqlen=full_lengths,
    )
    attn_bias1 = fmha.attn_bias.BlockDiagonalPaddedKeysMask.from_seqlens(
        q_seqlen=[2, 1],
        kv_padding=MAX_T,
        kv_seqlen=[shared_length, 0],
    )
    attn_bias2 = fmha.attn_bias.BlockDiagonalGappyKeysMask.from_seqlens(
        q_seqlen=[1, 1, 1],
        kv_seqstarts=[shared_length, MAX_T + shared_length, 2 * MAX_T, 3 * MAX_T],
        kv_seqlen=[
            full_lengths[0] - shared_length,
            full_lengths[1] - shared_length,
            full_lengths[2],
        ],
    )

    q = torch.randn((1, 3, G, N_H_L, D_H), dtype=dtype, device="cuda")
    k = torch.randn((3, MAX_T, G, 1 if gqa else N_H_L, D_H), dtype=dtype, device="cuda")
    v = torch.randn_like(k)
    k[1, :shared_length] = k[0, :shared_length]
    v[1, :shared_length] = v[0, :shared_length]
    k = k.flatten(end_dim=1)[None]
    v = v.flatten(end_dim=1)[None]
    k = k.expand((1, 3 * MAX_T, G, N_H_L, D_H))
    v = v.expand((1, 3 * MAX_T, G, N_H_L, D_H))

    attn_chunk1, lse_chunk1 = fmha.memory_efficient_attention_partial(
        q,
        k,
        v,
        attn_bias1,
        op=op,
        output_dtype=output_dtype,
    )
    assert attn_chunk1.shape == (1, 3, G, N_H_L, D_H)
    assert lse_chunk1.shape == (1, G, N_H_L, 3)
    if gqa:
        attn_chunk1a, lse_chunk1a = fmha.memory_efficient_attention_partial(
            q,
            k.contiguous(),
            v,
            attn_bias1,
            op=op,
            output_dtype=output_dtype,
        )
        assert attn_chunk1a.shape == (1, 3, G, N_H_L, D_H)
        assert lse_chunk1a.shape == (1, G, N_H_L, 3)
        assert_allclose(
            attn_chunk1a.nan_to_num(0, 0, 0), attn_chunk1.nan_to_num(0, 0, 0)
        )
        assert_allclose(lse_chunk1a.nan_to_num(0, 0, 0), lse_chunk1.nan_to_num(0, 0, 0))

    attn_chunk2, lse_chunk2 = fmha.memory_efficient_attention_partial(
        q,
        k,
        v,
        attn_bias2,
        op=op,
        output_dtype=output_dtype,
    )
    assert attn_chunk2.shape == (1, 3, G, N_H_L, D_H)
    assert lse_chunk2.shape == (1, G, N_H_L, 3)
    # Merge attention from all chunks

    attn_out, lse_out = fmha.merge_attentions(
        [attn_chunk1, attn_chunk2],
        [lse_chunk1, lse_chunk2],
        output_dtype=dtype,  # type: ignore
    )
    assert lse_out is not None

    # Compute attention on the full K/V
    attn_full, lse_full = fmha.memory_efficient_attention_partial(
        q,
        k,
        v,
        attn_bias,
        op=op,
        output_dtype=output_dtype,
    )
    assert_allclose(
        attn_out.to(attn_full.dtype), attn_full, rtol=1e-2, atol=2e-3, msg="out"
    )
    assert_allclose(
        lse_out.to(lse_full.dtype), lse_full, rtol=1e-3, atol=1e-3, msg="lse"
    )


@sm80_or_better_only
@pytest.mark.parametrize("bmghk", (False, True))
def test_merge_attentions_against_ref(bmghk: bool):
    split_k = 16
    B = 12
    M = 137
    G = 2 if bmghk else 1
    N_H_L = 8
    D_H = 128
    dtype = torch.float32

    attn_split = torch.randn([split_k, B, M, G, N_H_L, D_H], dtype=dtype, device="cuda")
    lse_split = torch.randn([split_k, B, G, N_H_L, M], dtype=dtype, device="cuda")

    if not bmghk:
        attn_split = attn_split[:, :, :, 0]
        lse_split = lse_split[:, :, 0]

    attn_out_ref, lse_out_ref = _merge_attentions_ref(attn_split, lse_split)
    attn_out, lse_out = fmha.merge_attentions(attn_split, lse_split)

    torch.testing.assert_close(lse_out, lse_out_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(attn_out, attn_out_ref, rtol=1e-4, atol=1e-4)


def _merge_attentions_ref(attn_split, lse_split):
    """
    attn_split: [split_k, B, M, (G,) H, Kq]
    lse_split: [split_k, B, (G,) H, M]
    """
    is_bmghk = len(attn_split.shape) == 6
    if not is_bmghk:
        attn_split = attn_split.unsqueeze(3)
        lse_split = lse_split.unsqueeze(2)

    lse_split = lse_split[..., None].moveaxis(4, 2)  # [split_k, B, M, G, H, 1]

    lse_max, _ = torch.max(lse_split, dim=0)  # [B, M, G, H, 1]
    sumexp_normalized = torch.exp(lse_split - lse_max)  # [split_k, B, M, G, H, 1]
    denominator = sumexp_normalized.sum(dim=0)  # [B, M, G, H, 1]
    numerator = (sumexp_normalized * attn_split).sum(dim=0)  # [B, M, G, H, K]

    attn_out = numerator / denominator  # [B, M_ceil, G, H, Kq]
    lse_out = lse_max + torch.log(denominator)
    lse_out = lse_out.squeeze(4).permute(0, 2, 3, 1)  # [B, G, H, M]

    if not is_bmghk:
        attn_out = attn_out.squeeze(2)
        lse_out = lse_out.squeeze(1)

    return attn_out, lse_out


@sm80_or_better_only
def test_merge_attention_with_compile() -> None:
    op = fmha.flash3.FwOp
    if not op.is_available():
        pytest.skip("Op is not available")
    dtype = torch.bfloat16
    B, M, H, K = 1, 256, 2, 128
    q, k, v = [
        (3 * torch.rand(B, M, H, K, dtype=dtype, device="cuda")) for _ in range(3)
    ]

    out1, lse1 = fmha.memory_efficient_attention_partial(q, k, v, op=op)

    def run_code() -> torch.Tensor:
        out, _ = fmha.merge_attentions([out1], [lse1], write_lse=True)
        return out

    out_ref = run_code()
    out_c = torch.compile(run_code, fullgraph=True)()

    assert torch.allclose(out_ref, out_c, atol=1e-2, rtol=1e-2)

    q.requires_grad_(True)
    out1, lse1 = fmha.memory_efficient_attention_partial(q, k, v, op=op)
    loss = fmha.merge_attentions([out1], [lse1])[0].sum()
    with pytest.raises(
        NotImplementedError,
        match="Backward pass is not implemented for merge_attentions",
    ):
        loss.backward()


@sm80_or_better_only
def test_merge_training():
    torch.manual_seed(1)
    B, M, H, K = 1, 50, 1, 128
    dtype = torch.bfloat16
    op = (fmha.flash3.FwOp, fmha.flash3.BwOp)
    q = 3 * torch.rand((B, M, H, K), device="cuda", dtype=dtype)
    k = 3 * torch.rand((B, M, H, K), device="cuda", dtype=dtype)
    v = 3 * torch.rand((B, M, H, K), device="cuda", dtype=dtype)
    grad_out = 3 * torch.rand((B, M, H, K), device="cuda", dtype=dtype)

    total_attention, (q_grad, k_grad, v_grad) = torch.autograd.functional.vjp(
        fmha.memory_efficient_attention, (q, k, v), grad_out
    )

    def total_attn_via_Partial(q_, k_, v_):
        return merge_attentions_autograd(
            memory_efficient_attention_partial_autograd(q_, k_, v_)
        )

    attn, (q_grad_, k_grad_, v_grad_) = torch.autograd.functional.vjp(
        total_attn_via_Partial, (q, k, v), grad_out
    )

    assert_allclose(attn, total_attention, rtol=1e-1, atol=1e-3, msg="out")
    assert_allclose(k_grad_, k_grad, rtol=1e-2, atol=1e-3, msg="dk_")
    assert_allclose(q_grad_, q_grad, rtol=1e-2, atol=1e-3, msg="dq_")
    assert_allclose(v_grad_, v_grad, rtol=1e-2, atol=1e-3, msg="dv_")

    def attn_via_Partial(q_, k_, v_):
        split = M // 2
        k1 = k_[:, :split]
        k2 = k_[:, split:]
        v1 = v_[:, :split]
        v2 = v_[:, split:]

        partial1 = memory_efficient_attention_partial_autograd(q_, k1, v1, op=op)
        partial2 = memory_efficient_attention_partial_autograd(q_, k2, v2, op=op)
        return merge_attentions_autograd(partial1, partial2)

    merged, (q_grad_, k_grad_, v_grad_) = torch.autograd.functional.vjp(
        attn_via_Partial, (q, k, v), grad_out
    )
    assert_allclose(
        merged.to(total_attention.dtype),
        total_attention,
        rtol=1e-1,
        atol=1e-3,
        msg="out",
    )
    assert_allclose(k_grad_, k_grad, rtol=0.1, atol=0.9, msg="dk")
    assert_allclose(q_grad_, q_grad, rtol=0.1, atol=0.9, msg="dq")
    assert_allclose(v_grad_, v_grad, rtol=0.1, atol=0.9, msg="dv")


@sm80_or_better_only
def test_merge_training_compile():
    torch.manual_seed(1)
    B, M, H, K = 1, 50, 1, 128
    dtype = torch.bfloat16
    q = 3 * torch.rand((B, M, H, K), device="cuda", dtype=dtype)
    k = 3 * torch.rand((B, M, H, K), device="cuda", dtype=dtype)
    v = 3 * torch.rand((B, M, H, K), device="cuda", dtype=dtype)

    def f(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        k1 = k[:, : M // 2]
        k2 = k[:, M // 2 :]
        v1 = v[:, : M // 2]
        v2 = v[:, M // 2 :]
        partial1 = memory_efficient_attention_partial_autograd(q, k1, v1)
        partial2 = memory_efficient_attention_partial_autograd(q, k2, v2)
        partial2 = partial2.pad(2, 3).pad(-2, -3)
        partial2 = partial2.pad(2, 3).do_slice(2, -3)
        merged = merge_attentions_autograd(partial1, partial2)
        return merged.sum()

    out, grads = torch.autograd.functional.vjp(f, (q, k, v))
    outc, gradsc = torch.autograd.functional.vjp(
        torch.compile(fullgraph=True)(f), (q, k, v)
    )
    assert_allclose(out, outc, rtol=1e-1, atol=1e-3)
    for i, (grad, gradc) in enumerate(zip(grads, gradsc)):
        assert_allclose(grad, gradc, rtol=1e-1, atol=1e-3, msg=f"grad{i}")

    def g(q: torch.Tensor, k1: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
        partial1 = memory_efficient_attention_partial_autograd(q, k1, v1)
        return merge_attentions_autograd(partial1).sum()

    out, grads = torch.autograd.functional.vjp(g, (q, k, v))
    outc, gradsc = torch.autograd.functional.vjp(
        torch.compile(fullgraph=True)(g), (q, k, v)
    )
    assert_allclose(out, outc, rtol=1e-1, atol=1e-3)
    for i, (grad, gradc) in enumerate(zip(grads, gradsc)):
        assert_allclose(grad, gradc, rtol=1e-1, atol=1e-3, msg=f"grad{i}")


@sm80_or_better_only
def test_merge_training_zilch():
    with pytest.raises(ValueError, match="No partials to merge"):
        merge_attentions_autograd()


@sm80_or_better_only
def test_merge_training_undilate():
    torch.manual_seed(1)

    def undilate(factor: int) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        For a given factor, operate on BMHK attention output as follows:
        If sequence length is M and there are H * factor heads,
        redistribute so there are H heads and each
        original sequence position is factor times inflated.
        """

        def inner(x: torch.Tensor) -> torch.Tensor:
            M = x.shape[1]
            H = x.shape[2] // factor
            return x.flatten(1, 2).unflatten(1, (M * factor, H))

        return inner

    B, M, F, H, K = 1, 2, 3, 5, 128
    dtype = torch.bfloat16
    q = 3 * torch.rand((B, M, F * H, K), device="cuda", dtype=dtype)
    k = 3 * torch.rand((B, M, F * H, K), device="cuda", dtype=dtype)
    v = 3 * torch.rand((B, M, F * H, K), device="cuda", dtype=dtype)

    bias = fmha.BlockDiagonalMask.from_seqlens([1] * M)
    partial = memory_efficient_attention_partial_autograd(q, k, v, bias)

    q = q.reshape(B, M * F, H, K)
    k = k.reshape(B, M * F, H, K)
    v = v.reshape(B, M * F, H, K)

    bias = fmha.BlockDiagonalMask.from_seqlens([1] * (M * F))
    expected = memory_efficient_attention_partial_autograd(q, k, v, bias)
    undilated = partial.apply(undilate(F))

    assert_allclose(undilated._attn, expected._attn)
    assert_allclose(undilated._lse, expected._lse)
