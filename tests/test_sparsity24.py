# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import functools
import random
from typing import cast, Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import xformers  # noqa: F401
import xformers.ops as xops
import xformers.ops.sp24 as sp24
from torch.sparse import to_sparse_semi_structured

from .utils import assert_allclose

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
compute_capability = (0, 0)
if torch.cuda.is_available():
    compute_capability = torch.cuda.get_device_capability("cuda")

requires_sp24 = pytest.mark.skipif(compute_capability < (8, 0), reason="requires sm80+")
requires_sp24_gemm = pytest.mark.skipif(
    compute_capability != (8, 0), reason="requires sm80"
)
requires_cusparselt = pytest.mark.skipif(
    not sp24._has_cusparseLt(), reason="requires cusparselt"
)
requires_h100_s24 = pytest.mark.skipif(
    compute_capability != (9, 0)
    or torch.version.cuda is None
    or int(torch.version.cuda.split(".")[0]) < 12,
    reason="requires sm90",
)
parametrize_dtype = pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16], ids=["f16", "bf16"]
)
parametrize_backend = pytest.mark.parametrize(
    "backend",
    (
        [sp24.BACKEND_CUTLASS, sp24.BACKEND_CUSPARSELT]
        if sp24._has_cusparseLt()
        else [sp24.BACKEND_CUTLASS]
    ),
)

atol_rtol_kw = {
    torch.float16: {
        "rtol": 2e-3,
        "atol": 1e-2,
    },
    torch.bfloat16: {
        "rtol": 1e-1,
        "atol": 1e-1,
    },
}


@cuda_only
def test_sparse24_largest_mask_2d() -> None:
    inp = torch.tensor(
        [[4, 3, 2, 1], [0, 0, 0.5, 0.5], [1, 2, 3, 4], [10, 2, -1, 5]],
        device="cuda",
        dtype=torch.float16,
    )
    out = torch.ops.xformers.sparse24_largest_mask_2d(inp)
    assert out.int().tolist() == [
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 1],
    ]


@requires_sp24_gemm
@parametrize_dtype
@parametrize_backend
def test_autocast(dtype, backend: str) -> None:
    N = 128
    inp = torch.randn([N, N], dtype=torch.float32, device="cuda")
    W = torch.randn([N, N], dtype=torch.float32, device="cuda")
    sInp = sp24.sparsify24(inp.to(dtype=dtype), backend=backend)
    y = sInp @ W.to(dtype=dtype)
    with torch.autocast("cuda", dtype=dtype):
        sInp_ac = sp24.sparsify24(inp, backend=backend)
        y_ac = sInp_ac @ W

    assert_allclose(
        sInp._sp24_to_dense(),
        sInp_ac._sp24_to_dense(),
        "sparse24",
        **atol_rtol_kw[dtype],
    )
    assert_allclose(y, y_ac, "gemm", **atol_rtol_kw[dtype])


@requires_sp24_gemm
@parametrize_dtype
def test_sparse24_causal1122(dtype) -> None:
    inp = torch.tensor(
        [[4, 3, 2, 1], [-1, -3, 0.6, 0.5], [1, 2, 3, 4], [10, 2, -1, 5]],
        device="cuda",
        dtype=dtype,
    )
    inp = F.pad(inp, (0, 128 - 4, 0, 128 - 4), "constant", 1)
    sInp = sp24.sparsify24(inp, algo="causal1122")

    mask = sInp._sp24_to_dense() / inp
    assert mask[:4, :4].int().tolist() == [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 1],
    ]


@requires_sp24_gemm
@parametrize_dtype
@parametrize_backend
def test_sparse24_largest_abs_values_greedy(dtype, backend) -> None:
    inp = torch.tensor(
        [[4, 3, 2, 1], [-1, -3, 0.6, 0.5], [1, 2, 3, 4], [10, 2, -1, 5]],
        device="cuda",
        dtype=dtype,
    )
    inp = F.pad(inp, (0, 128 - 4, 0, 128 - 4), "constant", 1)
    sInp = sp24.sparsify24(inp, algo="largest_abs_values_greedy", backend=backend)

    mask = sInp._sp24_to_dense() / inp
    assert mask[:4, :4].int().tolist() == [
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 1],
    ]


@cuda_only
@parametrize_dtype
def test_sparse24_largest_mask_2d_notaligned(dtype) -> None:
    inp = torch.randn([5, 5], device="cuda", dtype=dtype)
    with pytest.raises(RuntimeError):
        torch.ops.xformers.sparse24_largest_mask_2d(inp)


@cuda_only
@parametrize_dtype
def test_sparse24_largest_mask_2d_big(dtype) -> None:
    inp = torch.randn([2048, 2048], device="cuda", dtype=dtype)
    torch.ops.xformers.sparse24_largest_mask_2d(inp)


def create_random_mask(shape) -> torch.Tensor:
    r = random.Random(0)
    mask = torch.zeros(shape, dtype=torch.bool)
    for line in range(mask.shape[0]):
        for col in range(0, mask.shape[1], 4):
            sparsity = r.choice(
                [
                    [False, False, True, True],
                    [False, True, False, True],
                    [True, False, False, True],
                    [False, True, True, False],
                    [True, False, True, False],
                    [True, True, False, False],
                ]
            )
            mask[line, col : col + 4] = torch.tensor(sparsity, dtype=torch.bool)
    return mask


@cuda_only
def test_detach_requires_grad() -> None:
    x = torch.randn([128, 64], device="cuda", dtype=torch.float16, requires_grad=True)
    xs = sp24.sparsify24(x)
    assert xs.requires_grad

    # `detach` behavior
    xs2 = xs.detach()
    assert not xs2.requires_grad
    assert not (xs2 * 2).requires_grad

    xs2.requires_grad_(True)
    assert xs2.requires_grad
    ys = xs2 * 2
    assert ys.requires_grad
    ys.backward(ys)


@cuda_only
def test_detach2() -> None:
    x = torch.randn([128, 64], device="cuda", dtype=torch.float16, requires_grad=False)
    assert not sp24.sparsify24(x).requires_grad
    x.requires_grad_(True)
    xs = sp24.sparsify24(x)
    assert xs.requires_grad
    xs2 = xs.detach()
    xs2.requires_grad_(True)
    xs3 = xs2 * 2
    assert xs3.requires_grad
    xs3.backward(xs3)
    assert xs2.grad is not None
    assert x.grad is None


@cuda_only
def test_meta_pack_and_reorder() -> None:
    mask = create_random_mask([32, 64])
    # Test a specific line with a known pattern
    line = 3
    mask[line, :16] = torch.tensor(
        [
            False,
            True,
            True,
            False,  # 1 << 0 | 2 << 2
            True,
            True,
            False,
            False,  # 0 << 4 | 1 << 6
            True,
            False,
            False,
            True,  # 0 << 8 | 3 << 10
            False,
            True,
            True,
            False,  # 1 << 12 | 2 << 14
        ],
        dtype=torch.bool,
    )
    packed = torch.ops.xformers._sparse24_pack_mask(mask)
    assert packed.shape == (mask.shape[0], mask.shape[1] // 16)
    # cast int16 -> uint16
    value_packed = (packed[line, 0].item() + (1 << 16)) % (1 << 16)
    expected_value = (
        1 << 0 | 2 << 2 | 0 << 4 | 1 << 6 | 0 << 8 | 3 << 10 | 1 << 12 | 2 << 14
    )
    assert value_packed == expected_value

    meta_reordered = torch.ops.xformers._sparse24_reorder_meta(packed)
    assert meta_reordered.ndim == 3
    assert meta_reordered.shape[0] == packed.shape[0]

    assert (meta_reordered[0, 0, 0] == packed[0, 0]).item()
    assert (meta_reordered[0, 1, 0] == packed[8, 0]).item()
    assert (meta_reordered[1, 0, 0] == packed[0, 1]).item()
    assert (meta_reordered[1, 1, 0] == packed[8, 1]).item()
    assert (meta_reordered[2, 0, 0] == packed[16, 0]).item()
    assert (meta_reordered[2, 1, 0] == packed[24, 0]).item()
    # second column
    assert (meta_reordered[0, 0, 1] == packed[0, 2]).item()


@cuda_only
def test_pack_tensor_according_to_mask() -> None:
    mask = create_random_mask([32, 64])
    # Test a specific line with a known pattern
    line = 3
    line_pattern = [
        False,
        True,
        True,
        False,
        True,
        True,
        False,
        False,
        True,
        False,
        False,
        True,
        False,
        True,
        True,
        False,
    ]
    mask[line, :16] = torch.tensor(line_pattern, dtype=torch.bool)
    packed = torch.ops.xformers._sparse24_pack_mask(mask)
    reordered = torch.ops.xformers._sparse24_reorder_meta(packed)

    a_full = torch.randn(mask.shape, dtype=torch.float16)
    a_packed = torch.ops.xformers._sparse24_pack_tensor_according_to_mask(
        a_full, reordered
    )
    line_full = a_full[line, :16].tolist()
    line_packed = a_packed[line, :8].tolist()
    line_packed_expected = [
        value for index, value in enumerate(line_full) if line_pattern[index]
    ]
    assert line_packed == line_packed_expected


@requires_sp24_gemm
@parametrize_dtype
def test_sp24_gemm(dtype) -> None:
    M, N, K = 32, 32, 64
    a = torch.randn([M, K], device="cuda", dtype=dtype)
    b = torch.randn([K, N], device="cuda", dtype=dtype)
    mask = create_random_mask([M, K])
    mask_packed = torch.ops.xformers._sparse24_pack_mask(mask)
    mask_reordered = torch.ops.xformers._sparse24_reorder_meta(mask_packed)
    packed_a = torch.ops.xformers._sparse24_pack_tensor_according_to_mask(
        a.cpu(), mask_reordered
    )
    packed_a = packed_a.cuda()

    mask_reordered = mask_reordered.cuda()
    mask = mask.to(dtype).cuda()
    masked_a = a * mask

    ref_out = masked_a @ b
    sp24_out = torch.ops.xformers._sparse24_gemm(packed_a, b, mask_reordered)
    assert_allclose(ref_out, sp24_out, msg="sp24 GEMM", **atol_rtol_kw[dtype])


@cuda_only
@pytest.mark.parametrize("transpose", [True, False])
def test_pack_meta_shuffle(transpose: bool) -> None:
    local_meta = torch.zeros([4, 8, 8], dtype=torch.int64, device="cuda")
    local_meta[:2, :2] = torch.randint(
        0, 256, size=(2, 2, 8), dtype=torch.int64, device="cuda"
    )
    final_meta_tensor = torch.ops.xformers._sparse24_meta_shuffle_test(
        local_meta, transpose
    )
    assert final_meta_tensor[2:, 2:].abs().max().item() == 0
    final_meta = final_meta_tensor.tolist()

    def pack(line):
        if transpose:
            return int(
                local_meta[0, 0, line]
                | (local_meta[1, 0, line] << 8)
                | (local_meta[0, 1, line] << 16)
                | (local_meta[1, 1, line] << 24)
            )
        else:
            return int(
                local_meta[0, 0, line]
                | (local_meta[0, 1, line] << 8)
                | (local_meta[1, 0, line] << 16)
                | (local_meta[1, 1, line] << 24)
            )

    def meta_str(m):
        return " ".join(f"0x{mm:02x}" for mm in m.tolist())

    def expect_match(i, j, line):
        value = final_meta[i][j][0]
        expected = pack(line)
        assert (
            value == expected
        ), f"""value: 0x{value:02x} (expected: 0x{expected:02x})
{meta_str(local_meta[0, 0, :4])} (T0) |||| {meta_str(local_meta[0, 1, :4])} (T4)
{meta_str(local_meta[1, 0, :4])} (T1) |||| {meta_str(local_meta[1, 1, :4])} (T5)
"""

    expect_match(0, 0, 0)  # T0
    if transpose:
        expect_match(1, 0, 1)  # T1
        expect_match(0, 1, 2)  # T4
    else:
        expect_match(0, 1, 1)  # T4
        expect_match(1, 0, 2)  # T1
    expect_match(1, 1, 3)  # T5


@requires_sp24_gemm
@parametrize_dtype
@parametrize_backend
def test_pack_both_ways_meta_correctness(dtype, backend) -> None:
    M, N = 256, 512
    # Construct x to make sure we always have exactly 8 elements per 4x4 tile
    a = _gen_24_sparsifiable_both_ways(M, N, dtype)
    assert a.shape == (M, N)
    a = a.cuda().to(dtype)
    b = torch.randn([a.shape[1], 128], device="cuda", dtype=dtype)
    a_sparse = sp24.sparsify24(a, backend=backend)

    mask_dense = torch.ops.xformers.sparse24_largest_mask_2d(a)

    if backend == sp24.BACKEND_CUTLASS:
        assert isinstance(a_sparse, sp24.Sparse24TensorCutlass)
        mask_packed = torch.ops.xformers._sparse24_pack_mask(mask_dense.cpu().bool())
        mask_reordered = torch.ops.xformers._sparse24_reorder_meta(mask_packed).cuda()
        assert torch.allclose(a_sparse.meta.view(torch.short), mask_reordered)
    ref_gemm = (mask_dense * a) @ b
    pack_gemm = a_sparse @ b
    renorm = ref_gemm.std().item()
    assert_allclose(
        ref_gemm.float() / renorm,
        pack_gemm.float() / renorm,
        msg="sp24 GEMM",
        **atol_rtol_kw[dtype],
    )


@requires_sp24_gemm
@parametrize_dtype
def test_pack_both_ways_id(dtype) -> None:
    N = 512
    torch.manual_seed(0)
    a = torch.randn([N, N], dtype=dtype, device="cuda")
    b = torch.eye(N, dtype=dtype, device="cuda")

    packed, meta, packed_t, meta_t = torch.ops.xformers.sparse24_sparsify_both_ways(a)[
        :4
    ]
    # Heuristic to ensure we pack the same values
    assert torch.allclose(
        packed.to(torch.float64).sum(), packed_t.to(torch.float64).sum()
    )

    mask_dense = torch.ops.xformers.sparse24_largest_mask_2d(a.to(dtype))

    ref_gemm = mask_dense * a
    # Test A@B
    pack_gemm = torch.ops.xformers._sparse24_gemm(packed, b, meta)
    max_diff = (ref_gemm - pack_gemm).abs().argmax()
    assert torch.allclose(
        ref_gemm, pack_gemm
    ), f"packed is wrong at pos: ({max_diff // N}, {max_diff % N})"
    # Test A.t@B
    pack_gemm = torch.ops.xformers._sparse24_gemm(packed_t, b, meta_t)
    pack_gemm = pack_gemm.transpose(0, 1)
    max_diff = (ref_gemm - pack_gemm).abs().argmax()
    assert torch.allclose(
        ref_gemm, pack_gemm
    ), f"packed_t is wrong at pos: ({max_diff // N}, {max_diff % N})"


@cuda_only
@parametrize_dtype
def test_pack_both_ways_edge_case1(dtype) -> None:
    # In this case, the heuristic will keep 7 values out of 16
    # instead of 8. let's see how the kernel handles this
    quad = torch.tensor(
        [
            [2, -1, -2, -3],  # Should be packed as `2 <null>`
            [-1, 8, -1, 6],
            [-1, -1, 4, 5],
            [-1, 3, 7, -1],
        ],
        dtype=dtype,
        device="cuda",
    )
    a = torch.randn([32, 64], dtype=dtype, device="cuda")
    a[:4, :4] = quad
    packed, meta, packed_t, meta_t = torch.ops.xformers.sparse24_sparsify_both_ways(a)[
        :4
    ]
    # Check first line in A
    assert packed[0, 0].item() == 2
    assert packed[0, 1].item() == 0
    # And first column in A.t
    assert packed_t[0, 0].item() == 2
    assert packed_t[0, 1].item() == 0


@cuda_only
@parametrize_dtype
def test_sp24_apply(dtype) -> None:
    M, N = 256, 1024
    x = torch.randn([M, N], dtype=dtype, device="cuda")
    (
        packed,
        meta,
        packed_t,
        meta_t,
        threads_masks,
    ) = torch.ops.xformers.sparse24_sparsify_both_ways(x)
    packed2, _, packed_t2, _ = torch.ops.xformers.sparse24_apply(x, threads_masks)
    assert torch.allclose(packed, packed2)
    assert torch.allclose(packed_t, packed_t2)


@cuda_only
@parametrize_dtype
def test_sp24_api_different_pattern(dtype) -> None:
    M, N = 256, 256
    x = torch.randn([M, N], dtype=dtype, device="cuda")
    y = torch.randn([M, N], dtype=dtype, device="cuda")
    sx = sp24.sparsify24(x)
    sy = sp24.sparsify24(y)
    # Can't add with different sparsity pattern
    with pytest.raises(ValueError):
        sx + sy
    # Ok, same sparsity pattern
    assert isinstance(sx + sx, sp24.Sparse24Tensor)
    # Ok, sharing sparsity pattern of x
    sy2 = sp24.sparsify24_like(y, sx)
    assert isinstance(sx + sy2, sp24.Sparse24Tensor)


@cuda_only
@parametrize_dtype
def test_sp24_api_different_pattern_transposed(dtype) -> None:
    N = 256
    x = torch.randn([N, N], dtype=dtype, device="cuda")
    sx = sp24.sparsify24(x, backend=sp24.BACKEND_CUTLASS)
    sxt = sx.t()
    assert isinstance(sxt, sp24.Sparse24Tensor)
    # Can't add with different sparsity pattern
    with pytest.raises(ValueError):
        sx + sxt
    # But this should work
    sx + sxt.t()
    # And we should be able to sparsify with transposed pattern
    sxt2 = sp24.sparsify24_like(x.t(), sxt)
    assert torch.allclose(sxt2.packed, sxt.packed)
    assert torch.allclose(sxt2.packed_t, sxt.packed_t)


def _gen4x4(r: random.Random):
    # Create a 4x4 tile that can be 24 sparsified perfectly
    values = [
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 1],
    ]
    c1, c2 = r.sample([0, 1, 2, 3], 2)
    r1, r2 = r.sample([0, 1, 2, 3], 2)
    values[r1], values[r2] = values[r2], values[r1]
    for i in range(4):
        values[i][c1], values[i][c2] = values[i][c2], values[i][c1]
    return values


def _gen_24_sparsifiable_both_ways(
    M: int, N: int, dtype, seed: int = 0
) -> torch.Tensor:
    torch.manual_seed(0)
    r = random.Random(0)

    a = torch.zeros([M, N], device="cuda", dtype=torch.float16)
    assert M % 4 == 0 and N % 4 == 0
    for m in range(0, M, 4):
        for n in range(0, N, 4):
            a[m : m + 4, n : n + 4] = torch.tensor(
                _gen4x4(r), device="cuda", dtype=torch.float16
            )
    a = a * torch.randn_like(a).abs()
    return a


@requires_sp24_gemm
@parametrize_dtype
@parametrize_backend
def test_sp24_transpose_invariant(dtype, backend) -> None:
    M, N = 128, 256

    a = _gen_24_sparsifiable_both_ways(M, N, dtype)

    # Sparsify `a`` and `a.t()`
    a_s = sp24.sparsify24(a, backend=backend)
    a_t_s = sp24.sparsify24(a.t().contiguous(), backend=backend)

    assert_allclose(a_s.packed, a_t_s.packed_t)
    assert_allclose(a_s.meta, a_t_s.meta_t)
    assert_allclose(a_t_s.packed, a_s.packed_t)
    assert_allclose(a_t_s.meta, a_s.meta_t)

    assert_allclose(a_s._sp24_to_dense(), a_t_s.t()._sp24_to_dense())  # type: ignore
    assert_allclose(a_s.t()._sp24_to_dense(), a_t_s._sp24_to_dense())  # type: ignore

    assert_allclose(a_s._sp24_to_dense(), a)
    assert_allclose(a_t_s.t()._sp24_to_dense(), a)  # type: ignore
    assert_allclose(a_t_s._sp24_to_dense().t(), a)


@requires_cusparselt
@requires_sp24_gemm
@pytest.mark.parametrize("M", [128, 256, 512])
@pytest.mark.parametrize("N", [128, 256])
def test_cusparselt_format(M: int, N: int) -> None:
    a = _gen_24_sparsifiable_both_ways(M, N, torch.float16)
    a_s = sp24.sparsify24(a, backend="cusparselt")
    ref_a_s = to_sparse_semi_structured(a)
    ref_a_t_s = to_sparse_semi_structured(a.t().contiguous())

    assert_allclose(a_s.packed, ref_a_s.packed)
    assert_allclose(a_s.packed_t, ref_a_t_s.packed)

    assert_allclose(a_s._sp24_to_dense(), a)
    assert_allclose(a_s.t()._sp24_to_dense(), a.t())  # type: ignore[attr-defined]


@requires_sp24_gemm
@parametrize_dtype
def test_sp24_matmuls(dtype) -> None:
    M, N, K = 64, 256, 1024
    a = torch.randn([M, K], device="cuda", dtype=dtype)
    b = torch.randn([K, N], device="cuda", dtype=dtype)
    a_m = torch.ops.xformers.sparse24_largest_mask_2d(a)
    b_m = torch.ops.xformers.sparse24_largest_mask_2d(b)
    a_s = sp24.sparsify24(a)
    b_s = sp24.sparsify24(b)

    assert_allclose(a_s @ b, (a * a_m) @ b, msg="sp@dense", **atol_rtol_kw[dtype])
    assert_allclose(a @ b_s, a @ (b * b_m), msg="dense@sp", **atol_rtol_kw[dtype])
    assert_allclose(
        a @ a_s.t(), a @ (a * a_m).t(), msg="dense@sp.t", **atol_rtol_kw[dtype]
    )
    assert_allclose(
        a_s.t() @ a, (a * a_m).t() @ a, msg="sp.t@dense", **atol_rtol_kw[dtype]
    )


@requires_sp24
def test_sp24_matmuls_mat_vec() -> None:
    a = torch.randn([64, 128], device="cuda", dtype=torch.float16)
    b = torch.randn([128], device="cuda", dtype=torch.float16)
    a_m = torch.ops.xformers.sparse24_largest_mask_2d(a)
    a_s = sp24.sparsify24(a)

    with pytest.raises(NotImplementedError):
        assert_allclose(a_s @ b, (a * a_m) @ b, msg="sp@dense", **atol_rtol_kw[a.dtype])


@requires_sp24
def test_sp24_matmuls_bmm() -> None:
    a = torch.randn([64, 128], device="cuda", dtype=torch.float16)
    b = torch.randn([5, 6, 128], device="cuda", dtype=torch.float16)
    a_m = torch.ops.xformers.sparse24_largest_mask_2d(a)
    a_s = sp24.sparsify24(a)

    with pytest.raises(NotImplementedError):
        assert_allclose(a_s @ b, (a * a_m) @ b, msg="sp@dense", **atol_rtol_kw[a.dtype])


def sparsify24_dense(tensor: torch.Tensor):
    m = torch.ops.xformers.sparse24_largest_mask_2d(tensor)
    return m * tensor


@requires_sp24_gemm
@parametrize_dtype
@pytest.mark.parametrize("act", [F.gelu, F.relu])
def test_sp24_api_mlp_act24_correctness(dtype, act) -> None:
    B, in_ft, hid_ft, out_ft = 256, 2048, 6144, 2048
    torch.manual_seed(0)
    x = torch.randn([B, in_ft], dtype=dtype, device="cuda", requires_grad=True)
    w1 = (
        torch.randn([in_ft, hid_ft], dtype=dtype, device="cuda", requires_grad=False)
        * 0.01
    )
    w2 = (
        torch.randn([hid_ft, out_ft], dtype=dtype, device="cuda", requires_grad=False)
        * 0.01
    )
    grad = (
        torch.randn([B, out_ft], dtype=dtype, device="cuda", requires_grad=False) * 0.1
    )
    w1.requires_grad_(True)
    w2.requires_grad_(True)

    params_with_grads = [x, w1, w2]

    # Run baseline
    x1 = x @ w1
    x1 = sparsify24_dense(x1)
    x1 = act(x1)
    out = x1 @ w2
    out.backward(grad)

    grads_ref = [t.grad for t in params_with_grads]
    for t in params_with_grads:
        t.grad = None

    # Run with sparsity
    x1 = x @ w1
    x1 = sp24.sparsify24(x1)
    x1 = act(x1)
    out = x1 @ w2
    out.backward(grad)

    for grad_name, grad_ref, grad_calc in zip(
        ["x", "w1", "w2"], grads_ref, [t.grad for t in params_with_grads]
    ):
        assert grad_calc is not None, grad_name
        assert grad_ref is not None, grad_name
        assert_allclose(grad_calc, grad_ref, msg=grad_name, **atol_rtol_kw[dtype])


@requires_sp24_gemm
@parametrize_dtype
def test_sp24_api_swiglu_correctness(dtype) -> None:
    B, in_ft, hid_ft, out_ft = 256, 2048, 6144 // 2, 2048
    torch.manual_seed(0)
    x = torch.randn([B, in_ft], dtype=dtype, device="cuda", requires_grad=True)
    w1 = (
        torch.randn([in_ft, hid_ft], dtype=dtype, device="cuda", requires_grad=False)
        * 0.01
    )
    w2 = (
        torch.randn([in_ft, hid_ft], dtype=dtype, device="cuda", requires_grad=False)
        * 0.01
    )
    w3 = (
        torch.randn([hid_ft, out_ft], dtype=dtype, device="cuda", requires_grad=False)
        * 0.01
    )
    grad = (
        torch.randn([B, out_ft], dtype=dtype, device="cuda", requires_grad=False) * 0.1
    )
    w1.requires_grad_(True)
    w2.requires_grad_(True)
    w3.requires_grad_(True)

    params_with_grads = [x, w1, w2, w3]

    # Run baseline
    x1 = x @ w1
    x2 = x @ w2
    x1s = sparsify24_dense(F.silu(x1))
    hid = x1s * x2
    out = hid @ w3
    out.backward(grad)

    grads_ref = [t.grad for t in params_with_grads]
    for t in params_with_grads:
        t.grad = None

    # Run with sparsity
    x1 = x @ w1
    x2 = x @ w2
    x1s = sp24.sparsify24(F.silu(x1))
    hid = x1s * x2
    out = hid @ w3
    out.backward(grad)

    for grad_name, grad_ref, grad_calc in zip(
        ["x", "w1", "w2", "w3"], grads_ref, [t.grad for t in params_with_grads]
    ):
        assert grad_calc is not None, grad_name
        assert grad_ref is not None, grad_name
        assert_allclose(grad_calc, grad_ref, msg=grad_name, **atol_rtol_kw[dtype])


@requires_sp24_gemm
@parametrize_dtype
@pytest.mark.parametrize("M", [1, 8, 26, 31, 32, 48, 63])
def test_not_aligned(dtype, M):
    N, K = 64, 128
    A = torch.randn([M, K], dtype=dtype, device="cuda")
    B = torch.randn([K, N], dtype=dtype, device="cuda")
    As = sp24.sparsify24(A)
    A = As._sp24_to_dense()
    assert tuple(A.shape) == (M, K), A.shape
    assert_allclose(As @ B, A @ B, msg="not aligned", **atol_rtol_kw[dtype])


@requires_sp24_gemm
@parametrize_dtype
@parametrize_backend
@pytest.mark.parametrize("input_rowmajor", [True, False])
def test_sparsify24_like_dense(dtype, input_rowmajor, backend):
    M, N = 128, 256
    if input_rowmajor:
        x = torch.randn([M, N], dtype=dtype, device="cuda")
    else:
        x = torch.randn([N, M], dtype=dtype, device="cuda").t()
    sx = sp24.sparsify24(x.contiguous(), backend=backend)
    sx_like = sp24.sparsify24_like(x, pattern=sx, backend="dense")
    assert_allclose(
        sx_like, sx._sp24_to_dense(), msg="sp24_like", **atol_rtol_kw[dtype]
    )


@requires_sp24_gemm
@parametrize_dtype
@parametrize_backend
def test_sparsify24_weights(dtype, backend):
    x = torch.randn([128, 512], dtype=dtype, device="cuda", requires_grad=True)
    w = torch.randn([1024, 512], dtype=dtype, device="cuda", requires_grad=True)

    flat_w = w.flatten()  # FSDP-like processing
    w = flat_w.reshape(w.shape)

    sw = sp24.sparsify24(w, gradient="24dense", backend=backend)
    y = x @ sw.t()

    y.backward(y)


class LinearW24(torch.nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_shape = input.shape
        input = input.flatten(end_dim=-2)
        dim0 = input.shape[0]
        if dim0 % 8 != 0:
            # NOTE: This should be torch-compiled away
            input = F.pad(input, [0, 0, 0, -dim0 % 8])
        w_sparse = xops.sparsify24(
            self.weight,
            gradient="24dense",
            backend="cusparselt",
        )
        return F.linear(
            input,
            w_sparse,
            self.bias,
        )[
            :dim0
        ].unflatten(dim=0, sizes=input_shape[:-1])


# XXX: This is needed to avoid a CUDA internal error
# See the issue here:
# https://github.com/pytorch/pytorch/issues/113776
@functools.lru_cache()
def _workaround_cusparselt_internal_error() -> None:
    x0 = torch.randn([128, 128], device="cuda", dtype=torch.float16, requires_grad=True)
    m = LinearW24(128, 128, bias=False).cuda().to(torch.float16)
    out = m(x0)
    out.backward(out)


@requires_sp24
@parametrize_dtype
@pytest.mark.skipif(not sp24._has_cusparseLt(), reason="requires cusparselt")
@pytest.mark.parametrize("bias", [False, True], ids=["", "bias"])
@pytest.mark.parametrize("aligned", [False, True], ids=["misaligned", ""])
@pytest.mark.parametrize("amp", [False, True], ids=["", "amp"])
def test_linearw24(dtype, bias: bool, aligned: bool, amp: bool) -> None:
    _workaround_cusparselt_internal_error()

    B, ft_in, ft_out = 64, 128, 256
    if not aligned:
        B = 65
    model_dtype = torch.float32 if amp else dtype
    x = torch.randn([B, ft_in], device="cuda", dtype=model_dtype, requires_grad=True)
    grad = torch.randn([B, ft_out], device="cuda", dtype=model_dtype)
    m = torch.nn.Linear(ft_in, ft_out, bias=bias).cuda().to(model_dtype)
    m24 = LinearW24(ft_in, ft_out, bias=bias).cuda().to(model_dtype)

    with torch.autocast("cuda", dtype=dtype, enabled=amp):
        # Make weights sparse
        state_dict = m.state_dict()
        weight_sp24 = sp24.sparsify24(state_dict["weight"].abs())
        state_dict["weight"] = weight_sp24._sp24_to_dense().to(model_dtype).detach()
        m.load_state_dict(state_dict)
        m24.load_state_dict(state_dict)

        # FW with dense weights
        out = m(x)

        # FW with sparsity
        x24 = x.detach().requires_grad_()
        out24 = m24(x24)

    # Backward passes outside autocast
    out.backward(grad)
    out24.backward(grad)

    assert out24.is_contiguous()
    assert x24.grad is not None
    assert x24.grad.is_contiguous()
    assert m24.weight.grad is not None
    assert m24.weight.grad.is_contiguous()
    if bias:
        assert m24.bias.grad is not None

    assert_allclose(out24, out, msg="output", **atol_rtol_kw[dtype])
    assert x.grad is not None and x24.grad is not None
    assert_allclose(x24.grad, x.grad, msg="x.grad", **atol_rtol_kw[dtype])
    assert m.weight.grad is not None
    assert_allclose(
        m24.weight.grad.to(dtype),
        sp24.sparsify24_like(
            m.weight.grad.to(dtype), pattern=weight_sp24, out_dense=True
        ),
        msg="w.grad",
        **atol_rtol_kw[dtype],
    )
    if bias:
        assert m.bias.grad is not None
        assert m24.bias.grad is not None
        assert_allclose(
            m24.bias.grad.to(dtype),
            m.bias.grad.to(dtype),
            msg="bias.grad",
            **atol_rtol_kw[dtype],
        )


@requires_sp24
@pytest.mark.skipif(not sp24._has_cusparseLt(), reason="requires cusparselt")
def test_wrong_alignment_error_message() -> None:
    A = torch.randn([128, 128], device="cuda", dtype=torch.float16)
    B = torch.randn([128, 4], device="cuda", dtype=torch.float16)
    A = sp24.sparsify24(A, backend="cusparselt")
    with pytest.raises(NotImplementedError, match="aligned to 8"):
        A @ B


@requires_sp24
@pytest.mark.skipif(not sp24._has_cusparseLt(), reason="requires cusparselt")
def test_min_alignment() -> None:
    A = torch.randn([128, 128], device="cuda", dtype=torch.float16)
    B = torch.randn([128, 8], device="cuda", dtype=torch.float16)
    A = sp24.sparsify24(A, backend="cusparselt")
    assert_allclose(A @ B, A._sp24_to_dense() @ B, "output", **atol_rtol_kw[A.dtype])


@requires_sp24
@pytest.mark.skipif(not sp24._has_cusparseLt(), reason="requires cusparselt")
def test_wrong_dtype_error_message() -> None:
    A = torch.randn([128, 128], device="cuda", dtype=torch.float16)
    B = torch.randn([128, 16], device="cuda", dtype=torch.float32)
    A = sp24.sparsify24(A, backend="cusparselt")
    with pytest.raises(NotImplementedError, match="the same data type"):
        A @ B


@requires_sp24_gemm
@parametrize_backend
@pytest.mark.parametrize("with_bias", [False, True])
def test_linear_dispatch_inference_mode(backend: str, with_bias: bool) -> None:
    B, ft_in, ft_out = 128, 256, 512
    x = torch.randn([B, ft_in], device="cuda", dtype=torch.float16)
    weight = torch.randn([ft_out, ft_in], device="cuda", dtype=torch.float16)
    bias = (
        torch.randn([ft_out], device="cuda", dtype=torch.float16) if with_bias else None
    )

    w_sparse = sp24.sparsify24(
        weight,
        gradient="24dense",
        backend=backend,
    )
    # NOTE: When in `inference_mode`, PyTorch no longer dispatches to `addmm`, but to `linear`
    # so we need to support that as well
    with torch.inference_mode():
        # Does not support bias at the moment in CUTLASS backend
        if bias is not None and backend == sp24.BACKEND_CUTLASS:
            with pytest.raises(NotImplementedError):
                F.linear(x, w_sparse, bias)
            return
        out = F.linear(x, w_sparse, bias)
    out_ref = F.linear(x, w_sparse._sp24_to_dense(), bias)
    assert_allclose(out, out_ref, msg="output", **atol_rtol_kw[x.dtype])


@cuda_only
def test_sp24_meta() -> None:
    x = torch.randn([1024, 512], device="meta", dtype=torch.float16)
    x_s = sp24.sparsify24(x, backend="cusparselt")
    assert x_s.shape == x.shape
    x_s_t = x_s.t()
    assert x_s_t.shape == x.t().shape


@requires_sp24_gemm
@parametrize_backend
def test_sp24_compile(backend) -> None:
    x = torch.randn([1024, 512], device="cuda", dtype=torch.float16, requires_grad=True)
    e = torch.eye(x.shape[0], x.shape[0], device="cuda", dtype=torch.float16)

    def fn(x, e):
        y = sp24.sparsify24(x, backend=backend, gradient="24dense")
        y = y.t()
        return x @ y

    # Eager
    output = fn(x, e)
    output.backward(output)
    # Torch compile
    output = torch.compile(fn)(x, e)
    output.backward(output)


class _TransformerFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias: bool = True,
        linear_cls=nn.Linear,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = linear_cls(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = linear_cls(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


@requires_sp24_gemm
@pytest.mark.skipif(not sp24._has_cusparseLt(), reason="requires cusparselt")
def test_linearw24_block_compile() -> None:
    # TODO: Parametrize on `dtype` when torch.compile gets faster
    # currently takes ~5s per test
    dtype = torch.bfloat16
    B, FT_IN, FT_HIDDEN = 31, 512, 2048

    _workaround_cusparselt_internal_error()
    m = _TransformerFFN(FT_IN, FT_HIDDEN, linear_cls=LinearW24).to("cuda").to(dtype)
    m_c = _TransformerFFN(FT_IN, FT_HIDDEN, linear_cls=LinearW24).to("cuda").to(dtype)
    m_c.load_state_dict(m.state_dict())
    m_c = cast(_TransformerFFN, torch.compile(m_c))

    x, grad = [torch.randn([B, FT_IN], dtype=dtype, device="cuda") for _ in range(2)]
    x = x.requires_grad_()
    out = m(x)
    out.backward(grad)

    x_c = x.detach().requires_grad_()
    out_c = m_c(x_c)
    out_c.backward(grad)

    assert_allclose(out_c, out, "output", **atol_rtol_kw[dtype])
    assert x_c.grad is not None and x.grad is not None
    assert_allclose(x_c.grad, x.grad, "output", **atol_rtol_kw[dtype])
    for param_name, param_ref, param_c in [
        ["fc1.weight", m.fc1.weight, m_c.fc1.weight],
        ["fc1.bias", m.fc1.bias, m_c.fc1.bias],
        ["fc2.weight", m.fc2.weight, m_c.fc2.weight],
        ["fc2.bias", m.fc2.bias, m_c.fc2.bias],
    ]:
        assert param_ref.grad is not None and param_c.grad is not None, param_name
        assert_allclose(param_c.grad, param_ref.grad, param_name, **atol_rtol_kw[dtype])


@requires_sp24
@pytest.mark.skipif(not sp24._has_cusparseLt(), reason="requires cusparselt")
def test_sp24_ste():
    x = torch.randn([512, 512], dtype=torch.float16, device="cuda", requires_grad=True)
    grad = torch.randn_like(x)
    spX = sp24.sparsify24(x, gradient=sp24.GRADIENT_STE)
    spX.backward(grad)
    assert_allclose(x.grad, grad, "grad")


@requires_sp24_gemm
@parametrize_dtype
def test_sparsify24_ste(dtype):
    x = torch.randn([512, 512], dtype=dtype, device="cuda", requires_grad=True)
    y = torch.randn([512, 512], dtype=dtype, device="cuda", requires_grad=True)
    mul0 = 2.0  # (numbers that have an exact representation in f16)
    mul1 = 0.5
    spX = sp24.sparsify24_ste(x, bw_mul0=mul0, bw_mul1=mul1)
    spX.backward(y)
    spYd = sp24.sparsify24_like(y, pattern=spX)._sp24_to_dense()
    ref = mul1 * (spYd) + mul0 * (y - spYd)
    assert_allclose(x.grad, ref, "grad")


class _Sp24X(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        xs = sp24.sparsify24(x, backend="cusparselt", algo="largest_values_greedy")
        ctx.threads_masks = xs.threads_masks
        ctx.meta = xs.meta.clone()
        ctx.meta_t = xs.meta_t.clone()
        return xs

    @staticmethod
    def backward(ctx, x):
        packed, meta, packed_t, meta_t = sp24.SparsifyApply.OPERATOR(
            x, ctx.threads_masks, backend="cusparselt"
        )
        meta.copy_(ctx.meta)
        meta_t.copy_(ctx.meta_t)
        return sp24.Sparse24TensorCuSparseLt(
            x.shape,
            packed,
            meta,
            packed_t,
            meta_t,
            ctx.threads_masks,
            requires_grad=False,
        )


@requires_sp24_gemm
@pytest.mark.skipif(not sp24._has_cusparseLt(), reason="requires cusparselt")
def test_compile_unflatten():
    x = torch.randn(
        [1024, 1024], device="cuda", dtype=torch.float16, requires_grad=True
    )
    fnc = torch.compile(_Sp24X.apply)
    fnc(x)


def _to_fp8_rowwise(x: torch.Tensor, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    max_v = torch.finfo(dtype).max
    x_scale = (x.abs().max(1, keepdim=True)[0] / max_v).float()
    x = (x / x_scale).to(dtype)
    return x, x_scale


@cuda_only
@pytest.mark.parametrize("M", [4, 8])
@pytest.mark.parametrize("sort_preproc", ["largest", "largest_abs"])
def test_sparseNM_dense(M: int, sort_preproc: str) -> None:
    dtype = torch.bfloat16
    torch.manual_seed(0)
    N = 2

    A = torch.randn([128, 128], device="cuda", dtype=dtype)
    As = torch.ops.xformers.sparseNM_dense(A, N=N, M=M, sort_preproc=sort_preproc)
    Amask = A.reshape([-1, M])
    if sort_preproc == "largest_abs":
        Amask = Amask.abs()
    # NOTE: We want to know which of the 4 values will be masked out after
    # sparsity. We use 2 argsorts to compute the rank of each element
    # Example:
    # [8, 1, 2, 9] < Array values
    # [1, 2, 0, 3] < After the first `argsort`
    # [2, 0, 1, 3] < After the second `argsort`
    #  |  ^ 9 is in index 0 of the sorted array
    #  ^ 8 is in index 2 of the sorted array
    Amask = Amask.argsort().argsort()
    As_ref = A.clone().reshape([-1, M])
    As_ref[Amask < (M - N)] = 0
    As_ref = As_ref.reshape_as(A)
    # NOTE: Sometimes we have ties
    # [0, 1, 1, 2] can be sparsified as [0, 1, 0, 2] or [0, 0, 1, 2]
    # Both are valid so there is a small margin for error here
    assert (As != As_ref).mean(dtype=torch.float).item() < 0.002


@requires_h100_s24
def test_sparse24_fp8_sm90_cutlass_gemm_eye(
    M=512, K=256, dtype=torch.float8_e4m3fn
) -> None:
    torch.manual_seed(0)
    A = torch.randn([M, K], device="cuda", dtype=torch.bfloat16)
    A[A == 0] = 1
    A = torch.ops.xformers.sparseNM_dense(A, N=2, M=4, sort_preproc="largest")
    A, _ = _to_fp8_rowwise(A, dtype)

    # NOTE: CUTLASS compression kernel expects the input to be *exactly*
    # 2:4 sparse already (eg it does not select the largest values)
    A_packed, A_mdata = torch.ops.xformers._sparse24_sm90_cutlass_compress(A)
    assert torch.allclose(
        A_packed.float().sum(), A.float().sum()
    )  # Check all values are there

    # Check MM without scale
    eye = torch.eye(A.shape[1], device=A.device, dtype=A.dtype).T
    A_reconstructed = torch.ops.xformers._sparse24_fp8_sm90_cutlass_gemm(
        A_packed, A_mdata, eye
    )
    assert torch.allclose(A.float(), A_reconstructed.float())

    # Check MM with scale
    b_scale = torch.randn([1, A.shape[1]], device=eye.device, dtype=torch.float32)
    a_scale = torch.randn([A.shape[0], 1], device=eye.device, dtype=torch.float32)
    A_reconstructed = torch.ops.xformers._sparse24_fp8_sm90_cutlass_gemm(
        A_packed, A_mdata, eye, a_scale=a_scale, b_scale=b_scale
    )
    assert torch.allclose(
        A.float() * b_scale * a_scale, A_reconstructed.float(), rtol=0.01
    )


@requires_h100_s24
def test_sparse24_fp8_sm90_cutlass_gemm_random_tensor(
    M=512, N=1024, K=256, dtype=torch.float8_e4m3fn
) -> None:
    torch.manual_seed(0)
    A = torch.randn([M, K], device="cuda", dtype=torch.bfloat16)
    A[A == 0] = 1
    A = torch.ops.xformers.sparseNM_dense(A, N=2, M=4, sort_preproc="largest")
    A, a_scale = _to_fp8_rowwise(A, dtype)
    B, b_scale = _to_fp8_rowwise(
        torch.randn([N, K], device="cuda", dtype=torch.bfloat16), dtype
    )
    B = B.T
    b_scale = b_scale.T

    A_packed, A_mdata = torch.ops.xformers._sparse24_sm90_cutlass_compress(A)
    out_xformers = torch.ops.xformers._sparse24_fp8_sm90_cutlass_gemm(
        A_packed, A_mdata, B, a_scale=a_scale, b_scale=b_scale
    )
    out_ref = torch._scaled_mm(
        A, B, scale_a=a_scale, scale_b=b_scale, out_dtype=out_xformers.dtype
    )
    assert torch.allclose(out_xformers, out_ref, rtol=0.01, atol=0.01)


# end of OSS file
