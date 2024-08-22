# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import random

import pytest
import torch

from xformers import _is_triton_available
from xformers.ops.tiled_matmul import tiled_matmul

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
compute_capability = (0, 0)
if torch.cuda.is_available():
    compute_capability = torch.cuda.get_device_capability("cuda")
cuda_sm70_only = pytest.mark.skipif(
    compute_capability < (7, 0), reason="requires sm70+"
)

# We care about correctness, not performance, hence let's "disable" the
# expensive autotuning by removing all configs except one (the first one).
if _is_triton_available():
    from xformers.ops._triton.tiled_matmul_kernels import _xformers_tiled_matmul_kernel

    while len(_xformers_tiled_matmul_kernel.configs) > 1:
        _xformers_tiled_matmul_kernel.configs.pop()


def generate_test_shapes(*repeats, num_shapes=5):
    shapes = []
    r = random.Random(0)
    for repeat in repeats:
        m_num_tiles, n_num_tiles, k_num_tiles = repeat
        for _ in range(num_shapes):
            shapes.append(
                (
                    [r.randint(2, 1024 // m_num_tiles) for _ in range(m_num_tiles)],
                    [r.randint(2, 1024 // n_num_tiles) for _ in range(n_num_tiles)],
                    [r.randint(2, 1024 // k_num_tiles) for _ in range(k_num_tiles)],
                )
            )
    return shapes


_test_shapes = generate_test_shapes((1, 1, 1), (3, 3, 3))
_dtypes = [torch.float32, torch.bfloat16, torch.float16]


def ceil_of_ratio(n, k):
    return (n + k - 1) // k


def make_operands(m, n, k, *, dtype):
    """Produce lhs, rhs and reference output tensors

    To dodge numerical accuracy differences between our kernels and PyTorch's
    ones, we avoid random values and construct matrices whose product is an
    exact mathematical computation, specifically: the remainder!

    We do it by having the i-th row of lhs and the j-th column on rhs be like:
    * lhs: i times "1", followed by "0"
    * rhs: j-1 times "1", followed by "-(j-1)", then repeated
    The running sum of their pointwise product will thus be:
    1, 2, 3, ..., j-1, 0, 1, 2, 3, ... and so on
    And the final value will be remainder of i by j.

    If K is smaller than M and/or N, this function also takes care of repeating
    some rows and/or columns in order to "fill" M and/or K. Similarly, if the
    precision of the dtype is too low to store the result without losses, the
    function will only use small-enough values, and repeat them as needed.

    Finally, the function permutes the rows and columns, in order to avoid a
    predictable block structure.

    """
    max_value = min(k, int(1 / torch.finfo(dtype).eps) * 2)
    m_perm = torch.randperm(m)
    n_perm = torch.randperm(n)

    num_reps_m = ceil_of_ratio(m, max_value)
    lhs = (
        torch.ones((min(m, max_value), k), dtype=dtype)
        .tril()
        .repeat([num_reps_m, 1])[m_perm, :]
    )
    assert lhs.shape == (m, k)

    num_reps_n = ceil_of_ratio(n, max_value)
    rhs = torch.ones((k, min(n, max_value)), dtype=dtype)
    for i in range(2, min(n, max_value) + 2):
        rhs[:, i - 2][i - 1 :: i] = -i + 1
    rhs = rhs.repeat([1, num_reps_n])[:, n_perm]
    assert rhs.shape == (k, n)

    lhs_idxs = torch.arange(1, min(m, max_value) + 1).repeat([num_reps_m])[m_perm, None]
    rhs_idxs = torch.arange(2, min(n, max_value) + 2).repeat([num_reps_n])[None, n_perm]
    out = torch.remainder(lhs_idxs, rhs_idxs).to(dtype)
    assert out.shape == (m, n)

    return lhs, rhs, out


@cuda_only
@cuda_sm70_only
@pytest.mark.parametrize("shape", _test_shapes, ids=[str(x) for x in _test_shapes])
@pytest.mark.parametrize("dtype", _dtypes, ids=[str(x) for x in _dtypes])
@pytest.mark.parametrize(
    "compile", [pytest.param(False, id="eager"), pytest.param(True, id="compile")]
)
def test_forward_backward(
    shape,
    dtype,
    compile: bool,
):
    if compile and dtype is torch.bfloat16 and compute_capability < (8, 0):
        # https://fb.workplace.com/groups/1075192433118967/posts/1480158559289017
        pytest.skip("Dynamo misbehaves on V100 or earlier when handling bf16")

    m_tiles, n_tiles, k_tiles = shape
    m, n, k = sum(m_tiles), sum(n_tiles), sum(k_tiles)

    torch.manual_seed(m * n * k)
    torch._dynamo.reset_code_caches()  # avoids hitting recompilation limit

    a, b, c_reference = make_operands(m, n, k, dtype=dtype)
    a = a.cuda().requires_grad_()
    b = b.cuda().requires_grad_()
    c_reference = c_reference.cuda()

    # In one operand make each tile have its own strides, in the other use the
    # same stride for all tiles. And make the two operands have the stride==1
    # in different dimensions.
    a_tiled = [
        [y.t().clone().t() for y in x.split(k_tiles, dim=1)]
        for x in a.split(m_tiles, dim=0)
    ]
    b_tiled = [[y for y in x.split(n_tiles, dim=1)] for x in b.split(k_tiles, dim=0)]

    tiled_matmul_compiled = torch.compile(
        tiled_matmul, fullgraph=True, disable=not compile
    )

    c_test_tiled = tiled_matmul_compiled(a_tiled, b_tiled)
    c_test = torch.cat([torch.cat(x, dim=1) for x in c_test_tiled], dim=0)

    torch.testing.assert_close(c_test, c_reference)

    # To avoid numerical issues in the backward, set things up so that we only
    # multiply by a diagonal matrix whose entries are +/- 2^{-1/0/+1} (so that
    # it only changes the sign bit and the exponent).
    diag = torch.tensor(random.choices([-2, -1, -0.5, 0.5, 1, 2], k=min(m, n)))
    grad_c = torch.zeros_like(c_test)
    torch.diag(grad_c)[:] = diag
    grad_a_reference = torch.matmul(grad_c, b.detach().t())
    grad_b_reference = torch.matmul(a.detach().t(), grad_c)

    torch.autograd.backward([c_test], [grad_c], inputs=[a, b])

    torch.testing.assert_close(a.grad, grad_a_reference)
    torch.testing.assert_close(b.grad, grad_b_reference)
