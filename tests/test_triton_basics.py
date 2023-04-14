# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch

import xformers

SHAPES = [
    (384, 128),
    (8 * 384, 128),
    (34, 128),
    (16, 128),
    (16, 512),
    (8, 384),
    (8, 1024),
    (8, 2048),
    (8, 4096),
    (8, 4096),
    (4, 12288),
]


_triton_available = xformers._is_triton_available()
if _triton_available:
    try:
        import triton
        import triton.language as tl

        from xformers.triton.sum_strided import sum_2d_dim_0

    except (ImportError, ModuleNotFoundError):
        _triton_available = False

if _triton_available:

    @triton.jit
    def k_mean(X, Mean, Var, stride, N, BLOCK_SIZE_N: tl.constexpr):
        # fmt: on
        """
        Fused layernorm kernel over a 3d tensor.
        The layer norm is applied over the last dimension.

        Compute
            y = (x - E(x))/(sqrt(var(x) + epsilon)) * gamma + beta
        """

        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE_N)

        # Move to this row
        x_ptrs = X + row * stride + cols
        x = tl.load(x_ptrs, mask=cols < N, other=0.0).to(tl.float32)
        x = tl.where(cols < N, x, 0.0)

        # Compute variance
        x_mean = tl.sum(x, axis=0) / N
        x_zm = x - x_mean
        x_zm = tl.where(cols < N, x_zm, 0.0)
        x_var = tl.sum(x_zm * x_zm, axis=0) / N
        tl.store(Mean + row, x_mean)
        tl.store(Var + row, x_var)

    def stats(x: torch.Tensor):
        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape

        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE_N:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

        # heuristics for number of warps.
        num_warps = min(max(BLOCK_SIZE_N // 256, 1), 8)

        mean = torch.zeros((M,)).cuda()
        var = torch.zeros((M,)).cuda()

        # enqueue kernel
        # fmt: off
        k_mean[(M,)](
            x_arg, mean, var,
            x_arg.stride(0),
            N,
            num_warps=num_warps,
            BLOCK_SIZE_N=BLOCK_SIZE_N
        )
        # fmt: on

        return mean.reshape(x.shape[:-1]), var.reshape(x.shape[:-1])

    def test_mean():
        torch.random.manual_seed(0)
        a = torch.rand((4, 2048, 384), device=torch.device("cuda"))

        mean, var = stats(a)
        t_mean = torch.mean(a, dim=-1)
        t_var = torch.var(a, dim=-1)

        print(mean)
        print(t_mean)
        print(var)
        print(t_var)

        assert torch.allclose(mean, t_mean, rtol=1e-1)
        assert torch.allclose(var, t_var, rtol=1e-1)

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_sum_strided(shape, dtype):
        torch.random.manual_seed(0)
        a = torch.rand(shape, device=torch.device("cuda"), dtype=dtype)

        torch_sum = torch.sum(a, dim=0)
        triton_sum = sum_2d_dim_0(a)
        assert torch.allclose(
            torch_sum, triton_sum, rtol=0.01
        ), f"{torch_sum}\n{triton_sum}"

    def test_sum_strided_asserts():
        torch.random.manual_seed(0)
        a = torch.rand((128, 256), device=torch.device("cuda"), dtype=torch.float16)

        with pytest.raises(AssertionError):
            # This kernel is not useful in that case, assert to prevent misuse
            sum_2d_dim_0(a.transpose(1, 0))

        a = torch.rand((3, 128, 256), device=torch.device("cuda"), dtype=torch.float16)
        with pytest.raises(AssertionError):
            # This kernel expects 2D tensors, assert to prevent misuse
            sum_2d_dim_0(a)

    @triton.jit
    def k_rand(X, Y, SEED_X, SEED_Y, stride_x, stride_y, N: tl.constexpr):
        # fmt: on
        """
        Check the random number generation
        """

        row = tl.program_id(0)

        # Generate random numbers with seed A
        rand_offsets = tl.arange(0, N)
        seed_x = tl.load(SEED_X + row)
        randx, _, _, _ = tl.randint4x(seed_x, rand_offsets)

        rand_offsets = tl.arange(0, N)
        seed_y = tl.load(SEED_Y + row)
        randy, _, _, _ = tl.randint4x(seed_y, rand_offsets)

        # Move to this row
        tl.store(X + row * stride_x + tl.arange(0, N), randx)
        tl.store(Y + row * stride_y + tl.arange(0, N), randy)

    def test_rand():
        # Check that the random generator used in triton works fine
        torch.random.manual_seed(0)
        x = torch.zeros((512, 32), device=torch.device("cuda"), dtype=torch.int32)
        y = torch.zeros((512, 32), device=torch.device("cuda"), dtype=torch.int32)

        M, N = x.shape

        seeds_x = torch.randint(65536, (M,), device=x.device)
        seeds_y = torch.randint(65536, (M,), device=x.device)

        assert not torch.allclose(seeds_x, seeds_y)

        # enqueue kernels, one per line
        # fmt: off
        k_rand[(M,)](
            x, y,
            seeds_x, seeds_y,
            x.stride(0), y.stride(0),
            N,
        )
        # fmt: on

        assert not torch.allclose(x, y)
