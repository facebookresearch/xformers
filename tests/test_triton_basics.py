# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import torch

_triton_available = torch.cuda.is_available()
if _triton_available:
    try:
        import triton
        import triton.language as tl

    except (ImportError, ModuleNotFoundError):
        _triton_available = False

if _triton_available:

    @triton.jit
    def k_mean_var(X, Mean, Var, stride, N, **META):
        # fmt: on
        """
        Fused layernorm kernel over a 3d tensor.
        The layer norm is applied over the last dimension.

        Compute
            y = (x - E(x))/(sqrt(var(x) + epsilon)) * gamma + beta
        """

        row = tl.program_id(0)
        cols = tl.arange(0, META["BLOCK_SIZE_N"])

        # Move to this row
        x_ptrs = X + row * stride + cols
        x = tl.load(x_ptrs, mask=cols < N, other=0.0).to(tl.float32)
        x = tl.where(cols < N, x, 0.0)

        # Compute variance
        x_mean = tl.sum(x, axis=0) / N
        x_zm = x - x_mean
        x_zm = tl.where(cols < N, x_zm, 0.0)  # THIS SHOULD NOT BE NEEDED
        x_var = tl.sum(x_zm * x_zm, axis=0) / N
        tl.store(Mean + row, x_mean)
        tl.store(Var + row, x_var)

    def mean_var(x: torch.Tensor):
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
        k_mean_var[(M,)](
            x_arg, mean, var,
            x_arg.stride(0),
            N,
            num_warps=num_warps,
            BLOCK_SIZE_N=BLOCK_SIZE_N
        )
        # fmt: on

        return mean.reshape(x.shape[:-1]), var.reshape(x.shape[:-1])

    @triton.jit
    def k_add(
        x_ptr,
        a_output_ptr,
        n_elements,
        **meta,
    ):
        # See https://github.com/openai/triton/issues/221
        BLOCK_SIZE = meta["BLOCK_SIZE"]

        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        a_val = tl.sum(x, axis=0)
        tl.atomic_add(a_output_ptr, a_val)

    def add(x: torch.Tensor):
        a_output = torch.zeros(1, device="cuda:0")
        assert x.is_cuda and a_output.is_cuda
        n_elements = x.shape[0]

        def grid(meta):
            return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        k_add[grid](x, a_output, n_elements, BLOCK_SIZE=128)
        return a_output

    def test_kernel_add():
        size = 256
        x = torch.rand(size, device="cuda")
        output_triton = add(x)
        assert torch.allclose(output_triton, x.sum())

    def test_kernel_mean():
        torch.random.manual_seed(0)
        a = torch.rand((4, 2048, 384), device=torch.device("cuda"))

        mean, var = mean_var(a)
        t_mean = torch.mean(a, dim=-1)
        t_var = torch.var(a, dim=-1)

        print(mean)
        print(t_mean)
        print(var)
        print(t_var)

        assert torch.allclose(mean, t_mean, rtol=1e-1)
        assert torch.allclose(var, t_var, rtol=1e-1)
