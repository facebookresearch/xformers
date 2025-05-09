/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ATen.h>
#include <torch/types.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

#define CUDA_1D_KERNEL_LOOP(i, n)                                \
  for (int i = (blockIdx.x * blockDim.x) + threadIdx.x; i < (n); \
       i += (blockDim.x * gridDim.x))

template <typename integer>
constexpr __host__ __device__ inline integer ceil_div(integer n, integer m) {
  return (n + m - 1) / m;
}

template <typename scalar_t>
__global__ void matmul_with_sparse_mask_kernel(
    at::PackedTensorAccessor<scalar_t, 1> output,
    at::PackedTensorAccessor<scalar_t, 3> a,
    at::PackedTensorAccessor<scalar_t, 3> b,
    at::PackedTensorAccessor<int64_t, 2> idxs) {
  int64_t nnz = output.size(0);
  int64_t K = a.size(2);
  CUDA_1D_KERNEL_LOOP(i, nnz) {
    auto i1 = idxs[0][i];
    auto i2 = idxs[1][i];
    auto i3 = idxs[2][i];
    auto aar = a[i1][i2];
    auto bar = b[i1][i3];
    scalar_t r = 0;
    for (int64_t k = 0; k < K; k++) {
      r += aar[k] * bar[k];
    }
    output[i] = r;
  }
}

at::Tensor matmul_with_sparse_mask(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& mask) {
  TORCH_CHECK(a.dim() == b.dim());
  TORCH_CHECK(a.dim() == mask.dim());
  TORCH_CHECK(a.dim() == 3);
  TORCH_CHECK(a.size(2) == b.size(1));
  TORCH_CHECK(a.size(0) == b.size(0));
  TORCH_CHECK(a.size(1) == mask.size(1));
  TORCH_CHECK(b.size(2) == mask.size(2));
  TORCH_CHECK(a.size(0) == mask.size(0));

  TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
  TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
  TORCH_CHECK(mask.is_cuda(), "mask must be a CUDA tensor");

  TORCH_CHECK(!a.is_sparse(), "a must be a dense tensor");
  TORCH_CHECK(!b.is_sparse(), "b must be a dense tensor");
  TORCH_CHECK(mask.is_sparse(), "mask must be a sparse tensor");

  TORCH_CHECK(a.device() == b.device(), "a should be in the same device as b");
  TORCH_CHECK(
      a.device() == mask.device(), "a should be in the same device as mask");

  at::cuda::CUDAGuard device_guard(a.device());

  int64_t B = a.size(0);
  int64_t M = a.size(1);
  int64_t N = b.size(2);

  auto mask_ = mask.coalesce();
  auto idxs = mask_.indices();
  int64_t nnz = idxs.size(1);
  auto bt = b.transpose(-2, -1);

  at::Tensor res = at::empty({nnz}, a.options());

  dim3 grid(
      std::min(
          ceil_div(static_cast<int64_t>(nnz), static_cast<int64_t>(512)),
          static_cast<int64_t>(4096)));
  dim3 block(512);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      a.scalar_type(), "matmul_with_sparse_mask_kernel", [&] {
        matmul_with_sparse_mask_kernel<scalar_t><<<grid, block, 0, stream>>>(
            res.packed_accessor64<scalar_t, 1>(),
            a.packed_accessor64<scalar_t, 3>(),
            bt.packed_accessor64<scalar_t, 3>(),
            idxs.packed_accessor64<int64_t, 2>());
      });

  AT_CUDA_CHECK(cudaGetLastError());
  auto out = at::sparse_coo_tensor(idxs, res, {B, M, N});

  return out;
}

} // namespace

TORCH_LIBRARY_IMPL(xformers, SparseCUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::matmul_with_mask"),
      TORCH_FN(matmul_with_sparse_mask));
}
