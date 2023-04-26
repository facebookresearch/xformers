/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <cmath>
#include <vector>

namespace {

template <typename scalar_t>
void matmul_with_sparse_mask_kernel(
    at::TensorAccessor<scalar_t, 1> output,
    at::TensorAccessor<scalar_t, 3> a,
    at::TensorAccessor<scalar_t, 3> b,
    at::TensorAccessor<int64_t, 2> idxs) {
  int64_t nnz = output.size(0);
  int64_t K = a.size(2);
  int64_t grain_size = 128; // TODO: tune this
  at::parallel_for(0, nnz, grain_size, [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; i++) {
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
  });
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

  TORCH_CHECK(!a.is_cuda(), "a must be a CPU tensor");
  TORCH_CHECK(!b.is_cuda(), "b must be a CPU tensor");
  TORCH_CHECK(!mask.is_cuda(), "mask must be a CPU tensor");

  TORCH_CHECK(!a.is_sparse(), "a must be a dense tensor");
  TORCH_CHECK(!b.is_sparse(), "b must be a dense tensor");
  TORCH_CHECK(mask.is_sparse(), "mask must be a sparse tensor");

  int64_t B = a.size(0);
  int64_t M = a.size(1);
  int64_t N = b.size(2);

  auto mask_ = mask.coalesce();
  auto idxs = mask_.indices();
  int64_t nnz = idxs.size(1);
  auto bt = b.transpose(-2, -1);

  at::Tensor res = at::empty({nnz}, a.options());

  AT_DISPATCH_FLOATING_TYPES(
      a.scalar_type(), "matmul_with_sparse_mask_kernel", [&] {
        matmul_with_sparse_mask_kernel<scalar_t>(
            res.accessor<scalar_t, 1>(),
            a.accessor<scalar_t, 3>(),
            bt.accessor<scalar_t, 3>(),
            idxs.accessor<int64_t, 2>());
      });

  auto out = at::sparse_coo_tensor(idxs, res, {B, M, N});

  return out;
}

} // namespace

TORCH_LIBRARY_IMPL(xformers, SparseCPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::matmul_with_mask"),
      TORCH_FN(matmul_with_sparse_mask));
}
