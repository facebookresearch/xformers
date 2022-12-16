// This file was modified from sputnik to implement batch support for
// sparse softmax directly in the kernels
//
// Copyright 2020 The Sputnik Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cmath>

#include "sputnik/cuda_utils.h"
#include "sputnik/load_store.h"

#include <ATen/ATen.h>
#include <torch/types.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace sputnik {

namespace {

__global__ void SparseSoftmaxKernel(
    int m,
    int n,
    const float* __restrict__ values,
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    float* __restrict__ output_values,
    int nnz) {
  // Calculate the index of the row that this block will process.
  int m_index = blockIdx.x * blockDim.y + threadIdx.y;
  if (m_index >= m)
    return;
  m_index = Load(row_indices + m_index);

  // Load the row offset and calculate the number of non-zeros in
  // the row.
  int row_offset = Load(row_offsets + m_index);
  int nonzeros = Load(row_offsets + m_index + 1) - row_offset;

  int batch_offset = blockIdx.y * nnz;

  // Step 1: Find the maximum value in our row.
  const float* in = values + row_offset + batch_offset;
  float max = -INFINITY;
  for (int idx = threadIdx.x; idx < nonzeros; idx += blockDim.x) {
    float x = Load(in + idx);
    max = x > max ? x : max;
  }
  for (int idx = 1; idx < blockDim.x; idx *= 2) {
    float x = __shfl_xor_sync(0xffffffff, max, idx);
    max = x > max ? x : max;
  }

  // Step 2: Compute the normalization constant. Invert the norm
  // once so we don't need to do repeated division.
  float norm = 0.0f;
  for (int idx = threadIdx.x; idx < nonzeros; idx += blockDim.x) {
    norm += expf(Load(in + idx) - max);
  }
  for (int idx = 1; idx < blockDim.x; idx *= 2) {
    norm += __shfl_xor_sync(0xffffffff, norm, idx);
  }
  norm = 1.0f / norm;

  // step 3: Normalize the exponentials of the input and store the
  // results.
  float* out = output_values + row_offset + batch_offset;
  for (int idx = threadIdx.x; idx < nonzeros; idx += blockDim.x) {
    Store(expf(Load(in + idx) - max) * norm, out + idx);
  }
}

} // namespace

cudaError_t SparseSoftmax(
    int m,
    int n,
    int nonzeros,
    const float* __restrict__ values,
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    float* __restrict__ output_values,
    cudaStream_t stream,
    int batch) {
  // NOTE: SparseSoftmaxKernel currently only supports 1 warp per row
  // of the input matrix. We launch two warps per block, with each
  // mapped to different rows to enable us to hit max occupancy.
  constexpr int kBlockWidth = 32;
  constexpr int kWarpsPerBlock = 2;
  dim3 grid_dim(std::ceil(static_cast<float>(m) / kWarpsPerBlock), batch);
  dim3 block_dim(kBlockWidth, kWarpsPerBlock);

  SparseSoftmaxKernel<<<grid_dim, block_dim, 0, stream>>>(
      m,
      n,
      values,
      row_indices,
      row_offsets,
      column_indices,
      output_values,
      nonzeros);
  return cudaGetLastError();
}

} // namespace sputnik

at::Tensor sparse_softmax_sputnik(
    int64_t m,
    int64_t n,
    const at::Tensor& row_indices,
    const at::Tensor& values,
    const at::Tensor& row_offsets,
    const at::Tensor& column_indices) {
  TORCH_CHECK(values.dim() == 2);
  TORCH_CHECK(row_indices.dim() == 1);
  TORCH_CHECK(row_offsets.dim() == 1);
  TORCH_CHECK(column_indices.dim() == 1);
  TORCH_CHECK(values.size(1) == column_indices.size(0));

  TORCH_CHECK(row_indices.is_cuda(), "row_indices must be a CUDA tensor");
  TORCH_CHECK(values.is_cuda(), "values must be a CUDA tensor");
  TORCH_CHECK(row_offsets.is_cuda(), "row_offsets must be a CUDA tensor");
  TORCH_CHECK(column_indices.is_cuda(), "column_offsets must be a CUDA tensor");

  TORCH_CHECK(
      row_indices.is_contiguous(), "row_indices must be a contiguous tensor");
  TORCH_CHECK(values.is_contiguous(), "values must be a contiguous tensor");
  TORCH_CHECK(
      row_offsets.is_contiguous(), "row_offsets must be a contiguous tensor");
  TORCH_CHECK(
      column_indices.is_contiguous(),
      "column_offsets must be a contiguous tensor");

  TORCH_CHECK(!row_indices.is_sparse(), "row_indices must be a dense tensor");
  TORCH_CHECK(!values.is_sparse(), "values must be a dense tensor");
  TORCH_CHECK(!row_offsets.is_sparse(), "row_offsets must be a dense tensor");
  TORCH_CHECK(
      !column_indices.is_sparse(), "column_offsets must be a dense tensor");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int batch = values.size(0);
  int nonzeros = column_indices.size(0);

  at::Tensor output = at::empty({batch, nonzeros}, values.options());

  AT_CUDA_CHECK(sputnik::SparseSoftmax(
      m,
      n,
      nonzeros,
      values.data_ptr<float>(),
      row_indices.data_ptr<int>(),
      row_offsets.data_ptr<int>(),
      column_indices.data_ptr<int>(),
      output.data_ptr<float>(),
      stream,
      batch));

  return output;
}

// Taken from sputnik SparseSoftmax with minor modifications
// to adapt it to perform the backward operation
__global__ void SparseSoftmaxBackwardKernel(
    int m,
    int n,
    const float* __restrict__ gradient,
    const float* __restrict__ values,
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    float* __restrict__ output_values,
    int nnz) {
  // Calculate the index of the row that this block will process.
  int m_index = blockIdx.x * blockDim.y + threadIdx.y;
  if (m_index >= m)
    return;
  m_index = sputnik::Load(row_indices + m_index);

  // Load the row offset and calculate the number of non-zeros in
  // the row.
  int row_offset = sputnik::Load(row_offsets + m_index);
  int nonzeros = sputnik::Load(row_offsets + m_index + 1) - row_offset;

  int batch_offset = blockIdx.y * nnz;

  const float* in = values + row_offset + batch_offset;
  const float* grad = gradient + row_offset + batch_offset;

  // Step 1: Compute the intermediate sum used for the gradient
  float sum = 0.0f;
  for (int idx = threadIdx.x; idx < nonzeros; idx += blockDim.x) {
    sum += sputnik::Load(in + idx) * sputnik::Load(grad + idx);
  }
  for (int idx = 1; idx < blockDim.x; idx *= 2) {
    sum += __shfl_xor_sync(0xffffffff, sum, idx);
  }

  // step 2: Compute the gradients
  float* out = output_values + row_offset + batch_offset;
  for (int idx = threadIdx.x; idx < nonzeros; idx += blockDim.x) {
    sputnik::Store(
        sputnik::Load(in + idx) * (sputnik::Load(grad + idx) - sum), out + idx);
  }
}

at::Tensor sparse_softmax_backward_sputnik(
    int64_t m,
    int64_t n,
    const at::Tensor& row_indices,
    const at::Tensor& values,
    const at::Tensor& grad,
    const at::Tensor& row_offsets,
    const at::Tensor& column_indices) {
  TORCH_CHECK(grad.dim() == 2);
  TORCH_CHECK(values.dim() == 2);
  TORCH_CHECK(row_indices.dim() == 1);
  TORCH_CHECK(row_offsets.dim() == 1);
  TORCH_CHECK(column_indices.dim() == 1);
  TORCH_CHECK(values.size(1) == column_indices.size(0));
  TORCH_CHECK(values.size(0) == grad.size(0));
  TORCH_CHECK(values.size(1) == grad.size(1));

  TORCH_CHECK(grad.is_cuda(), "grad must be a CUDA tensor");
  TORCH_CHECK(row_indices.is_cuda(), "row_indices must be a CUDA tensor");
  TORCH_CHECK(values.is_cuda(), "values must be a CUDA tensor");
  TORCH_CHECK(row_offsets.is_cuda(), "row_offsets must be a CUDA tensor");
  TORCH_CHECK(column_indices.is_cuda(), "column_offsets must be a CUDA tensor");

  TORCH_CHECK(grad.is_contiguous(), "grad must be a contiguous tensor");
  TORCH_CHECK(
      row_indices.is_contiguous(), "row_indices must be a contiguous tensor");
  TORCH_CHECK(values.is_contiguous(), "values must be a contiguous tensor");
  TORCH_CHECK(
      row_offsets.is_contiguous(), "row_offsets must be a contiguous tensor");
  TORCH_CHECK(
      column_indices.is_contiguous(),
      "column_offsets must be a contiguous tensor");

  TORCH_CHECK(!grad.is_sparse(), "grad must be a dense tensor");
  TORCH_CHECK(!row_indices.is_sparse(), "row_indices must be a dense tensor");
  TORCH_CHECK(!values.is_sparse(), "values must be a dense tensor");
  TORCH_CHECK(!row_offsets.is_sparse(), "row_offsets must be a dense tensor");
  TORCH_CHECK(
      !column_indices.is_sparse(), "column_offsets must be a dense tensor");

  TORCH_CHECK(
      values.device() == grad.device(),
      "values should be in the same device as grad");
  TORCH_CHECK(
      values.device() == row_indices.device(),
      "a should be in the same device as row_indices");
  TORCH_CHECK(
      values.device() == row_offsets.device(),
      "a should be in the same device as row_offsets");
  TORCH_CHECK(
      values.device() == column_indices.device(),
      "a should be in the same device as column_indices");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int batch = values.size(0);
  int nonzeros = column_indices.size(0);

  at::Tensor output = at::empty({batch, nonzeros}, values.options());

  // NOTE: SparseSoftmaxBackwardKernel currently only supports 1 warp per row
  // of the input matrix. We launch two warps per block, with each
  // mapped to different rows to enable us to hit max occupancy.
  constexpr int kBlockWidth = 32;
  constexpr int kWarpsPerBlock = 2;
  dim3 grid_dim(std::ceil(static_cast<float>(m) / kWarpsPerBlock), batch);
  dim3 block_dim(kBlockWidth, kWarpsPerBlock);

  SparseSoftmaxBackwardKernel<<<grid_dim, block_dim, 0, stream>>>(
      m,
      n,
      grad.data_ptr<float>(),
      values.data_ptr<float>(),
      row_indices.data_ptr<int>(),
      row_offsets.data_ptr<int>(),
      column_indices.data_ptr<int>(),
      output.data_ptr<float>(),
      nonzeros);
  AT_CUDA_CHECK(cudaGetLastError());

  return output;
}

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::sparse_softmax_sputnik"),
      TORCH_FN(sparse_softmax_sputnik));
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::sparse_softmax_backward_sputnik"),
      TORCH_FN(sparse_softmax_backward_sputnik));
}
