/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ATen.h>
#include <torch/types.h>

namespace {

void SparseSoftmax(
    int m,
    int n,
    int nonzeros,
    const float* values,
    const int* row_indices,
    const int* row_offsets,
    const int* column_indices,
    float* output_values,
    int batch_size) {
  for (int b = 0; b < batch_size; b++) {
    for (int i = 0; i < m; ++i) {
      // find the max in a row
      float max = -INFINITY;
      for (int j = row_offsets[i]; j < row_offsets[i + 1]; ++j) {
        float x = values[b * nonzeros + j];
        max = x > max ? x : max;
      }
      // compute the normalization constant
      float norm = 0.0f;
      for (int j = row_offsets[i]; j < row_offsets[i + 1]; ++j) {
        float x = values[b * nonzeros + j];
        norm += expf(x - max);
      }
      norm = 1.0f / norm;

      // step 3: Normalize the exponentials of the input and store the
      // results.
      for (int j = row_offsets[i]; j < row_offsets[i + 1]; ++j) {
        int offset = b * nonzeros + j;
        float x = values[offset];
        float res = expf(x - max) * norm;
        output_values[offset] = res;
      }
    }
  }
}

void SparseSoftmaxBackwardKernel(
    int m,
    int n,
    const float* gradient,
    const float* values,
    const int* row_indices,
    const int* row_offsets,
    const int* column_indices,
    float* output_values,
    int nonzeros,
    int batch_size) {
  for (int b = 0; b < batch_size; b++) {
    for (int i = 0; i < m; ++i) {
      // Step 1: Compute the intermediate sum used for the gradient
      float sum = 0.0f;
      for (int j = row_offsets[i]; j < row_offsets[i + 1]; ++j) {
        float x = values[b * nonzeros + j];
        float g = gradient[b * nonzeros + j];
        sum += x * g;
      }

      // step 2: Compute the gradients
      for (int j = row_offsets[i]; j < row_offsets[i + 1]; ++j) {
        float x = values[b * nonzeros + j];
        float g = gradient[b * nonzeros + j];
        float res = x * (g - sum);
        output_values[b * nonzeros + j] = res;
      }
    }
  }
}

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

  TORCH_CHECK(!row_indices.is_cuda(), "row_indices must be a CPU tensor");
  TORCH_CHECK(!values.is_cuda(), "values must be a CPU tensor");
  TORCH_CHECK(!row_offsets.is_cuda(), "row_offsets must be a CPU tensor");
  TORCH_CHECK(!column_indices.is_cuda(), "column_offsets must be a CPU tensor");

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

  int batch = values.size(0);
  int nonzeros = column_indices.size(0);

  at::Tensor output = at::empty({batch, nonzeros}, values.options());

  SparseSoftmax(
      m,
      n,
      nonzeros,
      values.data_ptr<float>(),
      row_indices.data_ptr<int>(),
      row_offsets.data_ptr<int>(),
      column_indices.data_ptr<int>(),
      output.data_ptr<float>(),
      batch);

  return output;
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

  TORCH_CHECK(!grad.is_cuda(), "grad must be a CPU tensor");
  TORCH_CHECK(!row_indices.is_cuda(), "row_indices must be a CPU tensor");
  TORCH_CHECK(!values.is_cuda(), "values must be a CPU tensor");
  TORCH_CHECK(!row_offsets.is_cuda(), "row_offsets must be a CPU tensor");
  TORCH_CHECK(!column_indices.is_cuda(), "column_offsets must be a CPU tensor");

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

  int batch = values.size(0);
  int nonzeros = column_indices.size(0);

  at::Tensor output = at::empty({batch, nonzeros}, values.options());

  SparseSoftmaxBackwardKernel(
      m,
      n,
      grad.data_ptr<float>(),
      values.data_ptr<float>(),
      row_indices.data_ptr<int>(),
      row_offsets.data_ptr<int>(),
      column_indices.data_ptr<int>(),
      output.data_ptr<float>(),
      nonzeros,
      batch);

  return output;
}

} // namespace

TORCH_LIBRARY_IMPL(xformers, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::sparse_softmax_sputnik"),
      TORCH_FN(sparse_softmax_sputnik));
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::sparse_softmax_backward_sputnik"),
      TORCH_FN(sparse_softmax_backward_sputnik));
}
