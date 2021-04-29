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
