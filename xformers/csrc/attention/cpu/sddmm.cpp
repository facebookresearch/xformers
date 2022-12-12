#include <ATen/ATen.h>
#include <torch/types.h>

namespace {

// taken from
// https://github.com/google-research/google-research/blob/master/sgk/sparse/ops/cc/sddmm_launcher.cc
// with modifications to add batch support
// Simple CPU kernel launcher.
void LaunchSddmm(
    int m,
    int k,
    int n,
    int nonzeros,
    const int* row_indices,
    const int* row_offsets,
    const int* column_indices,
    const float* lhs_matrix,
    const float* rhs_matrix,
    float* output_values,
    int batch_size) {
  for (int b = 0; b < batch_size; b++) {
    for (int i = 0; i < m; ++i) {
      for (int j = row_offsets[i]; j < row_offsets[i + 1]; ++j) {
        int idx_n = column_indices[j];
        float accumulator = 0.0f;
        for (int l = 0; l < k; ++l) {
          accumulator += lhs_matrix[b * m * k + i * k + l] *
              rhs_matrix[b * n * k + idx_n * k + l];
        }
        output_values[b * nonzeros + j] = accumulator;
      }
    }
  }
}

at::Tensor sddmm_sputnik(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& row_indices,
    const at::Tensor& row_offsets,
    const at::Tensor& column_indices) {
  TORCH_CHECK(a.dim() == b.dim());
  TORCH_CHECK(a.dim() == 3);
  TORCH_CHECK(a.size(0) == b.size(0));
  TORCH_CHECK(a.size(2) == b.size(2));
  TORCH_CHECK(row_indices.dim() == 1);
  TORCH_CHECK(row_offsets.dim() == 1);
  TORCH_CHECK(column_indices.dim() == 1);

  TORCH_CHECK(!a.is_cuda(), "a must be a CPU tensor");
  TORCH_CHECK(!b.is_cuda(), "b must be a CPU tensor");
  TORCH_CHECK(!row_indices.is_cuda(), "row_indices must be a CPU tensor");
  TORCH_CHECK(!row_offsets.is_cuda(), "row_offsets must be a CPU tensor");
  TORCH_CHECK(!column_indices.is_cuda(), "column_offsets must be a CPU tensor");

  TORCH_CHECK(a.is_contiguous(), "a must be a contiguous tensor");
  TORCH_CHECK(b.is_contiguous(), "b must be a contiguous tensor");
  TORCH_CHECK(
      row_indices.is_contiguous(), "row_indices must be a contiguous tensor");
  TORCH_CHECK(
      row_offsets.is_contiguous(), "row_offsets must be a contiguous tensor");
  TORCH_CHECK(
      column_indices.is_contiguous(),
      "column_offsets must be a contiguous tensor");

  TORCH_CHECK(!a.is_sparse(), "a must be a dense tensor");
  TORCH_CHECK(!b.is_sparse(), "b must be a dense tensor");
  TORCH_CHECK(!row_indices.is_sparse(), "row_indices must be a dense tensor");
  TORCH_CHECK(!row_offsets.is_sparse(), "row_offsets must be a dense tensor");
  TORCH_CHECK(
      !column_indices.is_sparse(), "column_offsets must be a dense tensor");

  int batch = a.size(0);
  int m = a.size(1);
  int k = a.size(2);
  int n = b.size(1);

  int nonzeros = column_indices.size(0);

  at::Tensor output = at::empty({batch, nonzeros}, a.options());

  LaunchSddmm(
      m,
      k,
      n,
      nonzeros,
      row_indices.data_ptr<int>(),
      row_offsets.data_ptr<int>(),
      column_indices.data_ptr<int>(),
      a.data_ptr<float>(),
      b.data_ptr<float>(),
      output.data_ptr<float>(),
      batch);

  return output;
}
} // namespace

TORCH_LIBRARY_IMPL(xformers, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::sddmm_sputnik"), TORCH_FN(sddmm_sputnik));
}
