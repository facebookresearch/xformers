#include <ATen/ATen.h>
#include <torch/types.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <sputnik/sddmm/cuda_sddmm.h>
#include <sputnik/softmax/sparse_softmax.h>
#include <sputnik/spmm/cuda_spmm.h>

#include <sputnik/load_store.h>

namespace {

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
    float* __restrict__ output_values) {
  // Calculate the index of the row that this block will process.
  int m_index = blockIdx.x * blockDim.y + threadIdx.y;
  if (m_index >= m)
    return;
  m_index = sputnik::Load(row_indices + m_index);

  // Load the row offset and calculate the number of non-zeros in
  // the row.
  int row_offset = sputnik::Load(row_offsets + m_index);
  int nonzeros = sputnik::Load(row_offsets + m_index + 1) - row_offset;

  const float* in = values + row_offset;
  const float* grad = gradient + row_offset;

  // Step 1: Compute the intermediate sum used for the gradient
  float sum = 0.0f;
  for (int idx = threadIdx.x; idx < nonzeros; idx += blockDim.x) {
    sum += sputnik::Load(in + idx) * sputnik::Load(grad + idx);
  }
  for (int idx = 1; idx < blockDim.x; idx *= 2) {
    sum += __shfl_xor_sync(0xffffffff, sum, idx);
  }

  // step 2: Compute the gradients
  float* out = output_values + row_offset;
  for (int idx = threadIdx.x; idx < nonzeros; idx += blockDim.x) {
    sputnik::Store(
        sputnik::Load(in + idx) * (sputnik::Load(grad + idx) - sum), out + idx);
  }
}

} // namespace

at::Tensor sparse_softmax_backward_sputnik(
    int64_t m,
    int64_t n,
    const at::Tensor& row_indices,
    const at::Tensor& values,
    const at::Tensor& grad,
    const at::Tensor& row_offsets,
    const at::Tensor& column_indices) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int batch = values.size(0);
  int nonzeros = column_indices.size(0);

  at::Tensor output = at::empty({batch, nonzeros}, values.options());

  // NOTE: SparseSoftmaxBackwardKernel currently only supports 1 warp per row
  // of the input matrix. We launch two warps per block, with each
  // mapped to different rows to enable us to hit max occupancy.
  constexpr int kBlockWidth = 32;
  constexpr int kWarpsPerBlock = 2;
  dim3 grid_dim(std::ceil(static_cast<float>(m) / kWarpsPerBlock));
  dim3 block_dim(kBlockWidth, kWarpsPerBlock);

  for (int i = 0; i < batch; i++) {
    SparseSoftmaxBackwardKernel<<<grid_dim, block_dim, 0, stream>>>(
        m,
        n,
        grad.data_ptr<float>() + nonzeros * i,
        values.data_ptr<float>() + nonzeros * i,
        row_indices.data_ptr<int>(),
        row_offsets.data_ptr<int>(),
        column_indices.data_ptr<int>(),
        output.data_ptr<float>() + nonzeros * i);
    AT_CUDA_CHECK(cudaGetLastError());
  }

  return output;
}

at::Tensor sddmm_sputnik(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& row_indices,
    const at::Tensor& row_offsets,
    const at::Tensor& column_indices) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int batch = a.size(0);
  int m = a.size(1);
  int k = a.size(2);
  int n = b.size(1);

  int nonzeros = column_indices.size(0);

  at::Tensor output = at::empty({batch, nonzeros}, a.options());

  for (int i = 0; i < batch; i++) {
    AT_CUDA_CHECK(sputnik::CudaSddmm(
        m,
        k,
        n,
        nonzeros,
        row_indices.data_ptr<int>(),
        row_offsets.data_ptr<int>(),
        column_indices.data_ptr<int>(),
        a.data_ptr<float>() + m * k * i,
        b.data_ptr<float>() + k * n * i,
        output.data_ptr<float>() + nonzeros * i,
        stream));
  }

  return output;
}

at::Tensor spmm_sputnik(
    const at::Tensor& b,
    const at::Tensor& row_indices,
    const at::Tensor& values,
    const at::Tensor& row_offsets,
    const at::Tensor& column_indices,
    int64_t m) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int batch = b.size(0);
  int k = b.size(1);
  int n = b.size(2);

  int nonzeros = column_indices.size(0);
  TORCH_CHECK(
      batch == 1 || nonzeros % 4 == 0,
      "If batch size > 1 then number of nonzeros should be a multiple of 4");

  at::Tensor output = at::empty({batch, m, n}, b.options());

  for (int i = 0; i < batch; i++) {
    // TODO investigate misaligned address errors in values ptr
    AT_CUDA_CHECK(sputnik::CudaSpmm(
        m,
        k,
        n,
        nonzeros,
        row_indices.data_ptr<int>(),
        values.data_ptr<float>() + nonzeros * i,
        row_offsets.data_ptr<int>(),
        column_indices.data_ptr<int>(),
        b.data_ptr<float>() + k * n * i,
        output.data_ptr<float>() + m * n * i,
        stream));
  }

  return output;
}

at::Tensor sparse_softmax_sputnik(
    int64_t m,
    int64_t n,
    const at::Tensor& row_indices,
    const at::Tensor& values,
    const at::Tensor& row_offsets,
    const at::Tensor& column_indices) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int batch = values.size(0);
  int nonzeros = column_indices.size(0);

  at::Tensor output = at::empty({batch, nonzeros}, values.options());

  for (int i = 0; i < batch; i++) {
    AT_CUDA_CHECK(sputnik::SparseSoftmax(
        m,
        n,
        nonzeros,
        values.data_ptr<float>() + nonzeros * i,
        row_indices.data_ptr<int>(),
        row_offsets.data_ptr<int>(),
        column_indices.data_ptr<int>(),
        output.data_ptr<float>() + nonzeros * i,
        stream));
  }

  return output;
}

TORCH_LIBRARY_FRAGMENT(xformers, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::sddmm_sputnik(Tensor a, Tensor b, Tensor row_indices, Tensor row_offsets, Tensor column_indices) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::sparse_softmax_sputnik(int m, int n, Tensor row_indices, Tensor values, Tensor row_offsets, Tensor column_indices) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::sparse_softmax_backward_sputnik(int m, int n, Tensor row_indices, Tensor values, Tensor gradient, Tensor row_offsets, Tensor column_indices) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::spmm_sputnik(Tensor b, Tensor row_indices, Tensor values, Tensor row_offsets, Tensor column_indices, int m) -> Tensor"));
}

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::sddmm_sputnik"), TORCH_FN(sddmm_sputnik));
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::sparse_softmax_sputnik"),
      TORCH_FN(sparse_softmax_sputnik));
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::sparse_softmax_backward_sputnik"),
      TORCH_FN(sparse_softmax_backward_sputnik));
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::spmm_sputnik"), TORCH_FN(spmm_sputnik));
}
