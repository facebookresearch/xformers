#include <ATen/ATen.h>
#include <torch/types.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <sputnik/sddmm/cuda_sddmm.h>
#include <sputnik/spmm/cuda_spmm.h>

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

TORCH_LIBRARY_FRAGMENT(xformers, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::sddmm_sputnik(Tensor a, Tensor b, Tensor row_indices, Tensor row_offsets, Tensor column_indices) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::spmm_sputnik(Tensor b, Tensor row_indices, Tensor values, Tensor row_offsets, Tensor column_indices, int m) -> Tensor"));
}

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::sddmm_sputnik"), TORCH_FN(sddmm_sputnik));
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::spmm_sputnik"), TORCH_FN(spmm_sputnik));
}
