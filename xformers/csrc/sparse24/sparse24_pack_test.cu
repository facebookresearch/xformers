#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/TensorAccessor.h>

#include "pt_stable_utils.h"
#include "sparse24_pack.h"

using namespace xformers::sp24;

namespace {
__global__ void meta_shuffle_test_kernel(
    torch::headeronly::HeaderOnlyGenericPackedTensorAccessor<int64_t, 3>
        local_meta,
    torch::headeronly::HeaderOnlyGenericPackedTensorAccessor<int64_t, 3>
        final_meta,
    bool transpose) {
  uint32_t meta_ab = 0;
  uint32_t meta_cd = 0;
  for (int i = 0; i < 4; ++i) {
    meta_ab |= uint8b_t(uint32_t(local_meta[threadIdx.x][threadIdx.y][i]))
        << (8 * i);
    meta_cd |= uint8b_t(uint32_t(local_meta[threadIdx.x][threadIdx.y][4 + i]))
        << (8 * i);
  }
  final_meta[threadIdx.x][threadIdx.y][0] =
      warp_shuffle_meta(meta_ab, transpose);
  final_meta[threadIdx.x][threadIdx.y][1] =
      warp_shuffle_meta(meta_cd, transpose);
}

torch::stable::Tensor _sparse24_meta_shuffle_test(
    torch::stable::Tensor local_meta,
    bool transpose) {
  auto threads_grid = KernelTypes<cutlass::half_t>::Params::getThreadsGrid();

  STD_TORCH_CHECK(
      local_meta.scalar_type() == torch::headeronly::ScalarType::Long);
  STD_TORCH_CHECK(local_meta.dim() == 3);
  STD_TORCH_CHECK(local_meta.size(0) == threads_grid.x);
  STD_TORCH_CHECK(local_meta.size(1) == threads_grid.y);
  STD_TORCH_CHECK(local_meta.size(2) == kThreadY);
  torch::stable::accelerator::DeviceGuard device_guard(
      local_meta.device().index());
  cudaStream_t stream = xf_getCurrentCUDAStream();

  torch::stable::Tensor final_meta = xf_zeros(
      {threads_grid.x, threads_grid.y, 2},
      local_meta.scalar_type(),
      local_meta.device());
  size_t smem_bytes = 0;
  meta_shuffle_test_kernel<<<1, threads_grid, smem_bytes, stream>>>(
      xf_packed_accessor<int64_t, 3>(local_meta),
      xf_packed_accessor<int64_t, 3>(final_meta),
      transpose);
  return final_meta;
}
} // namespace

STABLE_TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      "_sparse24_meta_shuffle_test", XF_BOXED_FN(_sparse24_meta_shuffle_test));
}
