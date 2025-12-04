#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "pt_stable_utils.h"
#include "warp_tensor.h"

using namespace xformers::sp24;

namespace {
template <int N, int M, typename Element, typename PreprocFn>
void __global__ sparseNM_dense_kernel(
    Element const* in_ptr,
    int64_t in_s0,
    Element* out_ptr,
    int64_t out_s0,
    PreprocFn sort_preproc) {
  constexpr int kBlockSize0 = 64;
  constexpr int kBlockSize1 = 64;

  int64_t block0 = blockIdx.x * kBlockSize0;
  int64_t block1 = blockIdx.y * kBlockSize1;

  WarpTensor<Element, 8, M * 4> tile;
  static_assert(tile.kElementsPerThread == M);
  CUTLASS_PRAGMA_UNROLL
  for (int tile0 = 0; tile0 < kBlockSize0; tile0 += tile.kRows) {
    CUTLASS_PRAGMA_UNROLL
    for (int tile1 = 0; tile1 < kBlockSize1; tile1 += tile.kCols) {
      // load
      tile.load(in_ptr + (block0 + tile0) * in_s0 + block1 + tile1, in_s0);

      // sparsify dense
      tile = tile.template sparsify_dense<N, M>(sort_preproc);

      // store
      tile.store(out_ptr + (block0 + tile0) * out_s0 + block1 + tile1, out_s0);
    }
  }
}

template <bool kIsMeta>
torch::stable::Tensor sparseNM_dense(
    torch::stable::Tensor input,
    std::string sort_preproc,
    int64_t kN,
    int64_t kM) {
  std::optional<torch::stable::accelerator::DeviceGuard> device_guard;
  if (!kIsMeta) {
    STD_TORCH_CHECK(input.is_cuda(), "All tensors must be on GPU");
    device_guard.emplace(input.device().index());
  }

  STD_TORCH_CHECK(input.dim() == 2, "Can only sparsify 2d tensors");
  STD_TORCH_CHECK(
      input.stride(1) == 1,
      "Can only sparsify contiguous tensors. Sparsify the transpose otherwise.");
  STD_TORCH_CHECK(input.size(0) % 64 == 0);
  STD_TORCH_CHECK(input.size(1) % 64 == 0);

  torch::stable::Tensor out = torch::stable::empty_like(input);
  bool foundKernel = false;
  auto launchKernel = [&](auto dtype, auto N, auto M, auto sort_preproc) {
    if (foundKernel) {
      return;
    }
    foundKernel = true;
    if (kIsMeta) {
      return;
    }
    using Element = decltype(dtype);
    dim3 num_blocks(input.size(0) / 64, input.size(1) / 64, 1);
    sparseNM_dense_kernel<N.value, M.value>
        <<<num_blocks, 32, 0, xf_getCurrentCUDAStream()>>>(
            (Element const*)input.data_ptr(),
            input.stride(0),
            (Element*)out.data_ptr(),
            out.stride(0),
            sort_preproc);
  };
  STD_TORCH_CHECK(
      input.scalar_type() == torch::headeronly::ScalarType::BFloat16);
  auto dtype = cutlass::bfloat16_t();
  auto Identity = [] __device__(auto x) { return x; };
  auto Abs = [] __device__(auto x) { return cutlass::abs(x); };
  auto testNM = [&](auto N, auto M) {
    if (N.value != kN || M.value != kM) {
      return;
    }
    if (sort_preproc == "largest") {
      launchKernel(dtype, N, M, Identity);
    } else if (sort_preproc == "largest_abs") {
      launchKernel(dtype, N, M, Abs);
    }
  };
  // 2:8 sparsification
  testNM(std::integral_constant<int, 2>(), std::integral_constant<int, 8>());
  // 2:4 sparsification
  testNM(std::integral_constant<int, 2>(), std::integral_constant<int, 4>());
  STD_TORCH_CHECK(foundKernel, "Kernel not found");
  XF_CUDA_KERNEL_LAUNCH_CHECK();

  return out;
}
} // namespace

STABLE_TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl("sparseNM_dense", XF_BOXED_FN(sparseNM_dense<false>));
}

STABLE_TORCH_LIBRARY_IMPL(xformers, Meta, m) {
  m.impl("sparseNM_dense", XF_BOXED_FN(sparseNM_dense<true>));
}

STABLE_TORCH_LIBRARY_FRAGMENT(xformers, m) {
  m.def(
      "sparseNM_dense(Tensor input, str sort_preproc, int N, int M) -> Tensor");
}
