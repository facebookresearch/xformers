#include <ATen/ScalarOps.h>
#include <ATen/Tensor.h>
#include <ATen/autocast_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <torch/types.h>

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
at::Tensor sparseNM_dense(
    at::Tensor input,
    std::string sort_preproc,
    int64_t kN,
    int64_t kM) {
  std::optional<at::cuda::CUDAGuard> device_guard;
  if (!kIsMeta) {
    TORCH_CHECK(input.is_cuda(), "All tensors must be on GPU");
    device_guard.emplace(input.device());
  }

  TORCH_CHECK(input.dim() == 2, "Can only sparsify 2d tensors");
  TORCH_CHECK(
      input.stride(1) == 1,
      "Can only sparsify contiguous tensors. Sparsify the transpose otherwise.");
  TORCH_CHECK(input.size(0) % 64 == 0);
  TORCH_CHECK(input.size(1) % 64 == 0);

  at::Tensor out = at::empty_like(input);
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
        <<<num_blocks, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
            (Element const*)input.data_ptr(),
            input.stride(0),
            (Element*)out.data_ptr(),
            out.stride(0),
            sort_preproc);
  };
  TORCH_CHECK(input.scalar_type() == at::ScalarType::BFloat16);
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
  TORCH_CHECK(foundKernel, "Kernel not found");
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return out;
}
} // namespace

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::sparseNM_dense"),
      TORCH_FN(sparseNM_dense<false>));
}

TORCH_LIBRARY_IMPL(xformers, Meta, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::sparseNM_dense"),
      TORCH_FN(sparseNM_dense<true>));
}

TORCH_LIBRARY_FRAGMENT(xformers, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::sparseNM_dense(Tensor input, str sort_preproc, int N, int M) -> Tensor"));
}
