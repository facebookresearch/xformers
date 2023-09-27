/* 
  TODO: license header
*/

#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>
#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/Tensor.h>

namespace {

template<typename scalar_t>
__global__ void
efficient_attention_forward_decoder_ck_kernel(
    at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> XQ,
    at::PackedTensorAccessor64<scalar_t, 4, at::RestrictPtrTraits> cache_K,
    at::PackedTensorAccessor64<scalar_t, 4, at::RestrictPtrTraits> cache_V,
    at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> O,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> seq_positions,
    float qk_scale
) {
  __syncthreads();
}

#define AT_DISPATCH_CASE_3(SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, ...) \
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__) \
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__) \
  AT_DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__) 
  
#define AT_DISPATCH_SWITCH_3(SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                               \
      TYPE,                                                         \
      NAME,                                                         \
      AT_DISPATCH_CASE_3(SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, __VA_ARGS__))

at::Tensor
efficient_attention_forward_decoder_ck(
    const at::Tensor& XQ, // [B, 1, H, D]
    const at::Tensor& cache_K, // [B, T_MAX, H or 1, D]
    const at::Tensor& cache_V, // [B, T_MAX, H or 1, D]
    const at::Tensor& seq_positions, // [B]
    double qk_scale) {

  constexpr int32_t kThreadsPerWavefront = 32;
  constexpr int32_t kWavefrontsPerBlock = 32;
  constexpr int32_t D_H = 128;
  constexpr int32_t T_MAX = 8192;

  at::OptionalDeviceGuard guard(XQ.device());
  TORCH_CHECK(XQ.is_cuda());
  TORCH_CHECK(cache_K.is_cuda());
  TORCH_CHECK(cache_V.is_cuda());

  TORCH_CHECK(seq_positions.is_cuda());

  TORCH_CHECK(cache_K.size(1) <= T_MAX);
  TORCH_CHECK(cache_K.size(3) == D_H);

  auto O = at::empty_like(XQ);
  auto B = XQ.size(0);
  auto H = XQ.size(2);
  dim3 blocks(B, H);
  dim3 threads(kThreadsPerWavefront, kWavefrontsPerBlock);

  int32_t smem_softmax = T_MAX * sizeof(float) + kWavefrontsPerBlock * sizeof(float);
  int32_t smem_output = D_H * sizeof(float) * kWavefrontsPerBlock;
  int32_t smem = max(smem_softmax, smem_output);
  auto stream = at::cuda::getCurrentHIPStream().stream();

  AT_DISPATCH_SWITCH_3(at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::Float, 
    XQ.scalar_type(), "efficient_attention_forward_decoder_ck", [&] {
      auto* kernel = &efficient_attention_forward_decoder_ck_kernel<scalar_t>;
      if (smem > 48 * 1024) {
        C10_CUDA_CHECK(hipFuncSetAttribute(
            reinterpret_cast<void*&>(kernel),
            hipFuncAttributeMaxDynamicSharedMemorySize,
            smem));
      }
      kernel
          <<<blocks, threads, smem, stream>>>(
              XQ.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(),
              cache_K.packed_accessor64<scalar_t, 4, at::RestrictPtrTraits>(),
              cache_V.packed_accessor64<scalar_t, 4, at::RestrictPtrTraits>(),
              O.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(),
              seq_positions
                  .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
              qk_scale);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
  });

  return O;
}  

#undef AT_DISPATCH_CASE_3
#undef AT_DISPATCH_SWITCH_3

} // namespace

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::efficient_attention_forward_decoder_ck"),
      TORCH_FN(efficient_attention_forward_decoder_ck));
}