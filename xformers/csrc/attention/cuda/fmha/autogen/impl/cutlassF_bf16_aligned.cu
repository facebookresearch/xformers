// This file is auto-generated. See "generate_kernels.py"
#ifndef XFORMERS_MEM_EFF_ATTENTION_DISABLE_FORWARD
#include "../../kernel_forward.h"
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, true, true, true>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, true, true, true>::kMinBlocksPerSm)
fmha_cutlassF_bf16_aligned_64x64_rf_sm80(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, true, true, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 900
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, true, true, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassF_bf16_aligned_64x64_rf_sm80` is for sm80-sm90, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, true, true, true>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, true, true, true>::kMinBlocksPerSm)
fmha_cutlassF_bf16_aligned_32x128_rf_sm80(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, true, true, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 900
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, true, true, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassF_bf16_aligned_32x128_rf_sm80` is for sm80-sm90, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, false, true, true>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, false, true, true>::kMinBlocksPerSm)
fmha_cutlassF_bf16_aligned_32x128_gmem_sm80(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, false, true, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 900
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, false, true, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassF_bf16_aligned_32x128_gmem_sm80` is for sm80-sm90, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}
#endif // XFORMERS_MEM_EFF_ATTENTION_DISABLE_FORWARD
