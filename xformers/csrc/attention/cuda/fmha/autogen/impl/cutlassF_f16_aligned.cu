// This file is auto-generated. See "generate_kernels.py"
#ifndef XFORMERS_MEM_EFF_ATTENTION_DISABLE_FORWARD
#include "../../kernel_forward.h"
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, true, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, true, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_64x64_rf_sm50(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, true, true, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, true, true, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassF_f16_aligned_64x64_rf_sm50` is for sm50-sm70, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_64x64_rf_sm70(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, true, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, true, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassF_f16_aligned_64x64_rf_sm70` is for sm70-sm75, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, true, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, true, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_64x64_rf_sm75(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, true, true, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, true, true, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassF_f16_aligned_64x64_rf_sm75` is for sm75-sm80, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, true, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, true, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_64x64_rf_sm80(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, true, true, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 900
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, true, true, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassF_f16_aligned_64x64_rf_sm80` is for sm80-sm90, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, true, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, true, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_32x128_rf_sm50(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, true, true, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, true, true, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassF_f16_aligned_32x128_rf_sm50` is for sm50-sm70, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_32x128_rf_sm70(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, true, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, true, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassF_f16_aligned_32x128_rf_sm70` is for sm70-sm75, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, true, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, true, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_32x128_rf_sm75(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, true, true, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, true, true, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassF_f16_aligned_32x128_rf_sm75` is for sm75-sm80, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, true, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, true, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_32x128_rf_sm80(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, true, true, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 900
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, true, true, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassF_f16_aligned_32x128_rf_sm80` is for sm80-sm90, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, false, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, false, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_32x128_gmem_sm50(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, false, true, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, false, true, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassF_f16_aligned_32x128_gmem_sm50` is for sm50-sm70, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_32x128_gmem_sm70(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, true, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, true, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassF_f16_aligned_32x128_gmem_sm70` is for sm70-sm75, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, false, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, false, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_32x128_gmem_sm75(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, false, true, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, false, true, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassF_f16_aligned_32x128_gmem_sm75` is for sm75-sm80, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, false, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, false, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_32x128_gmem_sm80(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, false, true, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 900
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, false, true, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassF_f16_aligned_32x128_gmem_sm80` is for sm80-sm90, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}
#endif // XFORMERS_MEM_EFF_ATTENTION_DISABLE_FORWARD
