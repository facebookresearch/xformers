#pragma once

// All kernels are disabled by default
#define INSTANTIATE_ATTENTION_KERNEL_BACKWARD_SM50(...) \
  INSTANTIATE_ATTENTION_KERNEL_BACKWARD_DISABLED(50, __VA_ARGS__)
#define INSTANTIATE_ATTENTION_KERNEL_BACKWARD_SM70(...) \
  INSTANTIATE_ATTENTION_KERNEL_BACKWARD_DISABLED(70, __VA_ARGS__)
#define INSTANTIATE_ATTENTION_KERNEL_BACKWARD_SM75(...) \
  INSTANTIATE_ATTENTION_KERNEL_BACKWARD_DISABLED(75, __VA_ARGS__)
#define INSTANTIATE_ATTENTION_KERNEL_BACKWARD_SM80(...) \
  INSTANTIATE_ATTENTION_KERNEL_BACKWARD_DISABLED(80, __VA_ARGS__)

#ifndef XFORMERS_MEM_EFF_ATTENTION_DISABLE_BACKWARD
#include "../kernel_backward.h"

#define _ATTENTION_KERNEL_BACKWARD_BEGIN(...)                 \
  template <>                                                 \
  __global__ void __launch_bounds__(                          \
      __VA_ARGS__::kNumThreads, __VA_ARGS__::kMinBlocksPerSm) \
      attention_kernel_backward_batched<__VA_ARGS__>(         \
          typename __VA_ARGS__::Params p) {                   \
    using Kernel = __VA_ARGS__;
#define _ATTENTION_KERNEL_BACKWARD_END() }

#define INSTANTIATE_ATTENTION_KERNEL_BACKWARD(ARCH, ...)             \
  _ATTENTION_KERNEL_BACKWARD_BEGIN(                                  \
      AttentionBackwardKernel<cutlass::arch::Sm##ARCH, __VA_ARGS__>) \
  p.advance_to_block();                                              \
  Kernel::kernel(p);                                                 \
  _ATTENTION_KERNEL_BACKWARD_END();

#ifdef __CUDA_ARCH__
#define __CUDA_ARCH_OR_ZERO__ __CUDA_ARCH__
#else
#define __CUDA_ARCH_OR_ZERO__ 0
#endif

#define INSTANTIATE_ATTENTION_KERNEL_BACKWARD_DISABLED(ARCH, ...)                \
  _ATTENTION_KERNEL_BACKWARD_BEGIN(                                              \
      AttentionBackwardKernel<cutlass::arch::Sm##ARCH, __VA_ARGS__>)             \
  printf(                                                                        \
      "FATAL: this function is for sm%d, but was built with __CUDA_ARCH__=%d\n", \
      int(ARCH),                                                                 \
      int(__CUDA_ARCH_OR_ZERO__));                                               \
  _ATTENTION_KERNEL_BACKWARD_END();

// Enable the right one based on __CUDA_ARCH__
#ifndef __CUDA_ARCH__
#elif __CUDA_ARCH__ < 500
#error "Need cuda arch at least 5.0"
#elif __CUDA_ARCH__ < 700
#undef INSTANTIATE_ATTENTION_KERNEL_BACKWARD_SM50
#define INSTANTIATE_ATTENTION_KERNEL_BACKWARD_SM50(...) \
  INSTANTIATE_ATTENTION_KERNEL_BACKWARD(50, __VA_ARGS__)
#elif __CUDA_ARCH__ < 750
#undef INSTANTIATE_ATTENTION_KERNEL_BACKWARD_SM70
#define INSTANTIATE_ATTENTION_KERNEL_BACKWARD_SM70(...) \
  INSTANTIATE_ATTENTION_KERNEL_BACKWARD(70, __VA_ARGS__)
#elif __CUDA_ARCH__ < 800
#undef INSTANTIATE_ATTENTION_KERNEL_BACKWARD_SM75
#define INSTANTIATE_ATTENTION_KERNEL_BACKWARD_SM75(...) \
  INSTANTIATE_ATTENTION_KERNEL_BACKWARD(75, __VA_ARGS__)
#elif __CUDA_ARCH__ >= 800
#undef INSTANTIATE_ATTENTION_KERNEL_BACKWARD_SM80
#define INSTANTIATE_ATTENTION_KERNEL_BACKWARD_SM80(...) \
  INSTANTIATE_ATTENTION_KERNEL_BACKWARD(80, __VA_ARGS__)
#endif
#endif // XFORMERS_MEM_EFF_ATTENTION_DISABLE_BACKWARD
