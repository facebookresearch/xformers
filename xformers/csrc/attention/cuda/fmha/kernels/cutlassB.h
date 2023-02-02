// This file is auto-generated. See "generate_kernels.py"
#include "../kernel_backward.h"

#pragma once
#ifndef XFORMERS_MEM_EFF_ATTENTION_DISABLE_BACKWARD
// ======== f16 / sm50 ========
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, 32>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_k32_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, 64>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_k64_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, 128>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_k128_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_k65536_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, 32>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_k32_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, 64>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_k64_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, 128>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_k128_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_k65536_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, 32>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_k32_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, 64>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_k64_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, 128>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_k128_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_k65536_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, 32>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_k32_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, 64>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_k64_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, 128>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_k128_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_k65536_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, 65536>::Params p);

template <typename T> void dispatch_cutlassB_f16_sm50(T cb) {
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, 32>(), fmha_cutlassB_f16_aligned_k32_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, 64>(), fmha_cutlassB_f16_aligned_k64_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, 128>(), fmha_cutlassB_f16_aligned_k128_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, 65536>(), fmha_cutlassB_f16_aligned_k65536_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, 32>(), fmha_cutlassB_f16_aligned_k32_dropout_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, 64>(), fmha_cutlassB_f16_aligned_k64_dropout_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, 128>(), fmha_cutlassB_f16_aligned_k128_dropout_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, 65536>(), fmha_cutlassB_f16_aligned_k65536_dropout_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, 32>(), fmha_cutlassB_f16_notaligned_k32_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, 64>(), fmha_cutlassB_f16_notaligned_k64_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, 128>(), fmha_cutlassB_f16_notaligned_k128_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, 65536>(), fmha_cutlassB_f16_notaligned_k65536_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, 32>(), fmha_cutlassB_f16_notaligned_k32_dropout_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, 64>(), fmha_cutlassB_f16_notaligned_k64_dropout_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, 128>(), fmha_cutlassB_f16_notaligned_k128_dropout_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, 65536>(), fmha_cutlassB_f16_notaligned_k65536_dropout_sm50);
}

// ======== f32 / sm50 ========
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, 32>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_k32_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, 64>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_k64_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, 128>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_k128_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_k65536_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, 32>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_k32_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, 64>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_k64_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, 128>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_k128_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_k65536_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, float, false, false, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, float, false, false, 32>::kMinBlocksPerSm)
fmha_cutlassB_f32_notaligned_k32_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, float, false, false, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, float, false, false, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, float, false, false, 64>::kMinBlocksPerSm)
fmha_cutlassB_f32_notaligned_k64_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, float, false, false, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, float, false, false, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, float, false, false, 128>::kMinBlocksPerSm)
fmha_cutlassB_f32_notaligned_k128_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, float, false, false, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, float, false, false, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, float, false, false, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f32_notaligned_k65536_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, float, false, false, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, float, false, true, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, float, false, true, 32>::kMinBlocksPerSm)
fmha_cutlassB_f32_notaligned_k32_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, float, false, true, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, float, false, true, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, float, false, true, 64>::kMinBlocksPerSm)
fmha_cutlassB_f32_notaligned_k64_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, float, false, true, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, float, false, true, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, float, false, true, 128>::kMinBlocksPerSm)
fmha_cutlassB_f32_notaligned_k128_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, float, false, true, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, float, false, true, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, float, false, true, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f32_notaligned_k65536_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, float, false, true, 65536>::Params p);

template <typename T> void dispatch_cutlassB_f32_sm50(T cb) {
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, 32>(), fmha_cutlassB_f32_aligned_k32_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, 64>(), fmha_cutlassB_f32_aligned_k64_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, 128>(), fmha_cutlassB_f32_aligned_k128_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, 65536>(), fmha_cutlassB_f32_aligned_k65536_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, 32>(), fmha_cutlassB_f32_aligned_k32_dropout_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, 64>(), fmha_cutlassB_f32_aligned_k64_dropout_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, 128>(), fmha_cutlassB_f32_aligned_k128_dropout_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, 65536>(), fmha_cutlassB_f32_aligned_k65536_dropout_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, float, false, false, 32>(), fmha_cutlassB_f32_notaligned_k32_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, float, false, false, 64>(), fmha_cutlassB_f32_notaligned_k64_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, float, false, false, 128>(), fmha_cutlassB_f32_notaligned_k128_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, float, false, false, 65536>(), fmha_cutlassB_f32_notaligned_k65536_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, float, false, true, 32>(), fmha_cutlassB_f32_notaligned_k32_dropout_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, float, false, true, 64>(), fmha_cutlassB_f32_notaligned_k64_dropout_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, float, false, true, 128>(), fmha_cutlassB_f32_notaligned_k128_dropout_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, float, false, true, 65536>(), fmha_cutlassB_f32_notaligned_k65536_dropout_sm50);
}

// ======== f16 / sm70 ========
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, 32>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_k32_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, 64>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_k64_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, 128>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_k128_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_k65536_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, 32>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_k32_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, 64>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_k64_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, 128>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_k128_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_k65536_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, 32>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_k32_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, 64>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_k64_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, 128>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_k128_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_k65536_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, 32>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_k32_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, 64>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_k64_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, 128>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_k128_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_k65536_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, 65536>::Params p);

template <typename T> void dispatch_cutlassB_f16_sm70(T cb) {
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, 32>(), fmha_cutlassB_f16_aligned_k32_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, 64>(), fmha_cutlassB_f16_aligned_k64_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, 128>(), fmha_cutlassB_f16_aligned_k128_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, 65536>(), fmha_cutlassB_f16_aligned_k65536_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, 32>(), fmha_cutlassB_f16_aligned_k32_dropout_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, 64>(), fmha_cutlassB_f16_aligned_k64_dropout_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, 128>(), fmha_cutlassB_f16_aligned_k128_dropout_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, 65536>(), fmha_cutlassB_f16_aligned_k65536_dropout_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, 32>(), fmha_cutlassB_f16_notaligned_k32_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, 64>(), fmha_cutlassB_f16_notaligned_k64_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, 128>(), fmha_cutlassB_f16_notaligned_k128_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, 65536>(), fmha_cutlassB_f16_notaligned_k65536_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, 32>(), fmha_cutlassB_f16_notaligned_k32_dropout_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, 64>(), fmha_cutlassB_f16_notaligned_k64_dropout_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, 128>(), fmha_cutlassB_f16_notaligned_k128_dropout_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, 65536>(), fmha_cutlassB_f16_notaligned_k65536_dropout_sm70);
}

// ======== f32 / sm70 ========
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, float, true, false, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, float, true, false, 32>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_k32_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, float, true, false, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, float, true, false, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, float, true, false, 64>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_k64_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, float, true, false, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, float, true, false, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, float, true, false, 128>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_k128_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, float, true, false, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, float, true, false, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, float, true, false, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_k65536_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, float, true, false, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, float, true, true, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, float, true, true, 32>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_k32_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, float, true, true, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, float, true, true, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, float, true, true, 64>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_k64_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, float, true, true, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, float, true, true, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, float, true, true, 128>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_k128_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, float, true, true, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, float, true, true, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, float, true, true, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_k65536_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, float, true, true, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, 32>::kMinBlocksPerSm)
fmha_cutlassB_f32_notaligned_k32_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, 64>::kMinBlocksPerSm)
fmha_cutlassB_f32_notaligned_k64_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, 128>::kMinBlocksPerSm)
fmha_cutlassB_f32_notaligned_k128_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f32_notaligned_k65536_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, float, false, true, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, float, false, true, 32>::kMinBlocksPerSm)
fmha_cutlassB_f32_notaligned_k32_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, float, false, true, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, float, false, true, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, float, false, true, 64>::kMinBlocksPerSm)
fmha_cutlassB_f32_notaligned_k64_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, float, false, true, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, float, false, true, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, float, false, true, 128>::kMinBlocksPerSm)
fmha_cutlassB_f32_notaligned_k128_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, float, false, true, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, float, false, true, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, float, false, true, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f32_notaligned_k65536_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, float, false, true, 65536>::Params p);

template <typename T> void dispatch_cutlassB_f32_sm70(T cb) {
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, float, true, false, 32>(), fmha_cutlassB_f32_aligned_k32_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, float, true, false, 64>(), fmha_cutlassB_f32_aligned_k64_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, float, true, false, 128>(), fmha_cutlassB_f32_aligned_k128_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, float, true, false, 65536>(), fmha_cutlassB_f32_aligned_k65536_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, float, true, true, 32>(), fmha_cutlassB_f32_aligned_k32_dropout_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, float, true, true, 64>(), fmha_cutlassB_f32_aligned_k64_dropout_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, float, true, true, 128>(), fmha_cutlassB_f32_aligned_k128_dropout_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, float, true, true, 65536>(), fmha_cutlassB_f32_aligned_k65536_dropout_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, 32>(), fmha_cutlassB_f32_notaligned_k32_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, 64>(), fmha_cutlassB_f32_notaligned_k64_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, 128>(), fmha_cutlassB_f32_notaligned_k128_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, 65536>(), fmha_cutlassB_f32_notaligned_k65536_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, float, false, true, 32>(), fmha_cutlassB_f32_notaligned_k32_dropout_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, float, false, true, 64>(), fmha_cutlassB_f32_notaligned_k64_dropout_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, float, false, true, 128>(), fmha_cutlassB_f32_notaligned_k128_dropout_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, float, false, true, 65536>(), fmha_cutlassB_f32_notaligned_k65536_dropout_sm70);
}

// ======== f16 / sm75 ========
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, 32>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_k32_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, 64>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_k64_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, 128>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_k128_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_k65536_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, 32>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_k32_dropout_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, 64>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_k64_dropout_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, 128>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_k128_dropout_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_k65536_dropout_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, 32>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_k32_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, 64>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_k64_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, 128>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_k128_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_k65536_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, 32>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_k32_dropout_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, 64>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_k64_dropout_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, 128>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_k128_dropout_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_k65536_dropout_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, 65536>::Params p);

template <typename T> void dispatch_cutlassB_f16_sm75(T cb) {
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, 32>(), fmha_cutlassB_f16_aligned_k32_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, 64>(), fmha_cutlassB_f16_aligned_k64_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, 128>(), fmha_cutlassB_f16_aligned_k128_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, 65536>(), fmha_cutlassB_f16_aligned_k65536_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, 32>(), fmha_cutlassB_f16_aligned_k32_dropout_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, 64>(), fmha_cutlassB_f16_aligned_k64_dropout_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, 128>(), fmha_cutlassB_f16_aligned_k128_dropout_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, 65536>(), fmha_cutlassB_f16_aligned_k65536_dropout_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, 32>(), fmha_cutlassB_f16_notaligned_k32_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, 64>(), fmha_cutlassB_f16_notaligned_k64_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, 128>(), fmha_cutlassB_f16_notaligned_k128_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, 65536>(), fmha_cutlassB_f16_notaligned_k65536_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, 32>(), fmha_cutlassB_f16_notaligned_k32_dropout_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, 64>(), fmha_cutlassB_f16_notaligned_k64_dropout_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, 128>(), fmha_cutlassB_f16_notaligned_k128_dropout_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, 65536>(), fmha_cutlassB_f16_notaligned_k65536_dropout_sm75);
}

// ======== f32 / sm75 ========
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, float, true, false, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, float, true, false, 32>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_k32_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, float, true, false, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, float, true, false, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, float, true, false, 64>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_k64_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, float, true, false, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, float, true, false, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, float, true, false, 128>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_k128_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, float, true, false, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, float, true, false, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, float, true, false, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_k65536_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, float, true, false, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, float, true, true, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, float, true, true, 32>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_k32_dropout_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, float, true, true, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, float, true, true, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, float, true, true, 64>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_k64_dropout_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, float, true, true, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, float, true, true, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, float, true, true, 128>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_k128_dropout_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, float, true, true, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, float, true, true, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, float, true, true, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_k65536_dropout_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, float, true, true, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, 32>::kMinBlocksPerSm)
fmha_cutlassB_f32_notaligned_k32_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, 64>::kMinBlocksPerSm)
fmha_cutlassB_f32_notaligned_k64_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, 128>::kMinBlocksPerSm)
fmha_cutlassB_f32_notaligned_k128_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f32_notaligned_k65536_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, float, false, true, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, float, false, true, 32>::kMinBlocksPerSm)
fmha_cutlassB_f32_notaligned_k32_dropout_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, float, false, true, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, float, false, true, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, float, false, true, 64>::kMinBlocksPerSm)
fmha_cutlassB_f32_notaligned_k64_dropout_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, float, false, true, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, float, false, true, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, float, false, true, 128>::kMinBlocksPerSm)
fmha_cutlassB_f32_notaligned_k128_dropout_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, float, false, true, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, float, false, true, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, float, false, true, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f32_notaligned_k65536_dropout_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, float, false, true, 65536>::Params p);

template <typename T> void dispatch_cutlassB_f32_sm75(T cb) {
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, float, true, false, 32>(), fmha_cutlassB_f32_aligned_k32_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, float, true, false, 64>(), fmha_cutlassB_f32_aligned_k64_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, float, true, false, 128>(), fmha_cutlassB_f32_aligned_k128_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, float, true, false, 65536>(), fmha_cutlassB_f32_aligned_k65536_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, float, true, true, 32>(), fmha_cutlassB_f32_aligned_k32_dropout_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, float, true, true, 64>(), fmha_cutlassB_f32_aligned_k64_dropout_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, float, true, true, 128>(), fmha_cutlassB_f32_aligned_k128_dropout_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, float, true, true, 65536>(), fmha_cutlassB_f32_aligned_k65536_dropout_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, 32>(), fmha_cutlassB_f32_notaligned_k32_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, 64>(), fmha_cutlassB_f32_notaligned_k64_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, 128>(), fmha_cutlassB_f32_notaligned_k128_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, 65536>(), fmha_cutlassB_f32_notaligned_k65536_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, float, false, true, 32>(), fmha_cutlassB_f32_notaligned_k32_dropout_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, float, false, true, 64>(), fmha_cutlassB_f32_notaligned_k64_dropout_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, float, false, true, 128>(), fmha_cutlassB_f32_notaligned_k128_dropout_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, float, false, true, 65536>(), fmha_cutlassB_f32_notaligned_k65536_dropout_sm75);
}

// ======== bf16 / sm80 ========
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, 32>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_k32_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, 64>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_k64_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, 128>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_k128_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, 65536>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_k65536_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, 32>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_k32_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, 64>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_k64_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, 128>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_k128_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, 65536>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_k65536_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, 65536>::Params p);

template <typename T> void dispatch_cutlassB_bf16_sm80(T cb) {
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, 32>(), fmha_cutlassB_bf16_aligned_k32_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, 64>(), fmha_cutlassB_bf16_aligned_k64_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, 128>(), fmha_cutlassB_bf16_aligned_k128_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, 65536>(), fmha_cutlassB_bf16_aligned_k65536_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, 32>(), fmha_cutlassB_bf16_aligned_k32_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, 64>(), fmha_cutlassB_bf16_aligned_k64_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, 128>(), fmha_cutlassB_bf16_aligned_k128_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, 65536>(), fmha_cutlassB_bf16_aligned_k65536_dropout_sm80);
}

// ======== f16 / sm80 ========
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, 32>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_k32_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, 64>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_k64_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, 128>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_k128_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_k65536_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, 32>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_k32_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, 64>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_k64_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, 128>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_k128_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_k65536_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, 65536>::Params p);

template <typename T> void dispatch_cutlassB_f16_sm80(T cb) {
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, 32>(), fmha_cutlassB_f16_aligned_k32_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, 64>(), fmha_cutlassB_f16_aligned_k64_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, 128>(), fmha_cutlassB_f16_aligned_k128_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, 65536>(), fmha_cutlassB_f16_aligned_k65536_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, 32>(), fmha_cutlassB_f16_aligned_k32_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, 64>(), fmha_cutlassB_f16_aligned_k64_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, 128>(), fmha_cutlassB_f16_aligned_k128_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, 65536>(), fmha_cutlassB_f16_aligned_k65536_dropout_sm80);
}

// ======== f32 / sm80 ========
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, 32>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_k32_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, 64>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_k64_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, 128>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_k128_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_k65536_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, 32>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_k32_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, 64>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_k64_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, 128>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_k128_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_k65536_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, 65536>::Params p);

template <typename T> void dispatch_cutlassB_f32_sm80(T cb) {
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, 32>(), fmha_cutlassB_f32_aligned_k32_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, 64>(), fmha_cutlassB_f32_aligned_k64_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, 128>(), fmha_cutlassB_f32_aligned_k128_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, 65536>(), fmha_cutlassB_f32_aligned_k65536_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, 32>(), fmha_cutlassB_f32_aligned_k32_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, 64>(), fmha_cutlassB_f32_aligned_k64_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, 128>(), fmha_cutlassB_f32_aligned_k128_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, 65536>(), fmha_cutlassB_f32_aligned_k65536_dropout_sm80);
}


template <typename DT, typename T>
void dispatch_cutlassB(T cb, int cc = 0) {

    if (std::is_same<DT, cutlass::half_t>::value && 50 <= cc && cc < 70) {
        dispatch_cutlassB_f16_sm50(cb);
    }
    if (std::is_same<DT, float>::value && 50 <= cc && cc < 70) {
        dispatch_cutlassB_f32_sm50(cb);
    }
    if (std::is_same<DT, cutlass::half_t>::value && 70 <= cc && cc < 75) {
        dispatch_cutlassB_f16_sm70(cb);
    }
    if (std::is_same<DT, float>::value && 70 <= cc && cc < 75) {
        dispatch_cutlassB_f32_sm70(cb);
    }
    if (std::is_same<DT, cutlass::half_t>::value && 75 <= cc && cc < 80) {
        dispatch_cutlassB_f16_sm75(cb);
    }
    if (std::is_same<DT, float>::value && 75 <= cc && cc < 80) {
        dispatch_cutlassB_f32_sm75(cb);
    }
    if (std::is_same<DT, cutlass::bfloat16_t>::value && 80 <= cc && cc < 90) {
        dispatch_cutlassB_bf16_sm80(cb);
    }
    if (std::is_same<DT, cutlass::half_t>::value && 80 <= cc && cc < 90) {
        dispatch_cutlassB_f16_sm80(cb);
    }
    if (std::is_same<DT, float>::value && 80 <= cc && cc < 90) {
        dispatch_cutlassB_f32_sm80(cb);
    }
}
#endif // XFORMERS_MEM_EFF_ATTENTION_DISABLE_BACKWARD
