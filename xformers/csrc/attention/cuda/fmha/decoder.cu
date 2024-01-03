/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <ATen/cuda/Atomic.cuh>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
#include <cuda/atomic>
#endif

namespace {

// Each block handles a single batch and head

// Each warp handles separate D dimension.

// Load Q into registers in all warps.
// Split T across warps in a block
// Compute S[T_MAX] = for i in range(T): S[t] = sum(Q[d] * K[t, d])
// Use shared reduction to compute max and compute softmax on shared memory.

// Split T across warps in a block

// each warp compute sum(t_subset) P[t] * V[t_subset, d]
// outputs are of size float[D]

constexpr int32_t kThreadsPerWarp = 32;
constexpr int32_t kWarpsPerBlock = 32;
constexpr int32_t D_H = 128;
constexpr int32_t T_MAX = 8192;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700

inline __device__ float2 bf1622float2(const __nv_bfloat162 val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  float2 f_val;
  f_val.x = __low2float(val);
  f_val.y = __high2float(val);
  return f_val;
#else
  return __bfloat1622float2(val);
#endif
}

struct __align__(16) fx4 {
  float x;
  float y;
  float z;
  float w;
  __host__ __device__ fx4() {
    x = 0;
    y = 0;
    z = 0;
    w = 0;
  }
};

template <typename scalar_t>
struct scalar4;

template <> // bfx4
struct __align__(8) scalar4<at::BFloat16> {
  __nv_bfloat162 vals[2];
  using whole_int_t = uint2;
};
template <>
struct __align__(8) scalar4<at::Half> {
  __half2 vals[2];
  using whole_int_t = uint2;
};
template <>
struct scalar4<float> {
  fx4 v;
  using whole_int_t = uint4;
};

// bfx4_dot
__device__ __forceinline__ float scalar4_dot(
    scalar4<at::BFloat16> a,
    scalar4<at::BFloat16> b) {
  // float2 acc = {0, 0};
  // __nv_bfloat162 acc;
  // acc.x = static_cast<int>(0);
  // acc.y = static_cast<int>(0);
  // TODO: need to be performed in float32?
  auto a0 = bf1622float2(a.vals[0]);
  auto a1 = bf1622float2(a.vals[1]);
  auto b0 = bf1622float2(b.vals[0]);
  auto b1 = bf1622float2(b.vals[1]);
  return a0.x * b0.x + a0.y * b0.y + a1.x * b1.x + a1.y * b1.y;

  // acc = __hfma2(a.vals[0], b.vals[0], acc);
  // acc = __hfma2(a.vals[1], b.vals[1], acc);
  // auto r = bf1622float2(acc);
  // return r.x + r.y;
}
__device__ __forceinline__ float scalar4_dot(
    scalar4<at::Half> a,
    scalar4<at::Half> b) {
  auto a0 = __half22float2(a.vals[0]);
  auto a1 = __half22float2(a.vals[1]);
  auto b0 = __half22float2(b.vals[0]);
  auto b1 = __half22float2(b.vals[1]);
  return a0.x * b0.x + a0.y * b0.y + a1.x * b1.x + a1.y * b1.y;
}
__device__ __forceinline__ float scalar4_dot(
    scalar4<float> a,
    scalar4<float> b) {
  return a.v.x * b.v.x + a.v.y * b.v.y + a.v.z * b.v.z + a.v.w * b.v.w;
}

// bfx4_scale_acc
__device__ __forceinline__ fx4
scalar4_scale_acc(fx4 acc, scalar4<at::BFloat16> a, float b) {
  auto axy = bf1622float2(a.vals[0]);
  auto azw = bf1622float2(a.vals[1]);
  acc.x += axy.x * b;
  acc.y += axy.y * b;
  acc.z += azw.x * b;
  acc.w += azw.y * b;
  return acc;
}
__device__ __forceinline__ fx4
scalar4_scale_acc(fx4 acc, scalar4<at::Half> a, float b) {
  auto axy = __half22float2(a.vals[0]);
  auto azw = __half22float2(a.vals[1]);
  acc.x += axy.x * b;
  acc.y += axy.y * b;
  acc.z += azw.x * b;
  acc.w += azw.y * b;
  return acc;
}
__device__ __forceinline__ fx4
scalar4_scale_acc(fx4 acc, scalar4<float> a, float b) {
  acc.x += a.v.x * b;
  acc.y += a.v.y * b;
  acc.z += a.v.z * b;
  acc.w += a.v.w * b;
  return acc;
}
__device__ __forceinline__ fx4 fx4_acc(fx4 a, fx4 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
  return a;
}

template <typename scalar_t>
scalar4<scalar_t> fx4_to_scalar4(fx4 a);

template <> // fx4_to_bfx4
__device__ __forceinline__ scalar4<at::BFloat16> fx4_to_scalar4<at::BFloat16>(
    fx4 a) {
  scalar4<at::BFloat16> r;
  r.vals[0] = __floats2bfloat162_rn(a.x, a.y);
  r.vals[1] = __floats2bfloat162_rn(a.z, a.w);
  return r;
}
template <>
__device__ __forceinline__ scalar4<at::Half> fx4_to_scalar4<at::Half>(fx4 a) {
  scalar4<at::Half> r;
  r.vals[0] = __floats2half2_rn(a.x, a.y);
  r.vals[1] = __floats2half2_rn(a.z, a.w);
  return r;
}
template <>
__device__ __forceinline__ scalar4<float> fx4_to_scalar4<float>(fx4 a) {
  return {a};
}
#define FINAL_MASK 0xffffffff

template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

template <typename T>
__inline__ __device__ T warpReduceMax(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
  return val;
}

#endif // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700

// TODO: can also fuse RoPe into this kernel. Doesn't seem worth it.
template <
    typename scalar_t,
    // Offset from values read from seq_positions.
    // Never nonzero in Python xformers library.
    int seq_positions_shift = 0>
__global__ void mqa_attn_kernel(
    at::PackedTensorAccessor32<scalar_t, 5, at::RestrictPtrTraits> XQ,
    at::PackedTensorAccessor64<scalar_t, 5, at::RestrictPtrTraits> cache_K,
    at::PackedTensorAccessor64<scalar_t, 5, at::RestrictPtrTraits> cache_V,
    at::PackedTensorAccessor32<scalar_t, 5, at::RestrictPtrTraits> O,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> seq_positions,
    float qk_scale) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  using whole_int_t = typename scalar4<scalar_t>::whole_int_t;
  static_assert(4 * kThreadsPerWarp == D_H, "");
  static_assert(kWarpsPerBlock <= kThreadsPerWarp, "");

  extern __shared__ __align__(16) float smem[];

  // Each block handles a single batch and head
  int32_t b = blockIdx.x;
  int32_t g = blockIdx.y;
  int32_t h = blockIdx.z;

  // Note: this is decoding case where we attent to current and all previous
  // tokens.
  int32_t t_max = seq_positions[b] + seq_positions_shift;

  int32_t warp_idx = threadIdx.y;
  // need kWarpsPerBlock == blockDim.y;
  // Need D_H == 128
  auto* q_ = &(XQ[b][0][g][h][0]);

  auto* cache_K_base = &cache_K[b][0][g][h][0];
  auto* cache_V_base = &cache_V[b][0][g][h][0];

  // Load Q into registers in all warps.
  // Each thread handles 4 D dimensions
  scalar4<scalar_t> q_thread;
  *reinterpret_cast<whole_int_t*>(&q_thread) =
      *(reinterpret_cast<const whole_int_t*>(q_) + threadIdx.x);

  // Each block computes different B value
  float max_qk_acc = std::numeric_limits<float>::lowest();

  // Compute S[T_MAX] = for i in range(T): S[t] = sum(Q[d] * K[t, d])
  // Split T across warps in a block, unroll loads to expose more
  // parallelism.

  constexpr int32_t kTimeUnroll = 1;
  scalar4<scalar_t> k_loads[kTimeUnroll];

  int32_t t_max_unroll =
      (t_max / (kWarpsPerBlock * kTimeUnroll)) * (kWarpsPerBlock * kTimeUnroll);
  for (auto tt = warp_idx * kTimeUnroll; tt < t_max_unroll;
       tt += kWarpsPerBlock * kTimeUnroll) {
#pragma unroll kTimeUnroll
    for (auto ttt = 0; ttt < kTimeUnroll; ++ttt) {
      int32_t t = tt + ttt;
      auto* k_ = cache_K_base + t * cache_K.stride(1);
      // scalar4<scalar_t> k_thread;
      *reinterpret_cast<whole_int_t*>(&k_loads[ttt]) =
          *(reinterpret_cast<const whole_int_t*>(k_) + threadIdx.x);
    }
#pragma unroll kTimeUnroll
    for (auto ttt = 0; ttt < kTimeUnroll; ++ttt) {
      float qk_acc = 0;
      int32_t t = tt + ttt;
      qk_acc += scalar4_dot(q_thread, k_loads[ttt]) * qk_scale;

      qk_acc = warpReduceSum<float>(qk_acc);
      max_qk_acc = max(qk_acc, max_qk_acc);

      // write accumulated sums to smem.
      if (threadIdx.x == 0) {
        smem[t] = qk_acc;
      }
    }
  }

  constexpr int32_t kTimeUnroll1 = 1;
  for (auto tt = t_max_unroll + warp_idx; tt < t_max;
       tt += kWarpsPerBlock * kTimeUnroll1) {
#pragma unroll kTimeUnroll1
    for (auto ttt = 0; ttt < kTimeUnroll1; ++ttt) {
      int32_t t = tt + ttt;
      // &(cache_K[b][t][0][0]);
      auto* k_ = cache_K_base + t * cache_K.stride(1);
      // scalar4<scalar_t> k_thread;
      *reinterpret_cast<whole_int_t*>(&k_loads[ttt]) =
          *(reinterpret_cast<const whole_int_t*>(k_) + threadIdx.x);
    }
#pragma unroll kTimeUnroll1
    for (auto ttt = 0; ttt < kTimeUnroll1; ++ttt) {
      float qk_acc = 0;
      int32_t t = tt + ttt;
      qk_acc += scalar4_dot(q_thread, k_loads[ttt]) * qk_scale;

      qk_acc = warpReduceSum<float>(qk_acc);
      max_qk_acc = max(qk_acc, max_qk_acc);

      // write accumulated sums to smem.
      if (threadIdx.x == 0) {
        smem[t] = qk_acc;
      }
    }
  }

  // Use shared reduction to compute max and compute softmax on shared memory.
  // write max acc
  if (threadIdx.x == 0) {
    smem[T_MAX + warp_idx] = max_qk_acc;
  }
  __syncthreads();
  if (threadIdx.x < kWarpsPerBlock) {
    max_qk_acc = max(max_qk_acc, smem[T_MAX + threadIdx.x]);
  }
  // shared across all threads in block
  max_qk_acc = warpReduceMax(max_qk_acc);
  // each warp computes partial sum of exp.
  float softmax_denominator = 0.0f;
  for (int32_t t = threadIdx.x + warp_idx * kThreadsPerWarp; t < t_max;
       t += kWarpsPerBlock * kThreadsPerWarp) {
    softmax_denominator += __expf(smem[t] - max_qk_acc);
  }
  softmax_denominator = warpReduceSum(softmax_denominator);

  __syncthreads();
  if (threadIdx.x == 0) {
    smem[T_MAX + warp_idx] = softmax_denominator;
  }
  __syncthreads();
  // now, compute sum of exp(x - max(x)) over all intermediate results.
  softmax_denominator = 0.0;
  if (threadIdx.x < kWarpsPerBlock) {
    softmax_denominator = smem[T_MAX + threadIdx.x];
  }
  softmax_denominator = warpReduceSum(softmax_denominator);

  // now, compute the normalization across all threads.
  for (int32_t t = threadIdx.x + warp_idx * kThreadsPerWarp; t < t_max;
       t += kWarpsPerBlock * kThreadsPerWarp) {
    smem[t] = __expf(smem[t] - max_qk_acc) / softmax_denominator;
  }
  __syncthreads();

  // Now, we can comute the softmax and write the outputs.

  // Split T across warps in a block
  // each warp compute sum(t_subset) P[t] * V[t_subset, d]
  // outputs are of size float[D]

  float ps[kTimeUnroll];
  fx4 o_acc;
  for (auto tt = warp_idx * kTimeUnroll; tt < t_max_unroll;
       tt += kWarpsPerBlock * kTimeUnroll) {
#pragma unroll kTimeUnroll
    for (auto ttt = 0; ttt < kTimeUnroll; ++ttt) {
      int32_t t = tt + ttt;
      // &(cache_V[b][t][0][0]);
      auto* v_ = cache_V_base + t * cache_V.stride(1);
      //   scalar4<scalar_t> v_thread;
      *reinterpret_cast<whole_int_t*>(&k_loads[ttt]) =
          *(reinterpret_cast<const whole_int_t*>(v_) + threadIdx.x);
      ps[ttt] = smem[t];
    }

#pragma unroll kTimeUnroll
    for (auto ttt = 0; ttt < kTimeUnroll; ++ttt) {
      o_acc = scalar4_scale_acc(o_acc, k_loads[ttt], ps[ttt]);
    }
  }

  for (auto tt = t_max_unroll + warp_idx; tt < t_max;
       tt += kWarpsPerBlock * kTimeUnroll1) {
#pragma unroll kTimeUnroll1
    for (auto ttt = 0; ttt < kTimeUnroll1; ++ttt) {
      int32_t t = tt + ttt;
      // &(cache_V[b][t][0][0]);
      auto* v_ = cache_V_base + t * cache_V.stride(1);
      //   scalar4<scalar_t> v_thread;
      *reinterpret_cast<whole_int_t*>(&k_loads[ttt]) =
          *(reinterpret_cast<const whole_int_t*>(v_) + threadIdx.x);
      ps[ttt] = smem[t];
    }

#pragma unroll kTimeUnroll1
    for (auto ttt = 0; ttt < kTimeUnroll1; ++ttt) {
      o_acc = scalar4_scale_acc(o_acc, k_loads[ttt], ps[ttt]);
    }
  }

  // now, each thread has partial sums. Write to smem and get accumulated
  // results back.
  __syncthreads();
  *(reinterpret_cast<fx4*>(&smem[0]) + warp_idx * kThreadsPerWarp +
    threadIdx.x) = o_acc;
  __syncthreads();
  // sum up partial D rows from other warps
  if (warp_idx == 0) {
    fx4 r;
    for (int32_t w = 0; w < kWarpsPerBlock; ++w) {
      auto partial_r = *(
          reinterpret_cast<fx4*>(&smem[0]) + w * kThreadsPerWarp + threadIdx.x);
      r = fx4_acc(r, partial_r);
    }
    // write output D row
    auto* o_ = (&O[b][0][g][h][0]);
    auto bf_r = fx4_to_scalar4<scalar_t>(r);
    *(reinterpret_cast<whole_int_t*>(o_) + threadIdx.x) =
        *reinterpret_cast<const whole_int_t*>(&bf_r);
  }
#else
  printf("FATAL: kernel is for sm80+ only");
#endif // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
}

at::Tensor mqa_attn(
    at::Tensor XQ, // [B, 1, G, H, D]
    at::Tensor cache_K, // [B, T_MAX, G, H, D]
    at::Tensor cache_V, // [B, T_MAX, G, H, D]
    at::Tensor seq_positions, // [B]
    double qk_scale) {
  at::OptionalDeviceGuard guard(XQ.device());
  TORCH_CHECK(XQ.is_cuda());
  TORCH_CHECK(cache_K.is_cuda());
  TORCH_CHECK(cache_V.is_cuda());

  TORCH_CHECK(seq_positions.is_cuda());

  TORCH_CHECK(cache_K.size(1) <= T_MAX);
  TORCH_CHECK(cache_K.size(4) == D_H);

  auto O = at::empty_like(XQ);
  auto B = XQ.size(0);
  auto G = XQ.size(2);
  auto H = XQ.size(3);
  dim3 blocks(B, G, H);
  dim3 threads(kThreadsPerWarp, kWarpsPerBlock);

  int32_t smem_softmax = T_MAX * sizeof(float) + kWarpsPerBlock * sizeof(float);
  int32_t smem_output = D_H * sizeof(float) * kWarpsPerBlock;
  int32_t smem = max(smem_softmax, smem_output);

  if (XQ.scalar_type() == at::ScalarType::Half) {
    if (smem > 48 * 1024) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          mqa_attn_kernel<at::Half>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          smem));
    }
    mqa_attn_kernel<at::Half>
        <<<blocks, threads, smem, at::cuda::getCurrentCUDAStream()>>>(
            XQ.packed_accessor32<at::Half, 5, at::RestrictPtrTraits>(),
            cache_K.packed_accessor64<at::Half, 5, at::RestrictPtrTraits>(),
            cache_V.packed_accessor64<at::Half, 5, at::RestrictPtrTraits>(),
            O.packed_accessor32<at::Half, 5, at::RestrictPtrTraits>(),
            seq_positions
                .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            qk_scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else if (XQ.scalar_type() == at::ScalarType::BFloat16) {
    if (smem > 48 * 1024) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          mqa_attn_kernel<at::BFloat16>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          smem));
    }
    mqa_attn_kernel<at::BFloat16>
        <<<blocks, threads, smem, at::cuda::getCurrentCUDAStream()>>>(
            XQ.packed_accessor32<at::BFloat16, 5, at::RestrictPtrTraits>(),
            cache_K.packed_accessor64<at::BFloat16, 5, at::RestrictPtrTraits>(),
            cache_V.packed_accessor64<at::BFloat16, 5, at::RestrictPtrTraits>(),
            O.packed_accessor32<at::BFloat16, 5, at::RestrictPtrTraits>(),
            seq_positions
                .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            qk_scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    TORCH_CHECK(
        XQ.scalar_type() == at::ScalarType::Float,
        "Only supports bf16/f16/f32");
    if (smem > 48 * 1024) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          mqa_attn_kernel<float>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          smem));
    }
    mqa_attn_kernel<float>
        <<<blocks, threads, smem, at::cuda::getCurrentCUDAStream()>>>(
            XQ.packed_accessor32<float, 5, at::RestrictPtrTraits>(),
            cache_K.packed_accessor64<float, 5, at::RestrictPtrTraits>(),
            cache_V.packed_accessor64<float, 5, at::RestrictPtrTraits>(),
            O.packed_accessor32<float, 5, at::RestrictPtrTraits>(),
            seq_positions
                .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            qk_scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return O;
}

} // namespace

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::efficient_attention_forward_decoder"),
      TORCH_FN(mqa_attn));
}
