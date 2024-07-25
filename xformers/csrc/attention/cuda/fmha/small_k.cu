/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ATen.h>
#include <torch/library.h>
#include <cmath>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAGraphsUtils.cuh>

#include <curand_kernel.h>

#include "sputnik/vector_utils.h"

namespace {

template <typename integer>
constexpr __host__ __device__ inline integer ceil_div(integer n, integer m) {
  return (n + m - 1) / m;
}

template <typename scalar_t>
constexpr __host__ __device__ bool integerIsPowerOf2(scalar_t v) {
  return (v && !(v & (v - 1)));
}

template <typename scalar_t>
__device__ __forceinline__ void iMul(scalar_t x1, float4* out) {
  out[0].x *= x1;
  out[0].y *= x1;
  out[0].z *= x1;
  out[0].w *= x1;
}

template <typename scalar_t>
__device__ __forceinline__ void iMul(scalar_t x1, float2* out) {
  out[0].x *= x1;
  out[0].y *= x1;
}

template <typename scalar_t>
__device__ __forceinline__ void iMul(scalar_t x1, float* out) {
  out[0] *= x1;
}

template <typename scalar_t>
__device__ __forceinline__ void iDiv(scalar_t x1, float4* out) {
  out[0].x /= x1;
  out[0].y /= x1;
  out[0].z /= x1;
  out[0].w /= x1;
}

template <typename scalar_t>
__device__ __forceinline__ void iDiv(scalar_t x1, float2* out) {
  out[0].x /= x1;
  out[0].y /= x1;
}

template <typename scalar_t>
__device__ __forceinline__ void iDiv(scalar_t x1, float* out) {
  out[0] /= x1;
}

template <typename scalar_t>
__device__ __forceinline__ void myGpuAtomicAdd(scalar_t* address, float4 val) {
  gpuAtomicAdd(address + 0, val.x);
  gpuAtomicAdd(address + 1, val.y);
  gpuAtomicAdd(address + 2, val.z);
  gpuAtomicAdd(address + 3, val.w);
}

template <typename scalar_t>
__device__ __forceinline__ void myGpuAtomicAdd(scalar_t* address, float2 val) {
  gpuAtomicAdd(address + 0, val.x);
  gpuAtomicAdd(address + 1, val.y);
}

template <typename scalar_t>
__device__ __forceinline__ void myGpuAtomicAdd(scalar_t* address, float val) {
  gpuAtomicAdd(address, val);
}

template <typename scalar_t, int WARP_SIZE>
__device__ __forceinline__ scalar_t warpSum(scalar_t val) {
  for (int stride = WARP_SIZE / 2; stride > 0; stride >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, stride, WARP_SIZE);
  }
  return val;
}

template <typename scalar_t, int WARP_SIZE>
__device__ __forceinline__ float2 warpSum(float2 val) {
  for (int stride = WARP_SIZE / 2; stride > 0; stride >>= 1) {
    val.x += __shfl_xor_sync(0xffffffff, val.x, stride, WARP_SIZE);
    val.y += __shfl_xor_sync(0xffffffff, val.y, stride, WARP_SIZE);
  }
  return val;
}

template <typename scalar_t, int WARP_SIZE>
__device__ __forceinline__ float4 warpSum(float4 val) {
  for (int stride = WARP_SIZE / 2; stride > 0; stride >>= 1) {
    val.x += __shfl_xor_sync(0xffffffff, val.x, stride, WARP_SIZE);
    val.y += __shfl_xor_sync(0xffffffff, val.y, stride, WARP_SIZE);
    val.z += __shfl_xor_sync(0xffffffff, val.z, stride, WARP_SIZE);
    val.w += __shfl_xor_sync(0xffffffff, val.w, stride, WARP_SIZE);
  }
  return val;
}

template <typename scalar_t, int WARP_SIZE>
__device__ __forceinline__ scalar_t warpMax(scalar_t val) {
  for (int stride = WARP_SIZE / 2; stride > 0; stride >>= 1) {
    scalar_t tmp = __shfl_xor_sync(0xffffffff, val, stride, WARP_SIZE);
    val = tmp > val ? tmp : val;
  }
  return val;
}

template <typename scalar_t, typename vec_t, int kBlockSizeK, int kBlockSizeQ>
__device__ void compute_dot(
    vec_t* queries[kBlockSizeQ],
    vec_t* keys,
    scalar_t out[kBlockSizeQ][kBlockSizeK],
    int64_t K) {
  constexpr int kVecSize = sizeof(vec_t) / sizeof(scalar_t);
  scalar_t scale = 1.0 / std::sqrt(scalar_t(K));
  vec_t q_i[kBlockSizeQ];
  for (int64_t k = 0; k < K / kVecSize; k += 1) {
#pragma unroll
    for (int64_t q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
      q_i[q_item_idx] = __ldg(queries[q_item_idx] + k);
      iMul(scale, q_i + q_item_idx);
    }
#pragma unroll
    for (int64_t k_item_idx = 0; k_item_idx < kBlockSizeK; k_item_idx++) {
      vec_t k_i = keys[k + K / kVecSize * k_item_idx];
#pragma unroll
      for (int64_t q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
        sputnik::VectorCompute<vec_t>::Dot(
            q_i[q_item_idx], k_i, &out[q_item_idx][k_item_idx]);
      }
    }
  }
}

/*
struct RNGMaskGenerator {

  uint64_t seed_;
  uint64_t offset_;
  int64_t N_;
  int64_t global_offset_;
  curandStatePhilox4_32_10_t state_;

  __device__ __forceinline__ RNGMaskGenerator (at::PhiloxCudaState philox_args)
{ auto seeds = at:cuda::philox::unpack(philox_args); seed_ = std::get<0>(seeds);
    offset_ = std::get<1>(seeds);
  }

  __device__ __forceinline__ void set_sublocation(int64_t x, int64_t y) {
    int64_t total_offset = global_offset_ + x * N_ + y;
    // ideally we would use the code below, but initializing the rng
    // takes a significant portion of time. So we instead modify the seed
    // so that each thread has a different seed. This has fewer statistical
    // guarantees than by doing it properly, but is much faster
    // curand_init(seed_, total_offset, offset_, &state_);
    curand_init(seed_ + (total_offset << 8) + offset_, 0, 0, &state_);
  }

  __device__ __forceinline__ float4 generate() {
      return curand_uniform4(&state_);
  }

}
*/
template <
    typename scalar_t,
    typename vec_t,
    int kBlockSizeK,
    int kBlockSizeQ,
    int BUFFER_SIZE>
__device__ void compute_final_mult(
    vec_t* vi,
    scalar_t s_delta[kBlockSizeQ][kBlockSizeK],
    scalar_t m_delta[kBlockSizeQ],
    vec_t buffer[kBlockSizeQ][BUFFER_SIZE] /*TODO [BUFFER_SIZE limitation]*/,
    int64_t K) {
  constexpr int kVecSize = sizeof(vec_t) / sizeof(scalar_t);

  for (int64_t k = 0; k < K / kVecSize; k += 1) {
#pragma unroll
    for (int64_t q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
      iMul<scalar_t>(m_delta[q_item_idx], &buffer[q_item_idx][k]);
    }
#pragma unroll
    for (int64_t k_item_idx = 0; k_item_idx < kBlockSizeK; k_item_idx++) {
      vec_t tmp2 = vi[k + K / kVecSize * k_item_idx];

#pragma unroll
      for (int64_t q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
        sputnik::VectorCompute<vec_t>::FMA(
            s_delta[q_item_idx][k_item_idx], tmp2, &buffer[q_item_idx][k]);
      }
    }
  }
}

template <typename scalar_t, typename vec_t, int kBlockSizeK, int kBlockSizeQ>
//__device__ __forceinline__ void apply_masking(
__device__ void apply_masking(
    scalar_t s_delta[kBlockSizeQ][kBlockSizeK],
    at::PhiloxCudaState philox_args,
    int64_t global_offset,
    int64_t N,
    scalar_t p,
    int64_t col_offset) {
  // strategy: initialize the rng so that each element in the attention
  // matrix has its own subsequence, so that we can easily retrieve
  // the element during backward

  curandStatePhilox4_32_10_t state;
  auto seeds = at::cuda::philox::unpack(philox_args);

  // we will always sample 4 random floats at a time
  // as it's more efficient
  constexpr int kSampled = 4;

  // because the forward and the backward have different
  // access patterns, we round the rng offset so that it's
  // a multiple of kSampled, and add the delta needed
  int delta = col_offset & (kSampled - 1);
  col_offset = col_offset - delta;

#pragma unroll
  for (int64_t q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
#pragma unroll
    for (int64_t k_item_idx = 0; k_item_idx < kBlockSizeK;
         k_item_idx += kSampled) {
      int64_t offset = global_offset + q_item_idx * N + k_item_idx + col_offset;
      // ideally we would use the code below, but initializing the rng
      // takes a significant portion of time. So we instead modify the seed
      // so that each thread has a different seed. This has fewer statistical
      // guarantees than by doing it properly, but is much faster
      curand_init(
          std::get<0>(seeds), offset, std::get<1>(seeds) + delta, &state);
      // curand_init(std::get<0>(seeds) + (offset << 8) + std::get<1>(seeds), 0,
      // 0, &state);
      float4 rand = curand_uniform4(&state);
      for (int kk = 0; kk < kSampled; kk++) {
        if (k_item_idx + kk < kBlockSizeK)
          s_delta[q_item_idx][k_item_idx + kk] *= (&rand.x)[kk] < p;
      }
    }
  }
}

template <typename scalar_t, int kBlockSizeK, int kBlockSizeQ>
__device__ __forceinline__ void compute_max(
    scalar_t a[kBlockSizeQ][kBlockSizeK],
    scalar_t b[kBlockSizeQ],
    scalar_t out[kBlockSizeQ]) {
#pragma unroll
  for (int64_t q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
    out[q_item_idx] =
        a[q_item_idx][0] > b[q_item_idx] ? a[q_item_idx][0] : b[q_item_idx];
#pragma unroll
    for (int64_t k_item_idx = 1; k_item_idx < kBlockSizeK; k_item_idx++) {
      out[q_item_idx] = a[q_item_idx][k_item_idx] > out[q_item_idx]
          ? a[q_item_idx][k_item_idx]
          : out[q_item_idx];
    }
  }
}

template <typename scalar_t, int kBlockSizeK, int kBlockSizeQ>
__device__ __forceinline__ void compute_and_update_scaling_coeffs(
    scalar_t m_i[kBlockSizeQ],
    scalar_t m_prime[kBlockSizeQ],
    scalar_t s_prime[kBlockSizeQ],
    scalar_t si[kBlockSizeQ][kBlockSizeK],
    scalar_t m_delta[kBlockSizeQ],
    scalar_t s_delta[kBlockSizeQ][kBlockSizeK]) {
#pragma unroll
  for (int64_t q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
    m_delta[q_item_idx] = std::exp(m_prime[q_item_idx] - m_i[q_item_idx]);
    m_delta[q_item_idx] =
        isfinite(m_delta[q_item_idx]) ? m_delta[q_item_idx] : scalar_t(0);
    m_prime[q_item_idx] = m_i[q_item_idx];
    s_prime[q_item_idx] = s_prime[q_item_idx] * m_delta[q_item_idx];
#pragma unroll
    for (int64_t k_item_idx = 0; k_item_idx < kBlockSizeK; k_item_idx++) {
      s_delta[q_item_idx][k_item_idx] =
          std::exp(si[q_item_idx][k_item_idx] - m_i[q_item_idx]);
      s_delta[q_item_idx][k_item_idx] =
          isfinite(s_delta[q_item_idx][k_item_idx])
          ? s_delta[q_item_idx][k_item_idx]
          : scalar_t(0);
      s_prime[q_item_idx] += s_delta[q_item_idx][k_item_idx];
    }
  }
}

template <typename scalar_t, typename vec_t, int kBlockSizeK, int kBlockSizeQ>
__device__ void add_attn_bias(
    scalar_t si[kBlockSizeQ][kBlockSizeK],
    scalar_t* attn_bias_i) {
// TODO: use vector loads if possible
#pragma unroll
  for (int64_t q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
#pragma unroll
    for (int64_t k_item_idx = 0; k_item_idx < kBlockSizeK; k_item_idx++) {
      si[q_item_idx][k_item_idx] += attn_bias_i[k_item_idx];
    }
  }
}

template <
    typename scalar_t,
    typename vec_t,
    int kBlockSizeK,
    int kBlockSizeQ,
    int BUFFER_SIZE>
__device__ void compute_loop(
    vec_t* query_block[kBlockSizeQ],
    vec_t* key_i,
    vec_t* value_i,
    scalar_t m_prime[kBlockSizeQ],
    scalar_t s_prime[kBlockSizeQ],
    vec_t buffer[kBlockSizeQ][BUFFER_SIZE] /*TODO [BUFFER_SIZE limitation]*/,
    int64_t K,
    scalar_t* attn_bias_i,
    at::PhiloxCudaState philox_args,
    int64_t global_offset,
    int64_t N,
    scalar_t p,
    int64_t col_offset) {
  scalar_t si[kBlockSizeQ][kBlockSizeK] = {0};
  compute_dot<scalar_t, vec_t, kBlockSizeK, kBlockSizeQ>(
      query_block, key_i, si, K);

  if (attn_bias_i != nullptr) {
    add_attn_bias<scalar_t, vec_t, kBlockSizeK, kBlockSizeQ>(si, attn_bias_i);
  }

  scalar_t m_i[kBlockSizeQ];
  compute_max<scalar_t, kBlockSizeK, kBlockSizeQ>(si, m_prime, m_i);

  scalar_t m_delta[kBlockSizeQ];
  scalar_t s_delta[kBlockSizeQ][kBlockSizeK];

  compute_and_update_scaling_coeffs<scalar_t, kBlockSizeK, kBlockSizeQ>(
      m_i, m_prime, s_prime, si, m_delta, s_delta);

  if (p < 1.0)
    apply_masking<scalar_t, vec_t, kBlockSizeK, kBlockSizeQ>(
        s_delta, philox_args, global_offset, N, p, col_offset);

  compute_final_mult<scalar_t, vec_t, kBlockSizeK, kBlockSizeQ, BUFFER_SIZE>(
      value_i, s_delta, m_delta, buffer, K);
}

template <
    typename scalar_t,
    typename vec_t,
    int kBlockSizeQ,
    int WARP_SIZE,
    int BUFFER_SIZE>
__device__ __forceinline__ void aggregate_coeffs(
    scalar_t m_prime[kBlockSizeQ],
    scalar_t s_prime[kBlockSizeQ],
    vec_t buffer[kBlockSizeQ][BUFFER_SIZE] /*TODO [BUFFER_SIZE limitation]*/,
    int64_t K) {
  constexpr int kVecSize = sizeof(vec_t) / sizeof(scalar_t);
  for (int64_t q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
    scalar_t m_i = m_prime[q_item_idx];
    scalar_t s_i = s_prime[q_item_idx];
    m_prime[q_item_idx] = warpMax<scalar_t, WARP_SIZE>(m_prime[q_item_idx]);
    scalar_t m_delta = std::exp(m_i - m_prime[q_item_idx]);
    scalar_t s_delta = s_i * m_delta;
    s_delta = warpSum<scalar_t, WARP_SIZE>(s_delta);
    s_prime[q_item_idx] = s_delta;
    for (int64_t k = 0; k < K / kVecSize; k += 1) {
      vec_t tmp = buffer[q_item_idx][k];
      iMul<scalar_t>(m_delta, &tmp);
      tmp = warpSum<vec_t, WARP_SIZE>(tmp);
      buffer[q_item_idx][k] = tmp;
    }
  }
}

template <
    bool first,
    typename scalar_t,
    typename vec_t,
    int kBlockSizeK,
    int kBlockSizeQ,
    int BUFFER_SIZE,
    int WARP_SIZE>
struct UnrollLoop {
  static __device__ __forceinline__ void eval(
      vec_t* query_block[kBlockSizeQ],
      at::TensorAccessor<scalar_t, 2> key,
      at::TensorAccessor<scalar_t, 2> value,
      scalar_t m_prime[kBlockSizeQ],
      scalar_t s_prime[kBlockSizeQ],
      vec_t buffer[kBlockSizeQ][BUFFER_SIZE] /*TODO [BUFFER_SIZE limitation]*/,
      int64_t K,
      int64_t N,
      at::TensorAccessor<scalar_t, 2> attn_bias,
      at::PhiloxCudaState philox_args,
      int64_t global_offset,
      scalar_t p) {
    constexpr int64_t step = kBlockSizeK * WARP_SIZE;
    int64_t l;
    if (first) {
      l = threadIdx.x * kBlockSizeK;
    } else {
      l = N - (N & (2 * step - 1)) + threadIdx.x * kBlockSizeK;
    }
    // this is equivalent to N - N % step, but faster
    // guaranteed to be the same as step is a power of 2
    int64_t end_iter = kBlockSizeK == 1 ? N : N - (N & (step - 1));
    // if (l < end_iter) {
    {
      for (; l < end_iter; l += step) {
        auto key_i = reinterpret_cast<vec_t*>(key[l].data());
        auto value_i = reinterpret_cast<vec_t*>(value[l].data());
        auto attn_bias_i = &attn_bias[0][l];

        compute_loop<scalar_t, vec_t, kBlockSizeK, kBlockSizeQ, BUFFER_SIZE>(
            query_block,
            key_i,
            value_i,
            m_prime,
            s_prime,
            buffer,
            K,
            attn_bias_i,
            philox_args,
            global_offset,
            N,
            p,
            l);
      }
    }
    {
      UnrollLoop<
          false,
          scalar_t,
          vec_t,
          kBlockSizeK / 2,
          kBlockSizeQ,
          BUFFER_SIZE,
          WARP_SIZE>::
          eval(
              query_block,
              key,
              value,
              m_prime,
              s_prime,
              buffer,
              K,
              N,
              attn_bias,
              philox_args,
              global_offset,
              p);
    }
  }
};

template <
    bool first,
    typename scalar_t,
    typename vec_t,
    int kBlockSizeQ,
    int BUFFER_SIZE,
    int WARP_SIZE>
struct UnrollLoop<
    first,
    scalar_t,
    vec_t,
    0,
    kBlockSizeQ,
    BUFFER_SIZE,
    WARP_SIZE> {
  static __device__ __forceinline__ void eval(
      vec_t* query_block[kBlockSizeQ],
      at::TensorAccessor<scalar_t, 2> key,
      at::TensorAccessor<scalar_t, 2> value,
      scalar_t m_prime[kBlockSizeQ],
      scalar_t s_prime[kBlockSizeQ],
      vec_t buffer[kBlockSizeQ][BUFFER_SIZE] /*TODO [BUFFER_SIZE limitation]*/,
      int64_t K,
      int64_t N,
      at::TensorAccessor<scalar_t, 2> attn_bias,
      at::PhiloxCudaState philox_args,
      int64_t global_offset,
      scalar_t p) {}
};

template <
    typename scalar_t,
    typename vec_t,
    int kBlockSizeK,
    int kBlockSizeQ,
    int WARP_SIZE,
    int BUFFER_SIZE,
    bool compute_logsumexp>
__global__ void attention_kernel(
    at::PackedTensorAccessor<scalar_t, 3> output,
    at::PackedTensorAccessor<scalar_t, 2> logsumexp,
    at::PackedTensorAccessor<scalar_t, 3> query,
    at::PackedTensorAccessor<scalar_t, 3> key,
    at::PackedTensorAccessor<scalar_t, 3> value,
    at::PackedTensorAccessor<scalar_t, 3> attn_bias,
    scalar_t p,
    at::PhiloxCudaState philox_args) {
  constexpr int kVecSize = sizeof(vec_t) / sizeof(scalar_t);
  static_assert(
      integerIsPowerOf2(kBlockSizeK * WARP_SIZE),
      "kBlockSizeK * WARP_SIZE should be a power of 2");
  int64_t K = query.size(2);
  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);

  int64_t batch_idx = blockIdx.y;
  int64_t query_idx =
      blockIdx.x * (blockDim.y * kBlockSizeQ) + threadIdx.y * kBlockSizeQ;

  int64_t global_offset = batch_idx * M * N + query_idx * N;

  if (query_idx >= M)
    return;

  vec_t* query_block[kBlockSizeQ];
  vec_t* output_block[kBlockSizeQ];
  scalar_t* logsumexp_block[kBlockSizeQ];
  // TODO [BUFFER_SIZE limitation]: the current strategy assumes a
  // statically-known size for K. Ideally we would like to remove this
  // limitation in the future, so that any K is supported
  vec_t buffer[kBlockSizeQ][BUFFER_SIZE] = {};
  scalar_t s_prime[kBlockSizeQ] = {0};
  scalar_t m_prime[kBlockSizeQ];
  for (int64_t q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
    int64_t index = query_idx + q_item_idx;
    index = index >= M ? M - 1 : index;
    query_block[q_item_idx] =
        reinterpret_cast<vec_t*>(query[batch_idx][index].data());
    output_block[q_item_idx] =
        reinterpret_cast<vec_t*>(output[batch_idx][index].data());
    m_prime[q_item_idx] = -std::numeric_limits<scalar_t>::infinity();
    logsumexp_block[q_item_idx] = &logsumexp[batch_idx][index];
  }

  // Computes s_prime, buffer (aka v_prime) and m_prime
  UnrollLoop<
      true,
      scalar_t,
      vec_t,
      kBlockSizeK,
      kBlockSizeQ,
      BUFFER_SIZE,
      WARP_SIZE>::
      eval(
          query_block,
          key[batch_idx],
          value[batch_idx],
          m_prime,
          s_prime,
          buffer,
          K,
          N,
          attn_bias[batch_idx],
          philox_args,
          global_offset,
          p);

  aggregate_coeffs<scalar_t, vec_t, kBlockSizeQ, WARP_SIZE, BUFFER_SIZE>(
      m_prime, s_prime, buffer, K);

  for (int64_t k = threadIdx.x; k < K / kVecSize; k += blockDim.x) {
    vec_t tmp;

#pragma unroll
    for (int64_t q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
      tmp = buffer[q_item_idx][k];
      iDiv<scalar_t>(s_prime[q_item_idx] * p, &tmp);

      if (query_idx + q_item_idx < M)
        output_block[q_item_idx][k] = tmp;
    }
  }

  if (compute_logsumexp) {
#pragma unroll
    for (int64_t q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
      *logsumexp_block[q_item_idx] =
          m_prime[q_item_idx] + std::log(s_prime[q_item_idx]);
    }
  }
}

template <typename scalar_t>
at::PackedTensorAccessor<scalar_t, 3> _packed_tensor_accessor_or_dummy(
    const at::Tensor& attn_bias) {
  if (attn_bias.defined()) {
    return attn_bias.packed_accessor64<scalar_t, 3>();
  } else {
    const std::array<int64_t, 3> zeros{{0}};
    return at::PackedTensorAccessor<scalar_t, 3>(
        nullptr, zeros.data(), zeros.data());
  }
}

template <bool compute_logsumexp>
void launch_attention(
    at::Tensor& res,
    at::Tensor& logsumexp,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& attn_bias,
    float p,
    at::PhiloxCudaState rng_engine_inputs) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);
  int64_t K = query.size(2);

  constexpr int WARP_SIZE = 4;

  constexpr int kBlockSizeK = 32;
  constexpr int kBlockSizeQ = 2;

  constexpr int TILE_SIZE = 32;
  constexpr int BUFFER_SIZE = 8;

  dim3 grid(ceil_div(M, int64_t(TILE_SIZE)), B);
  dim3 block(WARP_SIZE, TILE_SIZE / kBlockSizeQ);

  if (grid.x * grid.y * grid.z == 0 || key.numel() == 0) {
    res.zero_();
    return;
  }

  using scalar_t = float;

  auto attn_bias_packed = _packed_tensor_accessor_or_dummy<scalar_t>(attn_bias);

  if ((K % 4) == 0) {
    TORCH_CHECK(
        K / 4 <= BUFFER_SIZE,
        "For now only a certain number of K values are supported. Let us know if you hit this and we will fix it");
    attention_kernel<
        scalar_t,
        float4,
        kBlockSizeK,
        kBlockSizeQ,
        WARP_SIZE,
        BUFFER_SIZE,
        compute_logsumexp><<<grid, block, 0, stream>>>(
        res.packed_accessor64<scalar_t, 3>(),
        logsumexp.packed_accessor64<scalar_t, 2>(),
        query.packed_accessor64<scalar_t, 3>(),
        key.packed_accessor64<scalar_t, 3>(),
        value.packed_accessor64<scalar_t, 3>(),
        attn_bias_packed,
        p,
        rng_engine_inputs);
  } else if ((K % 2) == 0) {
    TORCH_CHECK(
        K / 2 <= BUFFER_SIZE,
        "For now only a certain number of K values are supported. Let us know if you hit this and we will fix it");
    attention_kernel<
        scalar_t,
        float2,
        kBlockSizeK,
        kBlockSizeQ,
        WARP_SIZE,
        BUFFER_SIZE,
        compute_logsumexp><<<grid, block, 0, stream>>>(
        res.packed_accessor64<scalar_t, 3>(),
        logsumexp.packed_accessor64<scalar_t, 2>(),
        query.packed_accessor64<scalar_t, 3>(),
        key.packed_accessor64<scalar_t, 3>(),
        value.packed_accessor64<scalar_t, 3>(),
        attn_bias_packed,
        p,
        rng_engine_inputs);

  } else {
    TORCH_CHECK(
        K <= BUFFER_SIZE,
        "For now only a certain number of K values are supported. Let us know if you hit this and we will fix it");
    attention_kernel<
        scalar_t,
        float,
        kBlockSizeK,
        kBlockSizeQ,
        WARP_SIZE,
        BUFFER_SIZE,
        compute_logsumexp><<<grid, block, 0, stream>>>(
        res.packed_accessor64<scalar_t, 3>(),
        logsumexp.packed_accessor64<scalar_t, 2>(),
        query.packed_accessor64<scalar_t, 3>(),
        key.packed_accessor64<scalar_t, 3>(),
        value.packed_accessor64<scalar_t, 3>(),
        attn_bias_packed,
        p,
        rng_engine_inputs);
  }
}

std::tuple<at::Tensor, at::Tensor, int64_t, int64_t> attention(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    bool compute_logsumexp,
    const std::optional<at::Tensor>& attn_bias_,
    double p) {
  TORCH_CHECK(query.dim() == key.dim());
  TORCH_CHECK(query.dim() == value.dim());
  TORCH_CHECK(query.dim() == 3);
  TORCH_CHECK(query.size(2) == key.size(2));
  TORCH_CHECK(query.size(0) == key.size(0));

  at::Tensor attn_bias;
  if (attn_bias_.has_value()) {
    attn_bias = *attn_bias_;
    TORCH_CHECK(query.dim() == attn_bias.dim());
    TORCH_CHECK(query.size(0) == attn_bias.size(0));
    TORCH_CHECK(query.size(1) == attn_bias.size(1));
    TORCH_CHECK(key.size(1) == attn_bias.size(2));
    TORCH_CHECK(attn_bias.stride(1) == 0);
  }

  TORCH_CHECK(query.size(0) == value.size(0));
  TORCH_CHECK(key.size(1) == value.size(1));
  TORCH_CHECK(
      query.size(2) ==
      value.size(2)); // TODO: drop this limitation in the future

  TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
  TORCH_CHECK(key.is_cuda(), "key must be a CUDA tensor");
  TORCH_CHECK(value.is_cuda(), "value must be a CUDA tensor");

  TORCH_CHECK(!query.is_sparse(), "query must be a dense tensor");
  TORCH_CHECK(!key.is_sparse(), "key must be a dense tensor");
  TORCH_CHECK(!value.is_sparse(), "value must be a dense tensor");

  // TODO drop this limitation in the future
  TORCH_CHECK(query.is_contiguous());
  TORCH_CHECK(key.is_contiguous());
  TORCH_CHECK(value.is_contiguous());

  // TODO: support other dtypes in the future
  TORCH_CHECK(
      query.scalar_type() == at::ScalarType::Float,
      "Only float32 type is supported for now");

  at::cuda::CUDAGuard device_guard(query.device());

  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);
  int64_t K = query.size(2);

  at::Tensor res = at::zeros({B, M, K}, query.options());
  at::Tensor logsumexp = at::empty({B, M}, query.options());

  // invert from drop probability to keep probability
  p = 1.0 - p;

  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());

  at::PhiloxCudaState rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    // each element in the attention matrix will have its own subsequence
    // in the generator, so the offset is 1 globally
    // int64_t counter_offset = p > 0 ? 1 : 0;
    int64_t counter_offset = p > 0 ? 4 : 0;
    rng_engine_inputs = gen->philox_cuda_state(counter_offset);
  }

  // have to pass compute_logsumexp as a template parameter
  // otherwise there is a slowdown in the kernel...
  if (compute_logsumexp) {
    launch_attention<true>(
        res, logsumexp, query, key, value, attn_bias, p, rng_engine_inputs);
  } else {
    launch_attention<false>(
        res, logsumexp, query, key, value, attn_bias, p, rng_engine_inputs);
  }

  AT_CUDA_CHECK(cudaGetLastError());

  // uint64_t -> int64_t bitwise casting as PyTorch don't support uint64_t
  // so just fake it as a int64_t
  int64_t seed, offset;
  std::memcpy(&seed, &rng_engine_inputs.seed_, sizeof(seed));
  std::memcpy(&offset, &rng_engine_inputs.offset_.val, sizeof(offset));

  return std::make_tuple(res, logsumexp, seed, offset);
}

template <
    typename scalar_t,
    typename vec_t,
    int kBlockSizeQ,
    int kBlockSizeK,
    int TILE_SIZEQ,
    int TILE_SIZEK,
    bool check_bounds>
__global__ void attention_backward_kernel(
    at::PackedTensorAccessor<scalar_t, 3> grad_q,
    at::PackedTensorAccessor<scalar_t, 3> grad_k,
    at::PackedTensorAccessor<scalar_t, 3> grad_v,
    at::PackedTensorAccessor<scalar_t, 3> grad_out,
    at::PackedTensorAccessor<scalar_t, 3> query,
    at::PackedTensorAccessor<scalar_t, 3> key,
    at::PackedTensorAccessor<scalar_t, 3> value,
    at::PackedTensorAccessor<scalar_t, 3> output,
    at::PackedTensorAccessor<scalar_t, 2> logsumexp_normalizer,
    at::PackedTensorAccessor<scalar_t, 3> attn_bias,
    scalar_t p,
    at::PhiloxCudaState philox_args) {
  int64_t K = query.size(2);
  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);

  constexpr int kVecSize = sizeof(vec_t) / sizeof(scalar_t);

  int64_t batch_idx = blockIdx.z;
  int64_t query_idx =
      blockIdx.x * blockDim.x * kBlockSizeQ + threadIdx.x * kBlockSizeQ;
  int64_t l = blockIdx.y * blockDim.y * kBlockSizeK + threadIdx.y * kBlockSizeK;

  __shared__ scalar_t fact[TILE_SIZEQ][TILE_SIZEK + 1];

#pragma unroll
  for (int k_item_idx = 0; k_item_idx < kBlockSizeK; k_item_idx++) {
#pragma unroll
    for (int q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
      fact[kBlockSizeQ * threadIdx.x + q_item_idx]
          [kBlockSizeK * threadIdx.y + k_item_idx] = 0;
    }
  }

  scalar_t normalizer[kBlockSizeQ];
  scalar_t tmp_sum[kBlockSizeQ] = {};
  scalar_t attn_b[kBlockSizeK];

  vec_t *qb[kBlockSizeQ], *kb[kBlockSizeK], *vb[kBlockSizeK], *gb[kBlockSizeQ],
      *qbb[TILE_SIZEQ], *kbb[TILE_SIZEK], *gbb[TILE_SIZEQ];
  scalar_t maskQ[kBlockSizeQ], maskK[kBlockSizeK];

  for (int k_item_idx = 0; k_item_idx < kBlockSizeK; k_item_idx++) {
    int64_t index = l + k_item_idx;
    maskK[k_item_idx] = index >= N ? scalar_t(0) : scalar_t(1);
    if (check_bounds)
      index = min(index, N - 1);
    kb[k_item_idx] = reinterpret_cast<vec_t*>(key[batch_idx][index].data());
    vb[k_item_idx] = reinterpret_cast<vec_t*>(value[batch_idx][index].data());
    attn_b[k_item_idx] = attn_bias.data() == nullptr
        ? scalar_t(0)
        : attn_bias[batch_idx][0][index];
    ;
  }

  for (int q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
    int64_t index = query_idx + q_item_idx;
    maskQ[q_item_idx] = index >= M ? scalar_t(0) : scalar_t(1);
    if (check_bounds)
      index = min(index, M - 1);
    qb[q_item_idx] = reinterpret_cast<vec_t*>(query[batch_idx][index].data());
    gb[q_item_idx] =
        reinterpret_cast<vec_t*>(grad_out[batch_idx][index].data());
  }
  for (int64_t i = 0; i < TILE_SIZEQ; i++) {
    int64_t index = query_idx + i - kBlockSizeQ * threadIdx.x;
    if (check_bounds)
      index = min(index, M - 1);
    qbb[i] = reinterpret_cast<vec_t*>(query[batch_idx][index].data());
    gbb[i] = reinterpret_cast<vec_t*>(grad_out[batch_idx][index].data());
  }

  for (int64_t i = 0; i < TILE_SIZEK; i++) {
    int64_t index = l + i - kBlockSizeK * threadIdx.y;
    if (check_bounds)
      index = min(index, N - 1);
    kbb[i] = reinterpret_cast<vec_t*>(key[batch_idx][index].data());
  }

  for (int i = 0; i < kBlockSizeQ; i++) {
    int64_t index = query_idx + i;
    if (check_bounds)
      index = min(index, M - 1);
    normalizer[i] = logsumexp_normalizer[batch_idx][index];
  }

  for (int q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
    int64_t index = query_idx + q_item_idx;
    if (index >= M)
      break;

    auto out_i = reinterpret_cast<vec_t*>(output[batch_idx][index].data());
    auto grad_out_i =
        reinterpret_cast<vec_t*>(grad_out[batch_idx][index].data());
    for (int64_t k = 0; k < K / kVecSize; k += 1) {
      vec_t kk = __ldg(grad_out_i + k);
      vec_t tt = __ldg(out_i + k);
      sputnik::VectorCompute<vec_t>::Dot(kk, tt, tmp_sum + q_item_idx);
    }
  }

  scalar_t attn_v[kBlockSizeQ][kBlockSizeK] = {0};
  scalar_t grad_attn_v[kBlockSizeQ][kBlockSizeK] = {0};
  scalar_t scale = 1.0 / std::sqrt(scalar_t(K));

  for (int64_t k = 0; k < K / kVecSize; k += 1) {
#pragma unroll
    for (int k_item_idx = 0; k_item_idx < kBlockSizeK; k_item_idx++) {
      vec_t kk = __ldg(kb[k_item_idx] + k);
      iMul(scale, &kk);
      vec_t tt = __ldg(vb[k_item_idx] + k);
#pragma unroll
      for (int q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
        sputnik::VectorCompute<vec_t>::Dot(
            __ldg(qb[q_item_idx] + k), kk, &attn_v[q_item_idx][k_item_idx]);
        sputnik::VectorCompute<vec_t>::Dot(
            __ldg(gb[q_item_idx] + k),
            tt,
            &grad_attn_v[q_item_idx][k_item_idx]);
      }
    }
  }
  scalar_t one_over_p = 1.0 / p;
#pragma unroll
  for (int k_item_idx = 0; k_item_idx < kBlockSizeK; k_item_idx++) {
#pragma unroll
    for (int q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
      attn_v[q_item_idx][k_item_idx] =
          std::exp(
              attn_v[q_item_idx][k_item_idx] - normalizer[q_item_idx] +
              attn_b[k_item_idx]) *
          maskQ[q_item_idx] * maskK[k_item_idx];
    }
  }

  scalar_t mask[kBlockSizeQ][kBlockSizeK];
#pragma unroll
  for (int q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
#pragma unroll
    for (int k_item_idx = 0; k_item_idx < kBlockSizeK; k_item_idx++) {
      mask[q_item_idx][k_item_idx] = 1;
    }
  }

  if (p < 1.0) {
    int64_t global_offset = batch_idx * M * N + query_idx * N;
    apply_masking<scalar_t, vec_t, kBlockSizeK, kBlockSizeQ>(
        mask, philox_args, global_offset, N, p, l);
  }

#pragma unroll
  for (int k_item_idx = 0; k_item_idx < kBlockSizeK; k_item_idx++) {
#pragma unroll
    for (int q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
      fact[kBlockSizeQ * threadIdx.x + q_item_idx]
          [kBlockSizeK * threadIdx.y + k_item_idx] =
              attn_v[q_item_idx][k_item_idx] * mask[q_item_idx][k_item_idx] *
          one_over_p;
    }
  }
  __syncthreads();

  for (int64_t k = threadIdx.x; k < K / kVecSize; k += blockDim.x) {
    vec_t res[kBlockSizeK] = {0};
#pragma unroll
    for (int64_t i = 0; i < TILE_SIZEQ; i++) {
      vec_t kk = __ldg(gbb[i] + k);
#pragma unroll
      for (int k_item_idx = 0; k_item_idx < kBlockSizeK; k_item_idx++) {
        sputnik::VectorCompute<vec_t>::FMA(
            fact[i][kBlockSizeK * threadIdx.y + k_item_idx],
            kk,
            &res[k_item_idx]);
      }
    }
#pragma unroll
    for (int k_item_idx = 0; k_item_idx < kBlockSizeK; k_item_idx++) {
      int64_t index = l + k_item_idx;
      if (check_bounds)
        index = min(index, N - 1);
      myGpuAtomicAdd(&grad_v[batch_idx][index][k * kVecSize], res[k_item_idx]);
    }
  }
  __syncthreads();

#pragma unroll
  for (int k_item_idx = 0; k_item_idx < kBlockSizeK; k_item_idx++) {
#pragma unroll
    for (int q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
      fact[kBlockSizeQ * threadIdx.x + q_item_idx]
          [kBlockSizeK * threadIdx.y + k_item_idx] =
              attn_v[q_item_idx][k_item_idx] * scale *
          (grad_attn_v[q_item_idx][k_item_idx] * one_over_p *
               mask[q_item_idx][k_item_idx] -
           tmp_sum[q_item_idx]);
    }
  }
  __syncthreads();

  for (int64_t k = threadIdx.y; k < K / kVecSize; k += blockDim.y) {
    vec_t res[kBlockSizeQ] = {0};
#pragma unroll
    for (int64_t i = 0; i < TILE_SIZEK; i++) {
      vec_t kk = __ldg(kbb[i] + k);
#pragma unroll
      for (int q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
        sputnik::VectorCompute<vec_t>::FMA(
            fact[kBlockSizeQ * threadIdx.x + q_item_idx][i],
            kk,
            &res[q_item_idx]);
      }
    }
#pragma unroll
    for (int q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
      int64_t index = query_idx + q_item_idx;
      if (check_bounds)
        index = min(index, M - 1);
      myGpuAtomicAdd(&grad_q[batch_idx][index][k * kVecSize], res[q_item_idx]);
    }
  }

  for (int64_t k = threadIdx.x; k < K / kVecSize; k += blockDim.x) {
    vec_t res[kBlockSizeK] = {0};
#pragma unroll
    for (int64_t i = 0; i < TILE_SIZEQ; i++) {
      vec_t kk = __ldg(qbb[i] + k);
#pragma unroll
      for (int k_item_idx = 0; k_item_idx < kBlockSizeK; k_item_idx++) {
        sputnik::VectorCompute<vec_t>::FMA(
            fact[i][kBlockSizeK * threadIdx.y + k_item_idx],
            kk,
            &res[k_item_idx]);
      }
    }
#pragma unroll
    for (int k_item_idx = 0; k_item_idx < kBlockSizeK; k_item_idx++) {
      int64_t index = l + k_item_idx;
      if (check_bounds)
        index = min(index, N - 1);
      myGpuAtomicAdd(&grad_k[batch_idx][index][k * kVecSize], res[k_item_idx]);
    }
  }
}

template <typename scalar_t, typename vec_t>
void launch_attention_backward(
    at::Tensor& grad_q,
    at::Tensor& grad_k,
    at::Tensor& grad_v,
    const at::Tensor& grad_out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& logsumexp,
    const at::Tensor& output,
    const at::Tensor& attn_bias,
    float p,
    at::PhiloxCudaState rng_engine_inputs) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto attn_bias_packed = _packed_tensor_accessor_or_dummy<scalar_t>(attn_bias);

  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);

  constexpr int TILE_SIZEQ = 32;
  constexpr int TILE_SIZEK = 32;

  constexpr int64_t kBlockSizeQ = 4;
  constexpr int64_t kBlockSizeK = 8;

  dim3 grid(
      ceil_div(M, int64_t(TILE_SIZEQ)), ceil_div(N, int64_t(TILE_SIZEK)), B);
  dim3 block(TILE_SIZEQ / kBlockSizeQ, TILE_SIZEK / kBlockSizeK);
  if (grid.x * grid.y * grid.z == 0) {
    return;
  }

  // the bounds checking in device code is very expensive, making the code
  // around 25% slower. So let's skip those checks if possible.
  if ((M % TILE_SIZEQ == 0) && (N % TILE_SIZEK == 0)) {
    attention_backward_kernel<
        scalar_t,
        vec_t,
        kBlockSizeQ,
        kBlockSizeK,
        TILE_SIZEQ,
        TILE_SIZEK,
        false><<<grid, block, 0, stream>>>(
        grad_q.packed_accessor64<scalar_t, 3>(),
        grad_k.packed_accessor64<scalar_t, 3>(),
        grad_v.packed_accessor64<scalar_t, 3>(),
        grad_out.packed_accessor64<scalar_t, 3>(),
        query.packed_accessor64<scalar_t, 3>(),
        key.packed_accessor64<scalar_t, 3>(),
        value.packed_accessor64<scalar_t, 3>(),
        output.packed_accessor64<scalar_t, 3>(),
        logsumexp.packed_accessor64<scalar_t, 2>(),
        attn_bias_packed,
        p,
        rng_engine_inputs);
  } else {
    attention_backward_kernel<
        scalar_t,
        vec_t,
        kBlockSizeQ,
        kBlockSizeK,
        TILE_SIZEQ,
        TILE_SIZEK,
        true><<<grid, block, 0, stream>>>(
        grad_q.packed_accessor64<scalar_t, 3>(),
        grad_k.packed_accessor64<scalar_t, 3>(),
        grad_v.packed_accessor64<scalar_t, 3>(),
        grad_out.packed_accessor64<scalar_t, 3>(),
        query.packed_accessor64<scalar_t, 3>(),
        key.packed_accessor64<scalar_t, 3>(),
        value.packed_accessor64<scalar_t, 3>(),
        output.packed_accessor64<scalar_t, 3>(),
        logsumexp.packed_accessor64<scalar_t, 2>(),
        attn_bias_packed,
        p,
        rng_engine_inputs);
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> attention_backward(
    const at::Tensor& grad_out_,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& logsumexp,
    const at::Tensor& output,
    const std::optional<at::Tensor>& attn_bias_,
    double p,
    int64_t rng_seed,
    int64_t rng_offset) {
  TORCH_CHECK(query.dim() == grad_out_.dim());
  TORCH_CHECK(query.dim() == key.dim());
  TORCH_CHECK(query.dim() == value.dim());
  TORCH_CHECK(query.dim() == 3);

  TORCH_CHECK(query.size(0) == grad_out_.size(0));
  TORCH_CHECK(query.size(1) == grad_out_.size(1));
  TORCH_CHECK(query.size(2) == grad_out_.size(2));

  TORCH_CHECK(query.size(2) == key.size(2));
  TORCH_CHECK(query.size(0) == key.size(0));

  TORCH_CHECK(query.size(0) == value.size(0));
  TORCH_CHECK(key.size(1) == value.size(1));
  TORCH_CHECK(
      query.size(2) ==
      value.size(2)); // TODO: drop this limitation in the future

  at::Tensor attn_bias;
  if (attn_bias_.has_value()) {
    attn_bias = *attn_bias_;
    TORCH_CHECK(query.dim() == attn_bias.dim());
    TORCH_CHECK(query.size(0) == attn_bias.size(0));
    TORCH_CHECK(query.size(1) == attn_bias.size(1));
    TORCH_CHECK(key.size(1) == attn_bias.size(2));
    TORCH_CHECK(attn_bias.stride(1) == 0);
  }

  TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
  TORCH_CHECK(key.is_cuda(), "key must be a CUDA tensor");
  TORCH_CHECK(value.is_cuda(), "value must be a CUDA tensor");
  TORCH_CHECK(grad_out_.is_cuda(), "grad_out must be a CUDA tensor");

  TORCH_CHECK(!query.is_sparse(), "query must be a dense tensor");
  TORCH_CHECK(!key.is_sparse(), "key must be a dense tensor");
  TORCH_CHECK(!value.is_sparse(), "value must be a dense tensor");
  TORCH_CHECK(!grad_out_.is_sparse(), "grad_out must be a dense tensor");

  // TODO drop this limitation in the future
  TORCH_CHECK(query.is_contiguous());
  TORCH_CHECK(key.is_contiguous());
  TORCH_CHECK(value.is_contiguous());

  // TODO: support other dtypes in the future
  TORCH_CHECK(
      query.scalar_type() == at::ScalarType::Float,
      "Only float32 type is supported for now");

  at::cuda::CUDAGuard device_guard(query.device());

  // handle potentially non-contiguous grad_out through a copy
  auto grad_out = grad_out_.contiguous();

  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);
  int64_t K = query.size(2);

  at::Tensor grad_q = at::zeros_like(query);
  at::Tensor grad_k = at::zeros_like(key);
  at::Tensor grad_v = at::zeros_like(value);

  // invert from drop probability to keep probability
  p = 1.0 - p;

  // using scalar_t = float;
  // using vec_t = float4;
  // using vec_t = float;

  // get the state where we are supposed to be for the rng
  // in orther to sample the same dropout elements
  uint64_t seed, offset;
  std::memcpy(&seed, &rng_seed, sizeof(seed));
  std::memcpy(&offset, &rng_offset, sizeof(offset));
  at::PhiloxCudaState rng_engine_inputs(seed, offset);

  if ((K % 4) == 0) {
    launch_attention_backward<float, float4>(
        grad_q,
        grad_k,
        grad_v,
        grad_out,
        query,
        key,
        value,
        logsumexp,
        output,
        attn_bias,
        p,
        rng_engine_inputs);
  } else if ((K % 2) == 0) {
    launch_attention_backward<float, float2>(
        grad_q,
        grad_k,
        grad_v,
        grad_out,
        query,
        key,
        value,
        logsumexp,
        output,
        attn_bias,
        p,
        rng_engine_inputs);
  } else {
    launch_attention_backward<float, float>(
        grad_q,
        grad_k,
        grad_v,
        grad_out,
        query,
        key,
        value,
        logsumexp,
        output,
        attn_bias,
        p,
        rng_engine_inputs);
  }

  AT_CUDA_CHECK(cudaGetLastError());

  return std::make_tuple(grad_q, grad_k, grad_v);
}

// the functions below are only used for testing
// there is a lot of repetition compared to
// the forward code, so this could be refactored
// in the future

template <
    bool first,
    typename scalar_t,
    typename vec_t,
    int kBlockSizeK,
    int kBlockSizeQ,
    int WARP_SIZE>
struct UnrollLoopForMask {
  static __device__ __forceinline__ void eval(
      scalar_t* output[kBlockSizeQ],
      int64_t N,
      int64_t M,
      at::PhiloxCudaState philox_args,
      int64_t global_offset,
      scalar_t p) {
    constexpr int64_t step = kBlockSizeK * WARP_SIZE;
    int64_t l;
    if (first) {
      l = threadIdx.x * kBlockSizeK;
    } else {
      l = N - (N & (2 * step - 1)) + threadIdx.x * kBlockSizeK;
    }
    // this is equivalent to N - N % step, but faster
    // guaranteed to be the same as step is a power of 2
    int64_t end_iter = kBlockSizeK == 1 ? N : N - (N & (step - 1));
    scalar_t s_delta[kBlockSizeQ][kBlockSizeK];
    int64_t query_idx =
        blockIdx.x * (blockDim.y * kBlockSizeQ) + threadIdx.y * kBlockSizeQ;
    // if (l < end_iter) {
    {
      for (; l < end_iter; l += step) {
        for (int jj = 0; jj < kBlockSizeQ; jj++) {
          for (int kk = 0; kk < kBlockSizeK; kk++) {
            s_delta[jj][kk] = 1;
          }
        }

        apply_masking<scalar_t, vec_t, kBlockSizeK, kBlockSizeQ>(
            s_delta, philox_args, global_offset, N, p, l);

        for (int jj = 0; jj < kBlockSizeQ; jj++) {
          for (int kk = 0; kk < kBlockSizeK; kk++) {
            if (query_idx + jj < M)
              output[jj][l + kk] = s_delta[jj][kk];
          }
        }
      }
    }
    {
      UnrollLoopForMask<
          false,
          scalar_t,
          vec_t,
          kBlockSizeK / 2,
          kBlockSizeQ,
          WARP_SIZE>::eval(output, N, M, philox_args, global_offset, p);
    }
  }
};

template <
    bool first,
    typename scalar_t,
    typename vec_t,
    int kBlockSizeQ,
    int WARP_SIZE>
struct UnrollLoopForMask<first, scalar_t, vec_t, 0, kBlockSizeQ, WARP_SIZE> {
  static __device__ __forceinline__ void eval(
      scalar_t* s_delta[kBlockSizeQ],
      int64_t N,
      int64_t M,
      at::PhiloxCudaState philox_args,
      int64_t global_offset,
      scalar_t p) {}
};

template <
    typename scalar_t,
    typename vec_t,
    int kBlockSizeK,
    int kBlockSizeQ,
    int WARP_SIZE>
__global__ void dropout_kernel(
    at::PackedTensorAccessor<scalar_t, 3> output,
    scalar_t p,
    at::PhiloxCudaState philox_args) {
  static_assert(
      integerIsPowerOf2(kBlockSizeK * WARP_SIZE),
      "kBlockSizeK * WARP_SIZE should be a power of 2");
  int64_t B = output.size(0);
  int64_t M = output.size(1);
  int64_t N = output.size(2);

  int64_t batch_idx = blockIdx.y;
  int64_t query_idx =
      blockIdx.x * (blockDim.y * kBlockSizeQ) + threadIdx.y * kBlockSizeQ;

  int64_t global_offset = batch_idx * M * N + query_idx * N;

  if (query_idx >= M)
    return;

  scalar_t* output_block[kBlockSizeQ];
  for (int64_t q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
    int64_t index = query_idx + q_item_idx;
    index = index >= M ? M - 1 : index;
    output_block[q_item_idx] = output[batch_idx][index].data();
  }

  UnrollLoopForMask<
      true,
      scalar_t,
      vec_t,
      kBlockSizeK,
      kBlockSizeQ,
      WARP_SIZE>::eval(output_block, N, M, philox_args, global_offset, p);
}

at::Tensor _dropout_mask(at::Tensor output, double p) {
  at::cuda::CUDAGuard device_guard(output.device());
  int64_t B = output.size(0);
  int64_t M = output.size(1);
  int64_t N = output.size(2);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  constexpr int WARP_SIZE = 4;

  constexpr int kBlockSizeK = 32;
  constexpr int kBlockSizeQ = 2;

  constexpr int TILE_SIZE = 32;

  dim3 grid(ceil_div(M, int64_t(TILE_SIZE)), B);
  dim3 block(WARP_SIZE, TILE_SIZE / kBlockSizeQ);

  using scalar_t = float;

  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());

  at::PhiloxCudaState rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    // each element in the attention matrix will have its own subsequence
    // in the generator, so the offset is 1 globally
    // int64_t counter_offset = p > 0 ? 1 : 0;
    int64_t counter_offset = p > 0 ? 4 : 0;
    rng_engine_inputs = gen->philox_cuda_state(counter_offset);
  }

  // invert from drop probability to keep probability
  p = 1.0 - p;

  if (grid.x * grid.y * grid.z > 0) {
    dropout_kernel<scalar_t, scalar_t, kBlockSizeK, kBlockSizeQ, WARP_SIZE>
        <<<grid, block, 0, stream>>>(
            output.packed_accessor64<scalar_t, 3>(), p, rng_engine_inputs);
  }
  return output;
}

} // namespace

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::efficient_attention_forward_small_k"),
      TORCH_FN(attention));
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::efficient_attention_backward_small_k"),
      TORCH_FN(attention_backward));
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::_temp_dropout"), TORCH_FN(_dropout_mask));
}
