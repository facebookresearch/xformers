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

template <
    typename scalar_t,
    int kBlockWidth,
    int kBlockItemsY,
    int kBlockItemsX,
    int kBlockItemsK>
__global__ void attention_kernel(
    at::PackedTensorAccessor<scalar_t, 3> output,
    at::PackedTensorAccessor<scalar_t, 3> query,
    at::PackedTensorAccessor<scalar_t, 3> key,
    at::PackedTensorAccessor<scalar_t, 3> value) {
  int64_t K = query.size(2);
  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);

  int64_t batch_idx = blockIdx.y;
  int64_t query_idx = blockIdx.x * kBlockItemsY + threadIdx.y;

  if (query_idx >= M)
    return;

  using vec_t = float4;

  constexpr int kVecSize = sizeof(vec_t) / sizeof(scalar_t);
  constexpr int kThreadItemsK = kBlockItemsK / kBlockWidth / kVecSize;

  int KK = K / kVecSize;

  scalar_t s_prime = 0;
  scalar_t m_prime = -std::numeric_limits<scalar_t>::infinity();

  vec_t lhs_fragment[kBlockItemsK / kBlockWidth / kVecSize];
  vec_t rhs_fragment[kBlockItemsK * kBlockItemsX / kBlockWidth / kVecSize];

  auto out_i = reinterpret_cast<vec_t*>(output[batch_idx][query_idx].data());

  scalar_t scale = 1.0 / std::sqrt(scalar_t(K));

  for (int kv_idx = 0; kv_idx < N; kv_idx += kBlockItemsX) {
    scalar_t attn_fragment[kBlockItemsX] = {};

    auto query_i = reinterpret_cast<vec_t*>(query[batch_idx][query_idx].data());
    auto kkey_j = reinterpret_cast<vec_t*>(key[batch_idx].data());

    for (int k = K; k >= kBlockItemsK; k -= kBlockItemsK) {
      // load queries
#pragma unroll
      for (int k_item_idx = 0; k_item_idx < kThreadItemsK; ++k_item_idx) {
        lhs_fragment[k_item_idx] = __ldg(query_i + threadIdx.x);
        iMul(scale, lhs_fragment + k_item_idx);
        query_i += kBlockWidth;
      }

      // load keys
#pragma unroll
      for (int x_item_idx = 0; x_item_idx < kBlockItemsX; ++x_item_idx) {
        int offset = (kv_idx + x_item_idx) * KK;
        auto key_j = kkey_j + offset;
#pragma unroll
        for (int k_item_idx = 0; k_item_idx < kThreadItemsK; ++k_item_idx) {
          int fragment_offset = x_item_idx * kThreadItemsK + k_item_idx;
          rhs_fragment[fragment_offset] = __ldg(key_j + threadIdx.x);
          key_j += kBlockWidth;
        }
      }
      kkey_j += kBlockItemsK / kVecSize;

      // compute queries @ keys.T
#pragma unroll
      for (int k_item_idx = 0; k_item_idx < kThreadItemsK; ++k_item_idx) {
        const vec_t lhs_value = lhs_fragment[k_item_idx];
#pragma unroll
        for (int x_item_idx = 0; x_item_idx < kBlockItemsX; ++x_item_idx) {
          const vec_t rhs_value =
              rhs_fragment[k_item_idx + x_item_idx * kThreadItemsK];
          sputnik::VectorCompute<vec_t>::Dot(
              lhs_value, rhs_value, attn_fragment + x_item_idx);
        }
      }
    }

    int end_iter = N - kv_idx;
    scalar_t m_i = m_prime;

    // aggregate over different threads in a warp and compute max over wap
#pragma unroll
    for (int x_item_idx = 0; x_item_idx < kBlockItemsX; ++x_item_idx) {
      attn_fragment[x_item_idx] =
          warpSum<scalar_t, kBlockWidth>(attn_fragment[x_item_idx]);

      if (x_item_idx >= end_iter)
        attn_fragment[x_item_idx] =
            -std::numeric_limits<scalar_t>::infinity();

      m_i = max(attn_fragment[x_item_idx], m_i);
    }

    scalar_t m_delta = std::exp(m_prime - m_i);
    m_delta = isfinite(m_delta) ? m_delta : scalar_t(0);
    s_prime = s_prime * m_delta;
    m_prime = m_i;
    for (int x_item_idx = 0; x_item_idx < kBlockItemsX; x_item_idx++) {
      attn_fragment[x_item_idx] = std::exp(attn_fragment[x_item_idx] - m_i);
      attn_fragment[x_item_idx] = isfinite(attn_fragment[x_item_idx]) ? attn_fragment[x_item_idx] : scalar_t(0);
      s_prime += attn_fragment[x_item_idx];
    }

    vec_t* out_i_tmp = out_i;
    auto value_j = reinterpret_cast<vec_t*>(value[batch_idx].data());
    for (int k = K; k >= kBlockItemsK; k -= kBlockItemsK) {
      // load output
#pragma unroll
      for (int k_item_idx = 0; k_item_idx < kThreadItemsK; ++k_item_idx) {
        lhs_fragment[k_item_idx] = __ldg(out_i_tmp + threadIdx.x);
        iMul(m_delta, lhs_fragment + k_item_idx);
      }

      // load values
#pragma unroll
      for (int x_item_idx = 0; x_item_idx < kBlockItemsX; ++x_item_idx) {
        int offset = (kv_idx + x_item_idx) * KK;
        auto key_j = value_j + offset;
#pragma unroll
        for (int k_item_idx = 0; k_item_idx < kThreadItemsK; ++k_item_idx) {
          int fragment_offset = x_item_idx * kThreadItemsK + k_item_idx;
          rhs_fragment[fragment_offset] = __ldg(key_j + threadIdx.x);
          key_j += kBlockWidth;
        }
      }
      value_j += kBlockItemsK / kVecSize;

      // perform attn @ values computation
#pragma unroll
      for (int k_item_idx = 0; k_item_idx < kThreadItemsK; ++k_item_idx) {
        vec_t lhs_value = lhs_fragment[k_item_idx];
#pragma unroll
        for (int x_item_idx = 0; x_item_idx < kBlockItemsX; ++x_item_idx) {
          const vec_t rhs_value =
              rhs_fragment[k_item_idx + x_item_idx * kThreadItemsK];
          sputnik::VectorCompute<vec_t>::FMA(
              attn_fragment[x_item_idx], rhs_value, &lhs_value);
        }
        out_i_tmp[threadIdx.x] = lhs_value;
      }
      out_i_tmp += kBlockWidth;
    }
  }
  // avoid division by 0 when row is fully masked
  if (s_prime > 0)
    s_prime = 1.0 / s_prime;
  for (int k = threadIdx.x; k < KK; k += blockDim.x) {
    iMul(s_prime, out_i + k);
  }
}

at::Tensor attention(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value) {

  TORCH_CHECK(query.dim() == key.dim());
  TORCH_CHECK(query.dim() == value.dim());
  TORCH_CHECK(query.dim() == 3);
  TORCH_CHECK(query.size(2) == key.size(2));
  TORCH_CHECK(query.size(0) == key.size(0));

  TORCH_CHECK(query.size(0) == value.size(0));
  TORCH_CHECK(key.size(1) == value.size(1));
  TORCH_CHECK(
      query.size(2) ==
      value.size(2)); // TODO: drop this limitation in the future

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

  TORCH_CHECK(K % 32 == 0, "For now only K % 32 == 0 is supported. Let us know if you hit this and we will fix it");

  at::Tensor res = at::zeros({B, M, K}, query.options());

  using scalar_t = float;

  if (K % 128 == 0) {
  constexpr int kBlockItemsY = 16;
  constexpr int kBlockItemsX = 32;
  constexpr int kBlockItemsK = 128;
  constexpr int kBlockWidth = 8;

  dim3 grid(ceil_div(M, int64_t(kBlockItemsY)), B);
  dim3 block(kBlockWidth, kBlockItemsY);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  attention_kernel<
      scalar_t,
      kBlockWidth,
      kBlockItemsY,
      kBlockItemsX,
      kBlockItemsK><<<grid, block, 0, stream>>>(
      res.packed_accessor<scalar_t, 3>(),
      query.packed_accessor<scalar_t, 3>(),
      key.packed_accessor<scalar_t, 3>(),
      value.packed_accessor<scalar_t, 3>());

  } else if (K % 64 == 0) {
  constexpr int kBlockItemsY = 16;
  constexpr int kBlockItemsX = 32;
  constexpr int kBlockItemsK = 64;
  constexpr int kBlockWidth = 8;

  dim3 grid(ceil_div(M, int64_t(kBlockItemsY)), B);
  dim3 block(kBlockWidth, kBlockItemsY);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  attention_kernel<
      scalar_t,
      kBlockWidth,
      kBlockItemsY,
      kBlockItemsX,
      kBlockItemsK><<<grid, block, 0, stream>>>(
      res.packed_accessor<scalar_t, 3>(),
      query.packed_accessor<scalar_t, 3>(),
      key.packed_accessor<scalar_t, 3>(),
      value.packed_accessor<scalar_t, 3>());

  } else {
  constexpr int kBlockItemsY = 16;
  constexpr int kBlockItemsX = 32;
  constexpr int kBlockItemsK = 32;
  constexpr int kBlockWidth = 8;

  dim3 grid(ceil_div(M, int64_t(kBlockItemsY)), B);
  dim3 block(kBlockWidth, kBlockItemsY);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  attention_kernel<
      scalar_t,
      kBlockWidth,
      kBlockItemsY,
      kBlockItemsX,
      kBlockItemsK><<<grid, block, 0, stream>>>(
      res.packed_accessor<scalar_t, 3>(),
      query.packed_accessor<scalar_t, 3>(),
      key.packed_accessor<scalar_t, 3>(),
      value.packed_accessor<scalar_t, 3>());
  }

  AT_CUDA_CHECK(cudaGetLastError());

  return res;
}

} // namespace

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::efficient_attention2"),
      TORCH_FN(attention));
}
