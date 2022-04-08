#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <cmath>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

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
  vec_t q_i[kBlockSizeQ];
  for (int64_t k = 0; k < K / kVecSize; k += 1) {
#pragma unroll
    for (int64_t q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
      q_i[q_item_idx] = __ldg(queries[q_item_idx] + k);
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

template <typename scalar_t, typename vec_t, int kBlockSizeK, int kBlockSizeQ, int BUFFER_SIZE>
__device__ void compute_final_mult(
    vec_t* vi,
    scalar_t s_delta[kBlockSizeQ][kBlockSizeK],
    scalar_t m_delta[kBlockSizeQ],
    vec_t buffer[kBlockSizeQ][BUFFER_SIZE] /*TODO fix me*/,
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
__device__ __forceinline__ void compute_scaling_coeffs(
    scalar_t m_i[kBlockSizeQ],
    scalar_t m_prime[kBlockSizeQ],
    scalar_t si[kBlockSizeQ][kBlockSizeK],
    scalar_t m_delta[kBlockSizeQ],
    scalar_t s_delta[kBlockSizeQ][kBlockSizeK]) {
#pragma unroll
  for (int64_t q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++)
    m_delta[q_item_idx] = std::exp(m_prime[q_item_idx] - m_i[q_item_idx]);
#pragma unroll
  for (int64_t q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++)
#pragma unroll
    for (int64_t k_item_idx = 0; k_item_idx < kBlockSizeK; k_item_idx++)
      s_delta[q_item_idx][k_item_idx] =
          std::exp(si[q_item_idx][k_item_idx] - m_i[q_item_idx]);
}

template <typename scalar_t, int kBlockSizeK, int kBlockSizeQ>
__device__ __forceinline__ void update_scaling_coeffs(
    scalar_t m_delta[kBlockSizeQ],
    scalar_t m_i[kBlockSizeQ],
    scalar_t s_delta[kBlockSizeQ][kBlockSizeK],
    scalar_t m_prime[kBlockSizeQ],
    scalar_t s_prime[kBlockSizeQ]) {
#pragma unroll
  for (int64_t q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
    s_prime[q_item_idx] = s_prime[q_item_idx] * m_delta[q_item_idx];
#pragma unroll
    for (int64_t k_item_idx = 0; k_item_idx < kBlockSizeK; k_item_idx++)
      s_prime[q_item_idx] += s_delta[q_item_idx][k_item_idx];

    m_prime[q_item_idx] = m_i[q_item_idx];
  }
}

template <typename scalar_t, typename vec_t, int kBlockSizeK, int kBlockSizeQ, int BUFFER_SIZE>
__device__ void compute_loop(
    vec_t* query_block[kBlockSizeQ],
    vec_t* key_i,
    vec_t* value_i,
    scalar_t m_prime[kBlockSizeQ],
    scalar_t s_prime[kBlockSizeQ],
    vec_t buffer[kBlockSizeQ][BUFFER_SIZE] /*TODO fix me*/,
    int64_t K) {

    scalar_t si[kBlockSizeQ][kBlockSizeK] = {0};
    compute_dot<scalar_t, vec_t, kBlockSizeK, kBlockSizeQ>(
        query_block, key_i, si, K);

    scalar_t m_i[kBlockSizeQ];
    compute_max<scalar_t, kBlockSizeK, kBlockSizeQ>(si, m_prime, m_i);

    scalar_t m_delta[kBlockSizeQ];
    scalar_t s_delta[kBlockSizeQ][kBlockSizeK];

    compute_scaling_coeffs<scalar_t, kBlockSizeK, kBlockSizeQ>(
        m_i, m_prime, si, m_delta, s_delta);

    compute_final_mult<scalar_t, vec_t, kBlockSizeK, kBlockSizeQ, BUFFER_SIZE>(
        value_i, s_delta, m_delta, buffer, K);

    update_scaling_coeffs<scalar_t, kBlockSizeK, kBlockSizeQ>(
        m_delta, m_i, s_delta, m_prime, s_prime);
}


template <typename scalar_t, typename vec_t, int kBlockSizeQ, int WARP_SIZE, int BUFFER_SIZE>
__device__ __forceinline__ void aggregate_coeffs(
    scalar_t m_prime[kBlockSizeQ],
    scalar_t s_prime[kBlockSizeQ],
    vec_t buffer[kBlockSizeQ][BUFFER_SIZE] /*TODO fix me*/,
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

template <bool first, typename scalar_t, typename vec_t, int kBlockSizeK, int kBlockSizeQ, int BUFFER_SIZE, int WARP_SIZE>
struct UnrollLoop {
  static __device__ __forceinline__ void eval(
    vec_t* query_block[kBlockSizeQ],
    at::TensorAccessor<scalar_t, 2> key,
    at::TensorAccessor<scalar_t, 2> value,
    scalar_t m_prime[kBlockSizeQ],
    scalar_t s_prime[kBlockSizeQ],
    vec_t buffer[kBlockSizeQ][BUFFER_SIZE] /*TODO fix me*/,
    int64_t K,
    int64_t N) {
    constexpr int64_t step = kBlockSizeK * WARP_SIZE;
    int64_t l;
    if (first) {
      l = threadIdx.x * kBlockSizeK;
    } else {
      l = N - (N & (step - 1)) + threadIdx.x * kBlockSizeK;
    }
    // this is equivalent to N - N % step, but faster
    // guaranteed to be the same as step is a power of 2
    int64_t end_iter = N - (N & (step - 1));
    //if (l < end_iter) {
    {
    for (; l < end_iter;
         l += step) {
      auto key_i = reinterpret_cast<vec_t*>(key[l].data());
      auto value_i = reinterpret_cast<vec_t*>(value[l].data());

      compute_loop<scalar_t, vec_t, kBlockSizeK, kBlockSizeQ, BUFFER_SIZE>(query_block, key_i, value_i, m_prime, s_prime, buffer, K);
    }
    }
    if (l < N) {
      UnrollLoop<false, scalar_t, vec_t, kBlockSizeK / 2, kBlockSizeQ, BUFFER_SIZE, WARP_SIZE>::eval(query_block, key, value, m_prime, s_prime, buffer, K, N);
    }
  }
};

template <typename scalar_t, typename vec_t, int kBlockSizeQ, int BUFFER_SIZE, int WARP_SIZE>
struct UnrollLoop<false, scalar_t, vec_t, 0, kBlockSizeQ, BUFFER_SIZE, WARP_SIZE> {
  static __device__ __forceinline__ void eval(
    vec_t* query_block[kBlockSizeQ],
    at::TensorAccessor<scalar_t, 2> key,
    at::TensorAccessor<scalar_t, 2> value,
    scalar_t m_prime[kBlockSizeQ],
    scalar_t s_prime[kBlockSizeQ],
    vec_t buffer[kBlockSizeQ][BUFFER_SIZE] /*TODO fix me*/,
    int64_t K,
    int64_t N) {
  }
};


template <
    typename scalar_t,
    typename vec_t = float4,
    int kBlockSizeK = 32,
    int kBlockSizeQ = 2,
    int WARP_SIZE = 4,
    int BUFFER_SIZE = 8>
__global__ void attention_kernel(
    at::PackedTensorAccessor<scalar_t, 3> output,
    at::PackedTensorAccessor<scalar_t, 3> query,
    at::PackedTensorAccessor<scalar_t, 3> key,
    at::PackedTensorAccessor<scalar_t, 3> value) {
  constexpr int kVecSize = sizeof(vec_t) / sizeof(scalar_t);
  static_assert(integerIsPowerOf2(kBlockSizeK * WARP_SIZE), "kBlockSizeK * WARP_SIZE should be a power of 2");
  int64_t K = query.size(2);
  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);

  int64_t batch_idx = blockIdx.y;
  int64_t query_idx =
      blockIdx.x * (blockDim.y * kBlockSizeQ) + threadIdx.y * kBlockSizeQ;

  if (query_idx >= M)
    return;

  vec_t* query_block[kBlockSizeQ];
  vec_t* output_block[kBlockSizeQ];
  vec_t buffer[kBlockSizeQ][BUFFER_SIZE] = {0}; // TODO == K / 4
  scalar_t s_prime[kBlockSizeQ] = {0};
  scalar_t m_prime[kBlockSizeQ] = {-std::numeric_limits<scalar_t>::infinity()};
  for (int64_t q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
    int64_t index = query_idx + q_item_idx;
    index = index >= M ? M - 1 : index;
    query_block[q_item_idx] = reinterpret_cast<vec_t*>(
        query[batch_idx][index].data());
    output_block[q_item_idx] = reinterpret_cast<vec_t*>(
        output[batch_idx][index].data());
  }
#if 0
  // this for now makes things slower
  UnrollLoop<true, scalar_t, vec_t, kBlockSizeK, kBlockSizeQ, BUFFER_SIZE, WARP_SIZE>::eval(query_block, key[batch_idx], value[batch_idx], m_prime, s_prime, buffer, K, N);
#else
  int64_t l = threadIdx.x * kBlockSizeK;
  constexpr int64_t step = kBlockSizeK * WARP_SIZE;
  // this is equivalent to N - N % step, but faster
  // guaranteed to be the same as step is a power of 2
  int64_t end_iter = N - (N & (step - 1));
  for (; l < end_iter;
       l += step) {
    auto key_i = reinterpret_cast<vec_t*>(key[batch_idx][l].data());
    auto value_i = reinterpret_cast<vec_t*>(value[batch_idx][l].data());

    compute_loop<scalar_t, vec_t, kBlockSizeK, kBlockSizeQ, BUFFER_SIZE>(query_block, key_i, value_i, m_prime, s_prime, buffer, K);
  }

  if (l < N) {
    // TODO: unroll this in a generic manner
    l = N - (N & (step - 1)) + threadIdx.x * (kBlockSizeK / 2);
    end_iter = N - (N & (step / 2 - 1));
    for (; l < end_iter;
         l += step / 2) {
      auto key_i = reinterpret_cast<vec_t*>(key[batch_idx][l].data());
      auto value_i = reinterpret_cast<vec_t*>(value[batch_idx][l].data());
      compute_loop<scalar_t, vec_t, kBlockSizeK / 2, kBlockSizeQ, BUFFER_SIZE>(query_block, key_i, value_i, m_prime, s_prime, buffer, K);
    }

    l = N - (N & (step / 2 - 1)) + threadIdx.x * (kBlockSizeK / 4);
    end_iter = N - (N & (step / 4 - 1));
    for (; l < end_iter;
         l += step / 4) {
      auto key_i = reinterpret_cast<vec_t*>(key[batch_idx][l].data());
      auto value_i = reinterpret_cast<vec_t*>(value[batch_idx][l].data());
      compute_loop<scalar_t, vec_t, kBlockSizeK / 4, kBlockSizeQ, BUFFER_SIZE>(query_block, key_i, value_i, m_prime, s_prime, buffer, K);
    }

    l = N - (N & (step / 4 - 1)) + threadIdx.x * (kBlockSizeK / 8);
    end_iter = N - (N & (step / 8 - 1));
    for (; l < end_iter;
         l += step / 8) {
      auto key_i = reinterpret_cast<vec_t*>(key[batch_idx][l].data());
      auto value_i = reinterpret_cast<vec_t*>(value[batch_idx][l].data());
      compute_loop<scalar_t, vec_t, kBlockSizeK / 8, kBlockSizeQ, BUFFER_SIZE>(query_block, key_i, value_i, m_prime, s_prime, buffer, K);
    }

    l = N - (N & (step / 8 - 1)) + threadIdx.x * (kBlockSizeK / 16);
    end_iter = N - (N & (step / 16 - 1));
    for (; l < end_iter;
         l += step / 16) {
      auto key_i = reinterpret_cast<vec_t*>(key[batch_idx][l].data());
      auto value_i = reinterpret_cast<vec_t*>(value[batch_idx][l].data());
      compute_loop<scalar_t, vec_t, kBlockSizeK / 16, kBlockSizeQ, BUFFER_SIZE>(query_block, key_i, value_i, m_prime, s_prime, buffer, K);
    }

    l = N - (N & (step / 16 - 1)) + threadIdx.x;
    for (; l < N;
         l += blockDim.x) {
      auto key_i = reinterpret_cast<vec_t*>(key[batch_idx][l].data());
      auto value_i = reinterpret_cast<vec_t*>(value[batch_idx][l].data());
      compute_loop<scalar_t, vec_t, 1, kBlockSizeQ, BUFFER_SIZE>(query_block, key_i, value_i, m_prime, s_prime, buffer, K);
    }
  }
#endif

  aggregate_coeffs<scalar_t, vec_t, kBlockSizeQ, WARP_SIZE, BUFFER_SIZE>(m_prime, s_prime, buffer, K);

  for (int64_t k = threadIdx.x; k < K / kVecSize; k += blockDim.x) {
    vec_t tmp;

#pragma unroll
    for (int64_t q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
      tmp = buffer[q_item_idx][k];
      iDiv<scalar_t>(s_prime[q_item_idx], &tmp);

      output_block[q_item_idx][k] = tmp;
    }
  }
}

at::Tensor attention(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value
    // const at::Tensor& mask
) {
  TORCH_CHECK(query.dim() == key.dim());
  TORCH_CHECK(query.dim() == value.dim());
  // TORCH_CHECK(query.dim() == mask.dim());
  TORCH_CHECK(query.dim() == 3);
  TORCH_CHECK(query.size(2) == key.size(2));
  TORCH_CHECK(query.size(0) == key.size(0));

  TORCH_CHECK(query.size(0) == value.size(0));
  TORCH_CHECK(key.size(1) == value.size(1));
  TORCH_CHECK(query.size(2) == value.size(2)); // TODO: drop this limitation in the future

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
  TORCH_CHECK(query.scalar_type() == at::ScalarType::Float, "Only float supported by now");

  at::cuda::CUDAGuard device_guard(query.device());

  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);
  int64_t K = query.size(2);

  at::Tensor res = at::zeros({B, M, K}, query.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  constexpr int WARP_SIZE = 4;

  constexpr int kBlockSizeK = 32;
  constexpr int kBlockSizeQ = 2;

  constexpr int TILE_SIZE = 32;
  constexpr int BUFFER_SIZE = 8;

  dim3 grid(ceil_div(M, int64_t(TILE_SIZE)), B);
  dim3 block(WARP_SIZE, TILE_SIZE / kBlockSizeQ);

  using scalar_t = float;

  if ((K % 4) == 0) {
    attention_kernel<scalar_t, float4, kBlockSizeK, kBlockSizeQ, WARP_SIZE, BUFFER_SIZE>
        <<<grid, block, 0, stream>>>(
            res.packed_accessor<scalar_t, 3>(),
            query.packed_accessor<scalar_t, 3>(),
            key.packed_accessor<scalar_t, 3>(),
            value.packed_accessor<scalar_t, 3>()
        );
  } else if ((K % 2) == 0) {
    attention_kernel<scalar_t, float2, kBlockSizeK, kBlockSizeQ, WARP_SIZE, BUFFER_SIZE>
        <<<grid, block, 0, stream>>>(
            res.packed_accessor<scalar_t, 3>(),
            query.packed_accessor<scalar_t, 3>(),
            key.packed_accessor<scalar_t, 3>(),
            value.packed_accessor<scalar_t, 3>()
        );

  } else {
    attention_kernel<scalar_t, float, kBlockSizeK, kBlockSizeQ, WARP_SIZE, BUFFER_SIZE>
        <<<grid, block, 0, stream>>>(
            res.packed_accessor<scalar_t, 3>(),
            query.packed_accessor<scalar_t, 3>(),
            key.packed_accessor<scalar_t, 3>(),
            value.packed_accessor<scalar_t, 3>()
        );

  }

  return res;
}

} // namespace

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::efficient_attention"),
      TORCH_FN(attention));
}
