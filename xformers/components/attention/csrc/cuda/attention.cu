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
__device__ __forceinline__ void iMul(scalar_t x1, float4* out) {
  out[0].x *= x1;
  out[0].y *= x1;
  out[0].z *= x1;
  out[0].w *= x1;
}

template <typename scalar_t>
__device__ __forceinline__ void iDiv(scalar_t x1, float4* out) {
  out[0].x /= x1;
  out[0].y /= x1;
  out[0].z /= x1;
  out[0].w /= x1;
}

template <typename scalar_t, int WARP_SIZE>
__device__ __forceinline__ scalar_t warpSum(scalar_t val) {
  for (int stride = WARP_SIZE / 2; stride > 0; stride >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, stride, WARP_SIZE);
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

template <typename scalar_t, typename vec_t, int BLOCK, int BLOCK2>
__device__ void compute_dot(
    vec_t* aar[BLOCK2],
    vec_t* bar,
    scalar_t si[BLOCK2][BLOCK],
    int64_t K) {
  constexpr int kVecSize = sizeof(vec_t) / sizeof(scalar_t);
  for (int64_t k = 0; k < K / kVecSize; k += 1) {
    vec_t aaar[BLOCK2];
#pragma unroll
    for (int64_t rr = 0; rr < BLOCK2; rr++) {
      aaar[rr] = __ldg(aar[rr] + k);
    }
#pragma unroll
    for (int64_t rr = 0; rr < BLOCK; rr++) {
      vec_t bbb = bar[k + K / kVecSize * rr];
#pragma unroll
      for (int64_t rr2 = 0; rr2 < BLOCK2; rr2++) {
        sputnik::VectorCompute<vec_t>::Dot(aaar[rr2], bbb, &si[rr2][rr]);
      }
    }
  }
}

template <typename scalar_t, typename vec_t, int BLOCK, int BLOCK2>
__device__ void compute_final_mult(
    vec_t* vi,
    scalar_t s_delta[BLOCK2][BLOCK],
    scalar_t m_delta[BLOCK2],
    vec_t buffer[BLOCK2][8] /*TODO fix me*/,
    int64_t K) {
  constexpr int kVecSize = sizeof(vec_t) / sizeof(scalar_t);

  for (int64_t k = 0; k < K / kVecSize; k += 1) {
#pragma unroll
    for (int64_t rr2 = 0; rr2 < BLOCK2; rr2++) {
      iMul<scalar_t>(m_delta[rr2], &buffer[rr2][k]);
    }
#pragma unroll
    for (int64_t rr = 0; rr < BLOCK; rr++) {
      vec_t tmp2 = vi[k + K / kVecSize * rr];

#pragma unroll
      for (int64_t rr2 = 0; rr2 < BLOCK2; rr2++) {
        sputnik::VectorCompute<vec_t>::FMA(
            s_delta[rr2][rr], tmp2, &buffer[rr2][k]);
      }
    }
  }
}

template <typename scalar_t, int BLOCK, int BLOCK2>
__device__ __forceinline__ void compute_max(
    scalar_t si[BLOCK2][BLOCK],
    scalar_t m_prime[BLOCK2],
    scalar_t m_i[BLOCK2]) {
#pragma unroll
  for (int64_t rr2 = 0; rr2 < BLOCK2; rr2++) {
    m_i[rr2] = si[rr2][0] > m_prime[rr2] ? si[rr2][0] : m_prime[rr2];
#pragma unroll
    for (int64_t rr = 1; rr < BLOCK; rr++) {
      m_i[rr2] = si[rr2][rr] > m_i[rr2] ? si[rr2][rr] : m_i[rr2];
    }
  }
}

template <typename scalar_t, int BLOCK, int BLOCK2>
__device__ __forceinline__ void compute_scaling_coeffs(
    scalar_t m_i[BLOCK2],
    scalar_t m_prime[BLOCK2],
    scalar_t si[BLOCK2][BLOCK],
    scalar_t m_delta[BLOCK2],
    scalar_t s_delta[BLOCK2][BLOCK]) {
#pragma unroll
  for (int64_t rr2 = 0; rr2 < BLOCK2; rr2++)
    m_delta[rr2] = std::exp(m_prime[rr2] - m_i[rr2]);
#pragma unroll
  for (int64_t rr2 = 0; rr2 < BLOCK2; rr2++)
#pragma unroll
    for (int64_t rr = 0; rr < BLOCK; rr++)
      s_delta[rr2][rr] = std::exp(si[rr2][rr] - m_i[rr2]);
}

template <typename scalar_t, int BLOCK, int BLOCK2>
__device__ __forceinline__ void update_scaling_coeffs(
    scalar_t m_delta[BLOCK2],
    scalar_t m_i[BLOCK2],
    scalar_t s_delta[BLOCK2][BLOCK],
    scalar_t m_prime[BLOCK2],
    scalar_t s_prime[BLOCK2]) {
#pragma unroll
  for (int64_t rr2 = 0; rr2 < BLOCK2; rr2++) {
    s_prime[rr2] = s_prime[rr2] * m_delta[rr2];
#pragma unroll
    for (int64_t rr = 0; rr < BLOCK; rr++)
      s_prime[rr2] += s_delta[rr2][rr];

    m_prime[rr2] = m_i[rr2];
  }
}

template <typename scalar_t, typename vec_t = float4, int BLOCK = 32, int BLOCK2 = 2, int WARP_SIZE = 4>
__global__ void attention_kernel(
    at::PackedTensorAccessor<scalar_t, 3> output,
    at::PackedTensorAccessor<scalar_t, 3> query,
    at::PackedTensorAccessor<scalar_t, 3> key,
    at::PackedTensorAccessor<scalar_t, 3> value) {
  constexpr int kVecSize = sizeof(vec_t) / sizeof(scalar_t);
  int64_t K = query.size(2);
  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);

  int64_t i = blockIdx.y;
  int64_t j = blockIdx.x * (blockDim.y * BLOCK2) + threadIdx.y * BLOCK2;

  vec_t* aar[BLOCK2];
  vec_t* oo[BLOCK2];
  vec_t buffer[BLOCK2][8] = {0}; // TODO == K / 4
  scalar_t s_prime[BLOCK2] = {0};
  scalar_t m_prime[BLOCK2] = {-std::numeric_limits<scalar_t>::infinity()};
  for (int64_t rr = 0; rr < BLOCK2; rr++) {
    aar[rr] = reinterpret_cast<vec_t*>(query[i][j + rr].data());
    oo[rr] = reinterpret_cast<vec_t*>(output[i][j + rr].data());
  }

  for (int64_t l = threadIdx.x * BLOCK; l < N; l += BLOCK * blockDim.x) {
    auto bar = reinterpret_cast<vec_t*>(key[i][l].data());
    scalar_t si[BLOCK2][BLOCK] = {0};
    compute_dot<scalar_t, vec_t, BLOCK, BLOCK2>(aar, bar, si, K);

    scalar_t m_i[BLOCK2];
    compute_max<scalar_t, BLOCK, BLOCK2>(si, m_prime, m_i);

    auto vi = reinterpret_cast<vec_t*>(value[i][l].data());

    scalar_t m_delta[BLOCK2];
    scalar_t s_delta[BLOCK2][BLOCK];

    compute_scaling_coeffs<scalar_t, BLOCK, BLOCK2>(
        m_i, m_prime, si, m_delta, s_delta);

    compute_final_mult<scalar_t, vec_t, BLOCK, BLOCK2>(
        vi, s_delta, m_delta, buffer, K);

    update_scaling_coeffs<scalar_t, BLOCK, BLOCK2>(
        m_delta, m_i, s_delta, m_prime, s_prime);
  }

  for (int64_t rr = 0; rr < BLOCK2; rr++) {
    scalar_t m_i = m_prime[rr];
    scalar_t s_i = s_prime[rr];
    m_prime[rr] = warpMax<scalar_t, WARP_SIZE>(m_prime[rr]);
    scalar_t m_delta = std::exp(m_i - m_prime[rr]);
    scalar_t s_delta = s_i * m_delta;
    s_delta = warpSum<scalar_t, WARP_SIZE>(s_delta);
    s_prime[rr] = s_delta;
    for (int64_t k = 0; k < K / kVecSize; k += 1) {
      vec_t tmp = buffer[rr][k];
      iMul<scalar_t>(m_delta, &tmp);
      tmp = warpSum<vec_t, WARP_SIZE>(tmp);
      buffer[rr][k] = tmp;
    }
  }

  for (int64_t k = threadIdx.x; k < K / kVecSize; k += blockDim.x) {
    vec_t tmp;

#pragma unroll
    for (int64_t rr2 = 0; rr2 < BLOCK2; rr2++) {
      tmp = buffer[rr2][k];
      iDiv<scalar_t>(s_prime[rr2], &tmp);

      oo[rr2][k] = tmp;
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
  // TORCH_CHECK(query.dim() == mask.dim());
  TORCH_CHECK(query.dim() == 3);
  TORCH_CHECK(query.size(2) == key.size(2));
  TORCH_CHECK(query.size(0) == key.size(0));
  // TORCH_CHECK(query.size(1) == mask.size(1));
  // TORCH_CHECK(query.size(2) == mask.size(2));
  // TORCH_CHECK(query.size(0) == mask.size(0));

  /*
  TORCH_CHECK(!a.is_cuda(), "a must be a CPU tensor");
  TORCH_CHECK(!b.is_cuda(), "b must be a CPU tensor");
  TORCH_CHECK(!mask.is_cuda(), "mask must be a CPU tensor");

  TORCH_CHECK(!a.is_sparse(), "a must be a dense tensor");
  TORCH_CHECK(!b.is_sparse(), "b must be a dense tensor");
  //TORCH_CHECK(mask.is_sparse(), "mask must be a sparse tensor");
  */
  TORCH_CHECK(query.is_contiguous());
  TORCH_CHECK(key.is_contiguous());
  TORCH_CHECK(value.is_contiguous());

  at::cuda::CUDAGuard device_guard(query.device());

  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);
  int64_t K = query.size(2);

  at::Tensor res = at::zeros({B, M, K}, query.options());


  constexpr int WARP_SIZE = 4;

  constexpr int BLOCK = 32;
  constexpr int BLOCK2 = 2;

  constexpr int TILE_SIZE = 32;


  dim3 grid(M / TILE_SIZE, B);
  dim3 block(WARP_SIZE, TILE_SIZE / BLOCK2);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  using scalar_t = float;
  // AT_DISPATCH_FLOATING_TYPES(
  // query.scalar_type(), "attention_kernel", [&] {
  attention_kernel<scalar_t, float4, BLOCK, BLOCK2, WARP_SIZE><<<grid, block, 0, stream>>>(
      res.packed_accessor<scalar_t, 3>(),
      query.packed_accessor<scalar_t, 3>(),
      key.packed_accessor<scalar_t, 3>(),
      value.packed_accessor<scalar_t, 3>()
      // buffer.accessor<scalar_t, 3>()
      // idxs.accessor<int64_t, 2>()
  );
  //});

  return res;
}

} // namespace

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::efficient_attention"),
      TORCH_FN(attention));
}
