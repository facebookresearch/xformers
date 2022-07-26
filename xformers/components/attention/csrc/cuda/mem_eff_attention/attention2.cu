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

#include "sputnik/common.h"

namespace sputnik {

// very similar to sputnik/sddmm/all_reduce.h but with a minor
// fix so that this works for more sizes
template <typename LoadType, int kBlockItemsX, int kBlockWidth>
struct AllReduce {
  //
  /// Static members.
  //

  // The number of values that will be loaded per-thread, per-load.
  static constexpr int kValuesPerLoad = sizeof(LoadType) / sizeof(float);

  // The number of outputs each thread is responsible for.
  static constexpr int kThreadItemsX = kBlockItemsX / kBlockWidth;

  //
  /// Member variables.
  //

  // Thread mask used for warp shuffle operations.
  const uint32_t kShflMask;

  // Register file fragment storing the thread local partial results.
  float* inputs;

  // Registe file fragment for storing each threads results.
  float* outputs;

  __device__ __forceinline__ AllReduce(const uint32_t thread_mask,
                                       float* inputs_, float* outputs_)
      : kShflMask(thread_mask), inputs(inputs_), outputs(outputs_) {}
  __device__ __forceinline__ void Swap(int i, int j, float* x) {
    float t = x[i];
    x[i] = x[j];
    x[j] = t;
  }

  __device__ __forceinline__ void ReduceStep(int lane, int i, int j) {
    const int kStep = Log2(lane);
    if ((threadIdx.x >> kStep) & 1) Swap(i, j, inputs);
    inputs[i] += __shfl_xor_sync(kShflMask, inputs[j], lane, kBlockWidth);
  }

  __device__ __forceinline__ void Reduce() {
#pragma unroll
    for (int base_idx = 0; base_idx < kBlockItemsX; base_idx += kBlockWidth) {
#pragma unroll
      for (int k_item_idx = 1; k_item_idx < kBlockWidth; k_item_idx *= 2) {
        const int kBoundX = kBlockWidth / (k_item_idx * 2);
#pragma unroll
        for (int x_item_idx = 0; x_item_idx < kBoundX; ++x_item_idx) {
          const int idx_a = x_item_idx * 2 * kValuesPerLoad * k_item_idx;
          const int idx_b = (x_item_idx * 2 + 1) * kValuesPerLoad * k_item_idx;
          ReduceStep(k_item_idx, base_idx + idx_a, base_idx + idx_b);
        }
      }
    }

    // Move the last four values to the first four of the output. This
    // should get cleaned up during register allocation.
#pragma unroll
    for (int out_idx = 0; out_idx < kThreadItemsX; ++out_idx) {
      outputs[out_idx] = inputs[out_idx * kBlockWidth];
    }
  }
};

}  // namespace sputnik


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

template <typename scalar_t, typename vec_t, int kBlockSizeK>
//__device__ __forceinline__ void apply_masking(
__device__ void apply_masking(
    scalar_t s_delta[kBlockSizeK],
    at::PhiloxCudaState philox_args,
    int64_t global_offset,
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
  for (int64_t k_item_idx = 0; k_item_idx < kBlockSizeK;
       k_item_idx += kSampled) {
    int64_t offset = global_offset + k_item_idx + col_offset;
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
        s_delta[k_item_idx + kk] *= (&rand.x)[kk] < p;
    }
  }
}

template <typename scalar_t>
at::PackedTensorAccessor<scalar_t, 3> _packed_tensor_accessor_or_dummy(
    const at::Tensor& attn_bias) {
  if (attn_bias.defined()) {
    return attn_bias.packed_accessor<scalar_t, 3>();
  } else {
    const std::array<int64_t, 3> zeros{{0}};
    return at::PackedTensorAccessor<scalar_t, 3>(
        nullptr, zeros.data(), zeros.data());
  }
}

template <
    typename scalar_t,
    typename vec_t,
    int kBlockWidth,
    int kBlockItemsY,
    int kBlockItemsX,
    int kBlockItemsK>
__global__ void attention_kernel(
    at::PackedTensorAccessor<scalar_t, 3> output,
    at::PackedTensorAccessor<scalar_t, 2> logsumexp,
    at::PackedTensorAccessor<scalar_t, 3> query,
    at::PackedTensorAccessor<scalar_t, 3> key,
    at::PackedTensorAccessor<scalar_t, 3> value,
    at::PackedTensorAccessor<scalar_t, 3> attn_bias_,
    scalar_t p,
    at::PhiloxCudaState philox_args,
    bool is_causal
    ) {
  int64_t K = query.size(2);
  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);

  int64_t batch_idx = blockIdx.y;
  int64_t query_idx = blockIdx.x * kBlockItemsY + threadIdx.y;

  if (query_idx >= M)
    return;

  constexpr int kVecSize = sizeof(vec_t) / sizeof(scalar_t);
  constexpr int kThreadItemsK = kBlockItemsK / kBlockWidth / kVecSize;

  int KK = K / kVecSize;

  scalar_t s_prime = 0;
  scalar_t m_prime = -std::numeric_limits<scalar_t>::infinity();

  vec_t lhs_fragment[kThreadItemsK];
  vec_t rhs_fragment[kBlockItemsX * kThreadItemsK];

  auto out_i = reinterpret_cast<vec_t*>(output[batch_idx][query_idx].data());

  scalar_t scale = 1.0 / std::sqrt(scalar_t(K));

  auto attn_bias = attn_bias_[batch_idx][query_idx].data();
  int end = N;
  if (is_causal)
    end = std::min(int(query_idx) + 1, end);

  for (int kv_idx = 0; kv_idx < end; kv_idx += kBlockItemsX) {
    scalar_t attn_fragment[kBlockItemsX] = {};

    auto query_i = reinterpret_cast<vec_t*>(query[batch_idx][query_idx].data());
    auto kkey_j = reinterpret_cast<vec_t*>(key[batch_idx].data());

    int k = K;
    for (; k >= kBlockItemsK; k -= kBlockItemsK) {
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
    // residue computation
    if (k > 0) {
      k -= threadIdx.x * kVecSize;
      // load queries
      int residue = k;
#pragma unroll
      for (int k_item_idx = 0; k_item_idx < kThreadItemsK; ++k_item_idx) {
        if (residue > 0) {
          lhs_fragment[k_item_idx] = __ldg(query_i + threadIdx.x);
          iMul(scale, lhs_fragment + k_item_idx);
        }
        query_i += kBlockWidth;
        residue -= kBlockWidth * kVecSize;
      }

      // load keys and compute
      residue = k;
#pragma unroll
      for (int x_item_idx = 0; x_item_idx < kBlockItemsX; ++x_item_idx) {
        int offset = (kv_idx + x_item_idx) * KK;
        auto key_j = kkey_j + offset;
        int inner_residue = residue;
#pragma unroll
        for (int k_item_idx = 0; k_item_idx < kThreadItemsK; ++k_item_idx) {
          if (inner_residue > 0) {
            int fragment_offset = x_item_idx * kThreadItemsK + k_item_idx;
            rhs_fragment[fragment_offset] = __ldg(key_j + threadIdx.x);

            sputnik::VectorCompute<vec_t>::Dot(
                lhs_fragment[k_item_idx], rhs_fragment[fragment_offset], attn_fragment + x_item_idx);
          }
          key_j += kBlockWidth;
          inner_residue -= kBlockWidth * kVecSize;
        }
      }
    }

    int end_iter = end - kv_idx;
    scalar_t m_i = m_prime;

    // aggregate over different threads in a warp and compute max over wap
    // the reduced values will be shared across different threads, such that
    // threadIdx.x will have values of index i s.t. that i % kBlockWidth == threadIdx.x
    sputnik::AllReduce<scalar_t, kBlockItemsX, kBlockWidth> all_reduce(0xffffffff, attn_fragment, attn_fragment);
    all_reduce.Reduce();

#pragma unroll
    for (int x_item_idx = 0; x_item_idx < kBlockItemsX / kBlockWidth; ++x_item_idx) {
      bool in_bounds = x_item_idx * kBlockWidth + threadIdx.x < end_iter;
      if (!in_bounds)
        attn_fragment[x_item_idx] = -std::numeric_limits<scalar_t>::infinity();
      if ((attn_bias != nullptr) && in_bounds)
        attn_fragment[x_item_idx] += __ldg(attn_bias + kv_idx + x_item_idx * kBlockWidth + threadIdx.x);
      m_i = max(attn_fragment[x_item_idx], m_i);
    }
    m_i = warpMax<scalar_t, kBlockWidth>(m_i);

    scalar_t m_delta = std::exp(m_prime - m_i);
    m_delta = isfinite(m_delta) ? m_delta : scalar_t(0);
    s_prime = s_prime * m_delta;
    m_prime = m_i;
#pragma unroll
    for (int x_item_idx = 0; x_item_idx < kBlockItemsX / kBlockWidth; x_item_idx++) {
      attn_fragment[x_item_idx] = std::exp(attn_fragment[x_item_idx] - m_i);
      attn_fragment[x_item_idx] = isfinite(attn_fragment[x_item_idx]) ? attn_fragment[x_item_idx] : scalar_t(0);
      s_prime += attn_fragment[x_item_idx];
    }

    // distribute aggregated attention values to all threads in the warp
#pragma unroll
    for (int x_item_idx = kBlockItemsX - 1; x_item_idx >= 0; x_item_idx--) {
      scalar_t val = attn_fragment[x_item_idx / kBlockWidth];
      attn_fragment[x_item_idx] = __shfl_sync(0xffffffff, val, x_item_idx % kBlockWidth, kBlockWidth);
    }

    if (p < 1.0) {
      // this approach is suboptimal as different threads compute the
      // same mask value, and this could be parallelized
      int global_offset = batch_idx * M * N + query_idx * N;
      apply_masking<scalar_t, vec_t, kBlockItemsX>(
          attn_fragment, philox_args, global_offset, p, kv_idx);
    }

    vec_t* out_i_tmp = out_i;
    auto value_j = reinterpret_cast<vec_t*>(value[batch_idx].data());
    k = K;
    for (; k >= kBlockItemsK; k -= kBlockItemsK) {
      // load output
#pragma unroll
      for (int k_item_idx = 0; k_item_idx < kThreadItemsK; ++k_item_idx) {
        lhs_fragment[k_item_idx] = __ldg(out_i_tmp + threadIdx.x);
        iMul(m_delta, lhs_fragment + k_item_idx);
        out_i_tmp += kBlockWidth;
      }
      out_i_tmp -= kBlockWidth * kThreadItemsK;

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
        out_i_tmp += kBlockWidth;
      }
    }

    // residue handling
    if (k > 0) {
      k -= threadIdx.x * kVecSize;
      // load queries
      int residue = k;
#pragma unroll
      for (int k_item_idx = 0; k_item_idx < kThreadItemsK; ++k_item_idx) {
        if (residue > 0) {
          lhs_fragment[k_item_idx] = __ldg(out_i_tmp + threadIdx.x);
          iMul(m_delta, lhs_fragment + k_item_idx);
        }
        out_i_tmp += kBlockWidth;
        residue -= kBlockWidth * kVecSize;
      }
      out_i_tmp -= kBlockWidth * kThreadItemsK;

      // load keys and compute
      residue = k;
#pragma unroll
      for (int x_item_idx = 0; x_item_idx < kBlockItemsX; ++x_item_idx) {
        int offset = (kv_idx + x_item_idx) * KK;
        auto key_j = value_j + offset;
        int inner_residue = residue;
#pragma unroll
        for (int k_item_idx = 0; k_item_idx < kThreadItemsK; ++k_item_idx) {
          if (inner_residue > 0) {
            int fragment_offset = x_item_idx * kThreadItemsK + k_item_idx;
            rhs_fragment[fragment_offset] = __ldg(key_j + threadIdx.x);

            sputnik::VectorCompute<vec_t>::FMA(
              attn_fragment[x_item_idx], rhs_fragment[fragment_offset], lhs_fragment + k_item_idx);
          }
          key_j += kBlockWidth;
          inner_residue -= kBlockWidth * kVecSize;
        }
      }

#pragma unroll
      for (int k_item_idx = 0; k_item_idx < kThreadItemsK; ++k_item_idx) {
        if (residue > 0) {
          out_i_tmp[threadIdx.x] = lhs_fragment[k_item_idx];
        }
        out_i_tmp += kBlockWidth;
        residue -= kBlockWidth;
      }
    }
  }
  s_prime = warpSum<scalar_t, kBlockWidth>(s_prime);
  if (logsumexp.size(1) > 0) {
    logsumexp[batch_idx][query_idx] = m_prime + std::log(s_prime);
  }
  // update normalization constant with dropout probability
  s_prime *= p;
  // avoid division by 0 when row is fully masked
  if (s_prime > 0)
    s_prime = 1.0 / s_prime;
  for (int k = threadIdx.x; k < KK; k += blockDim.x) {
    iMul(s_prime, out_i + k);
  }
}

std::tuple<at::Tensor, at::Tensor, int64_t, int64_t> attention(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    bool compute_logsumexp,
    const c10::optional<at::Tensor>& attn_bias_,
    double p,
    bool is_causal) {

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
  }

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

  at::Tensor res = at::zeros({B, M, K}, query.options());
  at::Tensor logsumexp = at::empty(
      {B, compute_logsumexp ? M : 0},
      query.options().dtype(at::ScalarType::Float));

  using scalar_t = float;

  constexpr int kBlockItemsY = 16;
  constexpr int kBlockItemsX = 32;
  constexpr int kBlockItemsK = 32;
  constexpr int kBlockWidth = 8;

  dim3 grid(ceil_div(M, int64_t(kBlockItemsY)), B);
  dim3 block(kBlockWidth, kBlockItemsY);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

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

  auto attn_bias_packed = _packed_tensor_accessor_or_dummy<scalar_t>(attn_bias);

  if ((K % 4) == 0) {
    attention_kernel<
        scalar_t,
        float4,
        kBlockWidth,
        kBlockItemsY,
        kBlockItemsX,
        kBlockItemsK><<<grid, block, 0, stream>>>(
        res.packed_accessor<scalar_t, 3>(),
        logsumexp.packed_accessor<scalar_t, 2>(),
        query.packed_accessor<scalar_t, 3>(),
        key.packed_accessor<scalar_t, 3>(),
        value.packed_accessor<scalar_t, 3>(),
        attn_bias_packed,
        p,
        rng_engine_inputs,
        is_causal);
  } else if ((K % 2) == 0) {
    attention_kernel<
        scalar_t,
        float2,
        kBlockWidth,
        kBlockItemsY,
        kBlockItemsX,
        kBlockItemsK><<<grid, block, 0, stream>>>(
        res.packed_accessor<scalar_t, 3>(),
        logsumexp.packed_accessor<scalar_t, 2>(),
        query.packed_accessor<scalar_t, 3>(),
        key.packed_accessor<scalar_t, 3>(),
        value.packed_accessor<scalar_t, 3>(),
        attn_bias_packed,
        p,
        rng_engine_inputs,
        is_causal);
  } else {
    attention_kernel<
        scalar_t,
        float,
        kBlockWidth,
        kBlockItemsY,
        kBlockItemsX,
        kBlockItemsK><<<grid, block, 0, stream>>>(
        res.packed_accessor<scalar_t, 3>(),
        logsumexp.packed_accessor<scalar_t, 2>(),
        query.packed_accessor<scalar_t, 3>(),
        key.packed_accessor<scalar_t, 3>(),
        value.packed_accessor<scalar_t, 3>(),
        attn_bias_packed,
        p,
        rng_engine_inputs,
        is_causal);
  }

  AT_CUDA_CHECK(cudaGetLastError());

  // uint64_t -> int64_t bitwise casting as PyTorch don't support uint64_t
  // so just fake it as a int64_t
  int64_t seed, offset;
  std::memcpy(&seed, &rng_engine_inputs.seed_, sizeof(seed));
  std::memcpy(&offset, &rng_engine_inputs.offset_.val, sizeof(offset));

  return std::make_tuple(res, logsumexp, seed, offset);
}

} // namespace

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::efficient_attention2"),
      TORCH_FN(attention));
}
