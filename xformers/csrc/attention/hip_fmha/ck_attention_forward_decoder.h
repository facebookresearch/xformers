#pragma once 

#include <ck/host_utility/kernel_launch_hip.hpp>
#include <ck/stream_config.hpp>
#include <ck/tensor_operation/gpu/device/device_base.hpp>
#include <ck/utility/data_type.hpp>
#include <ck/utility/inner_product.hpp>

namespace ck {
template <>
__device__ void inner_product<bhalf_t, bhalf_t, float>(
    const bhalf_t& a,
    const bhalf_t& b,
    float& c) {
  inner_product(type_convert<float>(a), type_convert<float>(b), c);
}

template <>
__device__ void inner_product<bhalf4_t, bhalf4_t, float>(
    const bhalf4_t& a,
    const bhalf4_t& b,
    float& c) {
  const vector_type<bhalf_t, 4> a_vector{a};
  const vector_type<bhalf_t, 4> b_vector{b};
  ck::static_for<0, 4, 1>{}([&](auto i) {
    inner_product(
        a_vector.AsType<bhalf_t>()[i], b_vector.AsType<bhalf_t>()[i], c);
  });
}
} // namespace ck

namespace {

template <typename data4_t>
__device__ ck::float4_t scalar4_scale_acc(ck::float4_t acc, data4_t a, float b);

template <>
__device__ ck::float4_t scalar4_scale_acc<ck::float4_t>(
    ck::float4_t acc,
    ck::float4_t a,
    float b) {
  return acc + a * b;
}

template <>
__device__ ck::float4_t scalar4_scale_acc<ck::half4_t>(
    ck::float4_t acc,
    ck::half4_t a,
    float b) {
  acc.x += ck::type_convert<float>(a.x) * b;
  acc.y += ck::type_convert<float>(a.y) * b;
  acc.z += ck::type_convert<float>(a.z) * b;
  acc.w += ck::type_convert<float>(a.w) * b;
  return acc;
}

template <>
__device__ ck::float4_t scalar4_scale_acc<ck::bhalf4_t>(
    ck::float4_t acc,
    ck::bhalf4_t a,
    float b) {
  acc.x += ck::type_convert<float>(a.x) * b;
  acc.y += ck::type_convert<float>(a.y) * b;
  acc.z += ck::type_convert<float>(a.z) * b;
  acc.w += ck::type_convert<float>(a.w) * b;
  return acc;
}

template <typename F, int32_t n_threads_per_wavefront = 64>
float __device__ __forceinline__ wavefrontReduce(float val, F f) {
#pragma unroll
  for (int32_t mask = n_threads_per_wavefront >> 1; mask > 0; mask >>= 1) {
    val = f(__shfl_xor(val, mask, n_threads_per_wavefront), val);
  }
  return val;
}

template <typename TData, typename TDataVec>
__forceinline__ __device__ void load_v(
    const TData* __restrict__ data_ptr,
    int32_t vector_offset,
    TDataVec* __restrict__ load_to) {
      *load_to = *(reinterpret_cast<const TDataVec*>(data_ptr) + vector_offset);
}

template <typename TData, typename TDataVec>
__forceinline__ __device__ void store_v(
    TData* __restrict__ data_ptr,
    int32_t vector_offset,
    TDataVec value) {
      *(reinterpret_cast<TDataVec*>(data_ptr) + vector_offset) = value;
}

template <
    typename scalar_t,
    int32_t n_loop_unroll = 16,
    int32_t n_loop_unroll_tail = 2,
    int32_t T_MAX = 8192,
    int32_t n_wavefronts_per_block = 16>
__global__ void efficient_attention_forward_decoder_ck_kernel(
    const scalar_t* __restrict__ XQ,
    const scalar_t* __restrict__ cache_K,
    const scalar_t* __restrict__ cache_V,
    scalar_t* __restrict__ O,
    const int32_t* __restrict__ seq_positions,
    const ptrdiff_t XQ_stride_0,
    const ptrdiff_t XQ_stride_2,
    const ptrdiff_t K_stride_0,
    const ptrdiff_t K_stride_1,
    const ptrdiff_t K_stride_2,
    const int32_t D_H,
    const bool multiquery,
    const float qk_scale) {
  static_assert(n_loop_unroll_tail < n_loop_unroll, "");

  constexpr int32_t seq_positions_shift = 0;

  // Each block handles a single batch and head
  const int32_t b = blockIdx.x;
  const int32_t h = blockIdx.y;

  // Note: this is decoding case where we attend to current and all previous
  // tokens.
  const int32_t t_max = seq_positions[b] + seq_positions_shift;

  const int32_t lane_idx = threadIdx.x;
  const int32_t wavefront_idx = threadIdx.y;
  const int32_t threads_per_wavefront = blockDim.x;
  const int32_t wavefronts_per_block = blockDim.y;
  const int32_t threads_per_block =
      threads_per_wavefront * wavefronts_per_block;
  const int32_t thread_linear_idx =
      lane_idx + wavefront_idx * threads_per_wavefront;

  // const auto* q_ = &(XQ_acc[b][0][h][0]);
  const auto XQO_base_offset = b * XQ_stride_0 + h * XQ_stride_2;
  const auto* q_ = XQ + XQO_base_offset;

  const auto cache_KV_base_offset =
      b * K_stride_0 + (multiquery ? 0 : h * K_stride_2);
  const auto* cache_K_base = cache_K + cache_KV_base_offset;
  const auto* cache_V_base = cache_V + cache_KV_base_offset;

  // Load Q into registers in all wavefronts.
  // Each thread handles 4 D dimensions
  
  constexpr int32_t vec_size = 4;
  using data_t = scalar_t;
  using data_vec_t = typename ck::vector_type<data_t, vec_size>::type;
  using compute_t = float;
  using compute_vec_t = typename ck::vector_type<compute_t, vec_size>::type;

  extern __shared__ __align__(16) compute_t smem[];

  data_vec_t q_thread;
  load_v<data_t, data_vec_t>(q_, lane_idx, &q_thread);
  // Each block computes different B value
  compute_t max_qk_acc = ck::NumericLimits<compute_t>::Lowest();

  // Compute S[T_MAX] = for i in range(T): S[t] = sum(Q[d] * K[t, d])
  // Split T across wavefronts in a block, unroll loads to expose more
  // parallelism.

  data_vec_t k_loads[n_loop_unroll];

  constexpr auto dtt = n_wavefronts_per_block * n_loop_unroll;
  const int32_t t_max_unroll = (t_max / dtt) * dtt;

  for (auto tt = wavefront_idx * n_loop_unroll; tt < t_max_unroll; tt += dtt) {
#pragma unroll n_loop_unroll
    for (auto ttt = 0; ttt < n_loop_unroll; ++ttt) {
      const int32_t t = tt + ttt;
      // load the K[b][t][h|0][:] row into registers
      load_v<data_t, data_vec_t>(
          cache_K_base + t * K_stride_1, lane_idx, &k_loads[ttt]);
    }
    compute_t qk_accs[n_loop_unroll] = {};
#pragma unroll n_loop_unroll
    for (auto ttt = 0; ttt < n_loop_unroll; ++ttt) {
      ck::inner_product<data_vec_t, data_vec_t, compute_t>(
          q_thread, k_loads[ttt], qk_accs[ttt]);
      qk_accs[ttt] *= qk_scale;

      qk_accs[ttt] =
          wavefrontReduce(qk_accs[ttt], [](auto a, auto b) { return a + b; });
      max_qk_acc = max(qk_accs[ttt], max_qk_acc);
    }
    if (lane_idx == 0) {
      auto* smem_base = smem + tt;
#pragma unroll n_loop_unroll
      for (auto ttt = 0; ttt < n_loop_unroll; ++ttt) {
        smem_base[ttt] = qk_accs[ttt];
      }
    }
  }

  // NB: the length of the tail is <= (wavefronts_per_block * n_loop_unroll)
  for (auto tt = t_max_unroll + wavefront_idx * n_loop_unroll_tail; tt < t_max;
       tt += wavefronts_per_block * n_loop_unroll_tail) {
#pragma unroll n_loop_unroll_tail
    for (auto ttt = 0; ttt < n_loop_unroll_tail; ++ttt) {
      const int32_t t = tt + ttt;
      if (t < t_max) {
        // load the K[b][t][h|0][:] row into registers
        load_v<data_t, data_vec_t>(
            cache_K_base + t * K_stride_1, lane_idx, &k_loads[ttt]);
      }
    }
#pragma unroll n_loop_unroll_tail
    for (auto ttt = 0; ttt < n_loop_unroll_tail; ++ttt) {
      compute_t qk_acc = 0;
      const int32_t t = tt + ttt;
      if (t < t_max) {
        ck::inner_product<data_vec_t, data_vec_t, compute_t>(
            q_thread, k_loads[ttt], qk_acc);
        qk_acc *= qk_scale;

        qk_acc =
            wavefrontReduce(qk_acc, [](auto a, auto b) { return a + b; });
        max_qk_acc = max(qk_acc, max_qk_acc);

        // write accumulated sums to smem.
        if (lane_idx == 0) {
          smem[t] = qk_acc;
        }
      }
    }
  }

  // Use shared reduction to compute max and compute softmax on shared memory.
  // write max acc
  if (lane_idx == 0) {
    smem[T_MAX + wavefront_idx] = max_qk_acc;
  }
  __syncthreads();
  if (lane_idx < wavefronts_per_block) {
    max_qk_acc = max(max_qk_acc, smem[T_MAX + lane_idx]);
  }
  // shared across all threads in block
  max_qk_acc = wavefrontReduce(
      max_qk_acc, [](auto a, auto b) { return a > b ? a : b; });

  // each wavefront computes partial sum of exp.
  compute_t softmax_denominator = 0.0f;
  for (int32_t t = thread_linear_idx; t < t_max; t += threads_per_block) {
    softmax_denominator += __expf(smem[t] - max_qk_acc);
  }
  softmax_denominator = wavefrontReduce(
      softmax_denominator, [](auto a, auto b) { return a + b; });

  __syncthreads();
  if (lane_idx == 0) {
    smem[T_MAX + wavefront_idx] = softmax_denominator;
  }
  __syncthreads();

  // now, compute sum of exp(x - max(x)) over all intermediate results.
  softmax_denominator = 0.0;
  if (lane_idx < wavefronts_per_block) {
    softmax_denominator = smem[T_MAX + lane_idx];
  }
  softmax_denominator = wavefrontReduce(
      softmax_denominator, [](auto a, auto b) { return a + b; });

  const compute_t softmax_scale_factor = 1. / softmax_denominator;
  // now, compute the normalization across all threads.
  for (int32_t t = thread_linear_idx; t < t_max; t += threads_per_block) {
    smem[t] = __expf(smem[t] - max_qk_acc) * softmax_scale_factor;
  }
  __syncthreads();

  // Now, we can compute the softmax and write the outputs.

  // Split T across wavefronts in a block
  // each wavefront compute sum(t_subset) P[t] * V[t_subset, d]
  // outputs are of size float[D]

  compute_t ps[n_loop_unroll];
  compute_vec_t o_acc = 0;
  for (auto tt = wavefront_idx * n_loop_unroll; tt < t_max_unroll; tt += dtt) {
#pragma unroll n_loop_unroll
    for (auto ttt = 0; ttt < n_loop_unroll; ++ttt) {
      const int32_t t = tt + ttt;
      // load the V[b][t][h|0][:] row into registers, reusing K register storage
      load_v<data_t, data_vec_t>(
          cache_V_base + t * K_stride_1, lane_idx, &k_loads[ttt]);
      ps[ttt] = smem[t];
    }

#pragma unroll n_loop_unroll
    for (auto ttt = 0; ttt < n_loop_unroll; ++ttt) {
      o_acc = scalar4_scale_acc<data_vec_t>(o_acc, k_loads[ttt], ps[ttt]);
    }
  }

  for (auto tt = t_max_unroll + wavefront_idx * n_loop_unroll_tail; tt < t_max;
       tt += wavefronts_per_block * n_loop_unroll_tail) {
#pragma unroll n_loop_unroll_tail
    for (auto ttt = 0; ttt < n_loop_unroll_tail; ++ttt) {
      const int32_t t = tt + ttt;
      if (t < t_max) {
        // load the V[b][t][h|0][:] row into registers, reusing K register
        // storage
        load_v<data_t, data_vec_t>(
            cache_V_base + t * K_stride_1, lane_idx, &k_loads[ttt]);
        ps[ttt] = smem[t];
      }
    }

#pragma unroll n_loop_unroll_tail
    for (auto ttt = 0; ttt < n_loop_unroll_tail; ++ttt) {
      const int32_t t = tt + ttt;
      if (t < t_max) {
        o_acc = scalar4_scale_acc<data_vec_t>(o_acc, k_loads[ttt], ps[ttt]);
      }
    }
  }
  // now, each thread has partial sums. Write to smem and get accumulated
  // results back.
  __syncthreads();

  // NB: needs sizeof(smem) >= 4 * (sizeof(float)==4) * threadsPerBlock
  store_v<compute_t, compute_vec_t>(&smem[0], thread_linear_idx, o_acc);

  __syncthreads();
  // sum up partial D rows from other wavefronts
  if (wavefront_idx == 0) {
    union { compute_vec_t vec; compute_t[vec_size] arr; } r = 0;
    for (int32_t w = 0; w < wavefronts_per_block; ++w) {
      compute_vec_t partial_r;
      load_v<compute_t, compute_vec_t>(
          smem, w * threads_per_wavefront + lane_idx, &partial_r);
      r.vec += partial_r;
    }
    // elementwise convert from compute_t result to data_t out to be written
    union { data_vec_t vec; data_t[vec_size] arr; } bf_r = 0;
    for (int32_t i = 0; i < vec_size; ++i) {
      bf_r.arr[i] = ck::type_convert<data_t>(r.arr[i]);
    }
    // write output D row
    data_t* __restrict__ o_ = O + XQO_base_offset;
    store_v<data_t, data_vec_t>(o_, lane_idx, bf_r);
  }
}

} // namespace

namespace ck {
namespace tensor_operation {
namespace device {
template <typename scalar_t>
struct FMHADecoderSeqlen1DeviceOp : public BaseOperator {
  using DeviceOp = FMHADecoderSeqlen1DeviceOp;
  struct Argument : public BaseArgument {
    const scalar_t* __restrict__ XQ;
    const scalar_t* __restrict__ cache_K;
    const scalar_t* __restrict__ cache_V;
    scalar_t* __restrict__ O;
    const int32_t* __restrict__ seq_positions;
    const ptrdiff_t XQ_stride_0;
    const ptrdiff_t XQ_stride_2;
    const ptrdiff_t K_stride_0;
    const ptrdiff_t K_stride_1;
    const ptrdiff_t K_stride_2;
    const int32_t D_H;
    const bool multiquery;
    const float qk_scale;

    const dim3 grid_dim;
    const dim3 block_dim;
    const size_t lds_bytes;

    Argument(
        const scalar_t* __restrict__ XQ,
        const scalar_t* __restrict__ cache_K,
        const scalar_t* __restrict__ cache_V,
        scalar_t* __restrict__ O,
        const int32_t* __restrict__ seq_positions,
        const ptrdiff_t XQ_stride_0,
        const ptrdiff_t XQ_stride_2,
        const ptrdiff_t K_stride_0,
        const ptrdiff_t K_stride_1,
        const ptrdiff_t K_stride_2,
        const int32_t D_H,
        const bool multiquery,
        const float qk_scale,
        const dim3 grid_dim,
        const dim3 block_dim,
        const size_t lds_bytes)
        : XQ(XQ),
          cache_K(cache_K),
          cache_V(cache_V),
          O(O),
          seq_positions(seq_positions),
          XQ_stride_0(XQ_stride_0),
          XQ_stride_2(XQ_stride_2),
          K_stride_0(K_stride_0),
          K_stride_1(K_stride_1),
          K_stride_2(K_stride_2),
          D_H(D_H),
          multiquery(multiquery),
          qk_scale(qk_scale),
          grid_dim(grid_dim),
          block_dim(block_dim),
          lds_bytes(lds_bytes) {}
  };
  struct Invoker : public BaseInvoker {
    using Argument = DeviceOp::Argument;
    float Run(
        const Argument& arg,
        const StreamConfig& stream_config = StreamConfig{}) {
      return launch_and_time_kernel(
          stream_config,
          efficient_attention_forward_decoder_ck_kernel<scalar_t>,
          arg.grid_dim,
          arg.block_dim,
          arg.lds_bytes,
          arg.XQ,
          arg.cache_K,
          arg.cache_V,
          arg.O,
          arg.seq_positions,
          arg.XQ_stride_0,
          arg.XQ_stride_2,
          arg.K_stride_0,
          arg.K_stride_1,
          arg.K_stride_2,
          arg.D_H,
          arg.multiquery,
          arg.qk_scale);
    }
  };
};
} // namespace device
} // namespace tensor_operation
} // namespace ck