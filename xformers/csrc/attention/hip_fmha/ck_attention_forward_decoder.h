#pragma once 

#include <ck/host_utility/kernel_launch_hip.hpp>
#include <ck/stream_config.hpp>
#include <ck/tensor_operation/gpu/device/device_base.hpp>
#include <ck/utility/data_type.hpp>
#include <ck/utility/inner_product.hpp>
#include <ck/utility/math.hpp>

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

template <typename data_t, int32_t vec_size>
__device__ 
typename ck::vector_type<float, vec_size>::type
scalar_scale_acc(typename ck::vector_type<float, vec_size>::type acc, 
                 typename ck::vector_type<data_t, vec_size>::type a, 
                 float b) {
  
  union { decltype(acc) vec; float arr[vec_size]; } acc_u;
  union { decltype(a) vec; data_t arr[vec_size]; } a_u;

  acc_u.vec = acc;
  a_u.vec = a;

  #pragma unroll
  for (int32_t i = 0; i < vec_size; ++i) {
    acc_u.arr[i] += ck::type_convert<float>(a_u.arr[i]) * b;
  }

  return acc_u.vec;
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
    int32_t vec_size = 4,
    int32_t n_loop_unroll = 16,
    int32_t n_loop_unroll_tail = 2,
    int32_t T_MAX = 8192,
    int32_t n_wavefronts_per_block = 16>
__global__ void efficient_attention_forward_decoder_ck_kernel(
    const scalar_t* __restrict__ XQ,
    const scalar_t* __restrict__ cache_K,
    const scalar_t* __restrict__ cache_V,
    scalar_t* __restrict__ O,
    const int32_t* __restrict__ seq_kv_lens,
    const ptrdiff_t XQ_stride_0,
    const ptrdiff_t XQ_stride_1,
    const ptrdiff_t XQ_stride_2,
    const ptrdiff_t K_stride_0,
    const ptrdiff_t K_stride_1,
    const ptrdiff_t K_stride_2,
    const int32_t D_H,
    const bool multiquery,
    const float qk_scale) {
  static_assert(n_loop_unroll_tail < n_loop_unroll, "");

  // Each block handles a single batch and head and query
  const int32_t b = blockIdx.x;
  const int32_t h = blockIdx.y;
  const int32_t m = blockIdx.z;

  // Note: this is decoding case where we attend to current and all previous
  // tokens.
  const int32_t t_max = seq_kv_lens[b];

  const int32_t lane_idx = threadIdx.x;
  const int32_t wavefront_idx = threadIdx.y;
  const int32_t threads_per_wavefront = blockDim.x;
  const int32_t wavefronts_per_block = blockDim.y;
  const int32_t threads_per_block =
      threads_per_wavefront * wavefronts_per_block;
  const int32_t thread_linear_idx =
      lane_idx + wavefront_idx * threads_per_wavefront;
  // const auto* q_ = &(XQ_acc[b][m][h][0]);
  const auto XQO_base_offset = b * XQ_stride_0 + m * XQ_stride_1 + h * XQ_stride_2;
  const auto* __restrict__ q_ = XQ + XQO_base_offset;

  const auto cache_KV_base_offset =
      b * K_stride_0 + (multiquery ? 0 : h * K_stride_2);
  const auto* __restrict__ cache_K_base = cache_K + cache_KV_base_offset;
  const auto* __restrict__ cache_V_base = cache_V + cache_KV_base_offset;

  // Load Q into registers in all wavefronts.
  // Each thread handles 4 D dimensions
  
  using data_t = scalar_t;
  using data_vec_t = typename ck::vector_type<data_t, vec_size>::type;
  using compute_t = float;
  using compute_vec_t = typename ck::vector_type<compute_t, vec_size>::type;

  const bool lane_active_for_io = lane_idx * vec_size < D_H;

  extern __shared__ __align__(16) compute_t smem[];

  data_vec_t q_thread = 0;
  if (lane_active_for_io) {
    load_v<data_t, data_vec_t>(q_, lane_idx, &q_thread);
  } 
  // Each block computes different B value
  compute_t max_qk_acc = ck::NumericLimits<compute_t>::Lowest();

  // Compute S[T_MAX] = for i in range(T): S[t] = sum(Q[d] * K[t, d])
  // Split T across wavefronts in a block, unroll loads to expose more
  // parallelism.

  data_vec_t k_loads[n_loop_unroll] = {};

  constexpr auto dtt = n_wavefronts_per_block * n_loop_unroll;
  const int32_t t_max_unroll = (t_max / dtt) * dtt;

  for (auto tt = wavefront_idx * n_loop_unroll; tt < t_max_unroll; tt += dtt) {
#pragma unroll n_loop_unroll
    for (auto ttt = 0; ttt < n_loop_unroll; ++ttt) {
      const int32_t t = tt + ttt;
      // load the K[b][t][h|0][:] row into registers
      if (lane_active_for_io) {
        load_v<data_t, data_vec_t>(
            cache_K_base + t * K_stride_1, lane_idx, &k_loads[ttt]);
      } 
    }
    compute_t qk_accs[n_loop_unroll] = {};
#pragma unroll n_loop_unroll
    for (auto ttt = 0; ttt < n_loop_unroll; ++ttt) {
      ck::inner_product<data_vec_t, data_vec_t, compute_t>(
          q_thread, k_loads[ttt], qk_accs[ttt]);
      qk_accs[ttt] *= qk_scale;

      qk_accs[ttt] =
          wavefrontReduce(qk_accs[ttt], [](auto a, auto b) { return a + b; });
      max_qk_acc = ck::math::max(qk_accs[ttt], max_qk_acc);
    }
    if (lane_idx == 0) {
      auto* __restrict__ smem_base = smem + tt;
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
        if (lane_active_for_io) {
          // load the K[b][t][h|0][:] row into registers
          load_v<data_t, data_vec_t>(
              cache_K_base + t * K_stride_1, lane_idx, &k_loads[ttt]);
        } 
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
        max_qk_acc = ck::math::max(qk_acc, max_qk_acc);

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
    max_qk_acc = ck::math::max(max_qk_acc, smem[T_MAX + lane_idx]);
  }
  // shared across all threads in block
  max_qk_acc = wavefrontReduce(
      max_qk_acc, [](auto a, auto b) { return a > b ? a : b; });

  // each wavefront computes partial sum of exp.
  compute_t softmax_denominator = 0.0f;
  for (int32_t t = thread_linear_idx; t < t_max; t += threads_per_block) {
    softmax_denominator += ck::math::exp(smem[t] - max_qk_acc);
  }
  softmax_denominator = wavefrontReduce(
      softmax_denominator, [](auto a, auto b) { return a + b; });

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
    smem[t] = ck::math::exp(smem[t] - max_qk_acc) * softmax_scale_factor;
  }
  __syncthreads();

  // Split T across wavefronts in a block
  // each wavefront compute sum(t_subset) P[t] * V[t_subset, d]
  // outputs are of size float[D]

  compute_t ps[n_loop_unroll];
  compute_vec_t o_acc = 0;
  for (auto tt = wavefront_idx * n_loop_unroll; tt < t_max_unroll; tt += dtt) {
#pragma unroll n_loop_unroll
    for (auto ttt = 0; ttt < n_loop_unroll; ++ttt) {
      const int32_t t = tt + ttt;
      if (lane_active_for_io) {
        // load the V[b][t][h|0][:] row into registers, reusing K register storage
        load_v<data_t, data_vec_t>(
            cache_V_base + t * K_stride_1, lane_idx, &k_loads[ttt]);
      } 
      ps[ttt] = smem[t];
    }

#pragma unroll n_loop_unroll
    for (auto ttt = 0; ttt < n_loop_unroll; ++ttt) {
      o_acc = scalar_scale_acc<data_t, vec_size>(o_acc, k_loads[ttt], ps[ttt]);
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
        if (lane_active_for_io) {
          load_v<data_t, data_vec_t>(
              cache_V_base + t * K_stride_1, lane_idx, &k_loads[ttt]);
        } 
        ps[ttt] = smem[t];
      }
    }

#pragma unroll n_loop_unroll_tail
    for (auto ttt = 0; ttt < n_loop_unroll_tail; ++ttt) {
      const int32_t t = tt + ttt;
      if (t < t_max) {
        o_acc = scalar_scale_acc<data_t, vec_size>(o_acc, k_loads[ttt], ps[ttt]);
      }
    }
  }
  // now, each thread has partial sums. Write to smem and get accumulated
  // results back.
  __syncthreads();

  // NB: needs sizeof(smem) >= 4 * (sizeof(float)==4) * threadsPerBlock
  if (lane_active_for_io) {
    store_v<compute_t, compute_vec_t>(&smem[0], thread_linear_idx, o_acc);
  }

  __syncthreads();
  // sum up partial D rows from other wavefronts
  if (wavefront_idx == 0 && lane_active_for_io) {
    union { compute_vec_t vec = 0; compute_t arr[vec_size]; } r;
    for (int32_t w = 0; w < wavefronts_per_block; ++w) {
      compute_vec_t partial_r;
      load_v<compute_t, compute_vec_t>(
          smem, w * threads_per_wavefront + lane_idx, &partial_r);
      r.vec += partial_r;
    }
    // elementwise convert from compute_t result to data_t out to be written
    union { data_vec_t vec; data_t arr[vec_size]; } bf_r;
    #pragma unroll 
    for (int32_t i = 0; i < vec_size; ++i) {
      bf_r.arr[i] = ck::type_convert<data_t>(r.arr[i]);
    }
    // write output row O[b][m][h][:]
    data_t* __restrict__ o_ = O + XQO_base_offset;
    store_v<data_t, data_vec_t>(o_, lane_idx, bf_r.vec);
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
    const int32_t* __restrict__ seq_kv_lens;
    const ptrdiff_t XQ_stride_0;
    const ptrdiff_t XQ_stride_1;
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
        const int32_t* __restrict__ seq_kv_lens,
        const ptrdiff_t XQ_stride_0,
        const ptrdiff_t XQ_stride_1,
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
          seq_kv_lens(seq_kv_lens),
          XQ_stride_0(XQ_stride_0),
          XQ_stride_1(XQ_stride_1),
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
          arg.seq_kv_lens,
          arg.XQ_stride_0,
          arg.XQ_stride_1,
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