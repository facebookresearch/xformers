/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ck/host_utility/kernel_launch.hpp>
#include <ck/stream_config.hpp>
#include <ck/tensor_operation/gpu/device/device_base.hpp>
#include <ck/utility/data_type.hpp>
#include <ck/utility/math.hpp>

#include "ck_attention_inner_product.h"
#include "ck_attention_math_ext.h"

namespace {

template <typename data_t, int32_t vec_size>
__device__ typename ck::vector_type<float, vec_size>::type scalar_scale_acc(
    typename ck::vector_type<float, vec_size>::type acc,
    typename ck::vector_type<data_t, vec_size>::type a,
    float b) {
  union {
    decltype(acc) vec;
    float arr[vec_size];
  } acc_u{acc};
  union {
    decltype(a) vec;
    data_t arr[vec_size];
  } a_u{a};

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
    int32_t KV_M_MAX = 8192,
    int32_t n_wavefronts_per_block = 16>
__global__ void efficient_attention_forward_decoder_ck_kernel(
    const scalar_t* __restrict__ XQ,
    const scalar_t* __restrict__ cache_K,
    const scalar_t* __restrict__ cache_V,
    scalar_t* __restrict__ O,
    const int32_t* __restrict__ seq_kv_lens,
    const ptrdiff_t XQ_stride_b,
    const ptrdiff_t XQ_stride_m,
    const ptrdiff_t XQ_stride_g,
    const ptrdiff_t XQ_stride_h,
    const ptrdiff_t K_stride_b,
    const ptrdiff_t K_stride_m,
    const ptrdiff_t K_stride_g,
    const ptrdiff_t K_stride_h,
    const int32_t Q_size_m,
    const int32_t Q_size_g,
    const int32_t Q_size_h,
    const int32_t Q_size_k,
    const int32_t K_size_m,
    const bool multiquery,
    const float qk_scale) {
  static_assert(n_loop_unroll_tail < n_loop_unroll, "");

  // Each block handles a single batch and head and query and group
  const int32_t b = blockIdx.x / (Q_size_m * Q_size_g * Q_size_h);
  const int32_t m = (blockIdx.x / (Q_size_g * Q_size_h)) % Q_size_m;
  const int32_t g = (blockIdx.x / Q_size_h) % Q_size_g;
  const int32_t h = blockIdx.x % Q_size_h;

  // Note: this is decoding case where we attend to current and all previous
  // tokens.
  const int32_t t_max = seq_kv_lens ? seq_kv_lens[b] : K_size_m;

  const int32_t lane_idx = threadIdx.x;
  const int32_t wavefront_idx = threadIdx.y;
  const int32_t threads_per_wavefront = blockDim.x;
  const int32_t wavefronts_per_block = blockDim.y;
  const int32_t threads_per_block =
      threads_per_wavefront * wavefronts_per_block;
  const int32_t thread_linear_idx =
      lane_idx + wavefront_idx * threads_per_wavefront;
  // const auto* q_ = &(XQ_acc[b][m][g][h][0]);
  const auto XQO_base_offset =
      b * XQ_stride_b + m * XQ_stride_m + g * XQ_stride_g + h * XQ_stride_h;
  const auto* __restrict__ q_ = XQ + XQO_base_offset;

  const auto cache_KV_base_offset = b * K_stride_b + 0 * K_stride_m +
      g * K_stride_g + (multiquery ? 0 : h * K_stride_h);
  const auto* __restrict__ cache_K_base = cache_K + cache_KV_base_offset;
  const auto* __restrict__ cache_V_base = cache_V + cache_KV_base_offset;

  using data_t = scalar_t;
  using data_vec_t = typename ck::vector_type<data_t, vec_size>::type;
  using compute_t = float;
  using compute_vec_t = typename ck::vector_type<compute_t, vec_size>::type;

  const bool lane_active_for_io = lane_idx * vec_size < Q_size_k;

  extern __shared__ __align__(16) compute_t smem[];

  data_vec_t q_thread = 0;
  // Load Q into registers in all wavefronts.
  // Each thread handles `vec_size` D dimensions
  if (lane_active_for_io) {
    load_v<data_t, data_vec_t>(q_, lane_idx, &q_thread);
  }

  compute_t max_qk_acc = ck::NumericLimits<compute_t>::Lowest();

  // Compute S[0:t_max] =
  // ```
  // for t in range(t_max):
  //   S[t] = dot(Q, K[t])
  // ```
  // Split the 0:t_max range across wavefronts in a block,
  // unroll loads to expose more parallelism.
  // Reduce the dot product with cross-lane operation;
  // Q and K[t] are in the registers of threads in a single wavefront.

  data_vec_t k_loads[n_loop_unroll] = {};

  constexpr auto dtt = n_wavefronts_per_block * n_loop_unroll;
  const int32_t t_max_unroll = (t_max / dtt) * dtt;

  for (auto tt = wavefront_idx * n_loop_unroll; tt < t_max_unroll; tt += dtt) {
    if (lane_active_for_io) {
#pragma unroll n_loop_unroll
      for (auto ttt = 0; ttt < n_loop_unroll; ++ttt) {
        const int32_t t = tt + ttt;
        // load the K[b][t][g][h|0][:] row into registers
        load_v<data_t, data_vec_t>(
            cache_K_base + t * K_stride_m, lane_idx, &k_loads[ttt]);
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
    if (lane_active_for_io) {
#pragma unroll n_loop_unroll_tail
      for (auto ttt = 0; ttt < n_loop_unroll_tail; ++ttt) {
        const int32_t t = tt + ttt;
        if (t < t_max) {
          // load the K[b][t][g][h|0][:] row into registers
          load_v<data_t, data_vec_t>(
              cache_K_base + t * K_stride_m, lane_idx, &k_loads[ttt]);
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

        qk_acc = wavefrontReduce(qk_acc, [](auto a, auto b) { return a + b; });
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
    smem[KV_M_MAX + wavefront_idx] = max_qk_acc;
  }
  __syncthreads();
  if (lane_idx < wavefronts_per_block) {
    max_qk_acc = ck::math::max(max_qk_acc, smem[KV_M_MAX + lane_idx]);
  }
  // shared across all threads in block
  max_qk_acc =
      wavefrontReduce(max_qk_acc, [](auto a, auto b) { return a > b ? a : b; });

  // each wavefront computes partial sum of exp.
  compute_t softmax_denominator = 0.0f;
  for (int32_t t = thread_linear_idx; t < t_max; t += threads_per_block) {
    softmax_denominator += ck::math::exp(smem[t] - max_qk_acc);
  }
  softmax_denominator = wavefrontReduce(
      softmax_denominator, [](auto a, auto b) { return a + b; });

  if (lane_idx == 0) {
    smem[KV_M_MAX + wavefront_idx] = softmax_denominator;
  }
  __syncthreads();

  // now, compute sum of exp(x - max(x)) over all intermediate results.
  softmax_denominator = 0.0;
  if (lane_idx < wavefronts_per_block) {
    softmax_denominator = smem[KV_M_MAX + lane_idx];
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

  compute_t ps[n_loop_unroll] = {};
  compute_vec_t o_acc = 0;
  if (lane_active_for_io) {
    for (auto tt = wavefront_idx * n_loop_unroll; tt < t_max_unroll;
         tt += dtt) {
#pragma unroll n_loop_unroll
      for (auto ttt = 0; ttt < n_loop_unroll; ++ttt) {
        const int32_t t = tt + ttt;
        // load the V[b][t][g][h|0][:] row into registers, reusing K register
        // storage
        load_v<data_t, data_vec_t>(
            cache_V_base + t * K_stride_m, lane_idx, &k_loads[ttt]);
        ps[ttt] = smem[t];
      }

#pragma unroll n_loop_unroll
      for (auto ttt = 0; ttt < n_loop_unroll; ++ttt) {
        o_acc =
            scalar_scale_acc<data_t, vec_size>(o_acc, k_loads[ttt], ps[ttt]);
      }
    }

    for (auto tt = t_max_unroll + wavefront_idx * n_loop_unroll_tail;
         tt < t_max;
         tt += wavefronts_per_block * n_loop_unroll_tail) {
#pragma unroll n_loop_unroll_tail
      for (auto ttt = 0; ttt < n_loop_unroll_tail; ++ttt) {
        const int32_t t = tt + ttt;
        if (t < t_max) {
          // load the V[b][t][g][h|0][:] row into registers, reusing K register
          // storage
          load_v<data_t, data_vec_t>(
              cache_V_base + t * K_stride_m, lane_idx, &k_loads[ttt]);
          ps[ttt] = smem[t];
        }
      }

#pragma unroll n_loop_unroll_tail
      for (auto ttt = 0; ttt < n_loop_unroll_tail; ++ttt) {
        const int32_t t = tt + ttt;
        if (t < t_max) {
          o_acc =
              scalar_scale_acc<data_t, vec_size>(o_acc, k_loads[ttt], ps[ttt]);
        }
      }
    }
  }
  // now, each thread has partial sums. Write to smem and get accumulated
  // results back.
  __syncthreads();

  // NB: needs sizeof(smem) >= `vec_size` * (sizeof(float)==4) * threadsPerBlock
  if (lane_active_for_io) {
    store_v<compute_t, compute_vec_t>(&smem[0], thread_linear_idx, o_acc);
  }

  __syncthreads();
  // sum up partial D rows from other wavefronts
  if (wavefront_idx == 0 && lane_active_for_io) {
    union {
      compute_vec_t vec = 0;
      compute_t arr[vec_size];
    } r;
    for (int32_t w = 0; w < wavefronts_per_block; ++w) {
      compute_vec_t partial_r;
      load_v<compute_t, compute_vec_t>(
          smem, w * threads_per_wavefront + lane_idx, &partial_r);
      r.vec += partial_r;
    }
    // elementwise convert from compute_t result to data_t out to be written
    union {
      data_vec_t vec;
      data_t arr[vec_size];
    } bf_r;
#pragma unroll
    for (int32_t i = 0; i < vec_size; ++i) {
      bf_r.arr[i] = ck::type_convert<data_t>(r.arr[i]);
    }
    // write output row O[b][m][g][h][:]
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
    const ptrdiff_t XQ_stride_b;
    const ptrdiff_t XQ_stride_m;
    const ptrdiff_t XQ_stride_g;
    const ptrdiff_t XQ_stride_h;
    const ptrdiff_t K_stride_b;
    const ptrdiff_t K_stride_m;
    const ptrdiff_t K_stride_g;
    const ptrdiff_t K_stride_h;
    const int32_t Q_size_m;
    const int32_t Q_size_g;
    const int32_t Q_size_h;
    const int32_t Q_size_k;
    const int32_t K_size_m;
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
        const ptrdiff_t XQ_stride_b,
        const ptrdiff_t XQ_stride_m,
        const ptrdiff_t XQ_stride_g,
        const ptrdiff_t XQ_stride_h,
        const ptrdiff_t K_stride_b,
        const ptrdiff_t K_stride_m,
        const ptrdiff_t K_stride_g,
        const ptrdiff_t K_stride_h,
        const int32_t Q_size_m,
        const int32_t Q_size_g,
        const int32_t Q_size_h,
        const int32_t Q_size_k,
        const int32_t K_size_m,
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
          XQ_stride_b(XQ_stride_b),
          XQ_stride_m(XQ_stride_m),
          XQ_stride_g(XQ_stride_g),
          XQ_stride_h(XQ_stride_h),
          K_stride_b(K_stride_b),
          K_stride_m(K_stride_m),
          K_stride_g(K_stride_g),
          K_stride_h(K_stride_h),
          Q_size_m(Q_size_m),
          Q_size_g(Q_size_g),
          Q_size_h(Q_size_h),
          Q_size_k(Q_size_k),
          K_size_m(K_size_m),
          multiquery(multiquery),
          qk_scale(qk_scale),
          grid_dim(grid_dim),
          block_dim(block_dim),
          lds_bytes(lds_bytes) {}
  };

  struct Invoker : public BaseInvoker {
    using Argument = DeviceOp::Argument;
    float Run(
        const BaseArgument* argp_,
        const StreamConfig& stream_config = StreamConfig{}) {
      const Argument* argp = dynamic_cast<const Argument*>(argp_);

      auto threads_per_wavefront = argp->block_dim.x;

      auto Q_size_k_alignment_necessary = 0;

      for (auto vec_size : {4, 2, 1}) {
        if (argp->Q_size_k <= vec_size * threads_per_wavefront) {
          Q_size_k_alignment_necessary = vec_size;
        }
      }

      if (!Q_size_k_alignment_necessary) {
        throw std::runtime_error("Unsupported Q_size_k");
      }

      if (argp->Q_size_k % Q_size_k_alignment_necessary) {
        throw std::runtime_error("Unsupported alignment for Q_size_k");
      }

      return launch_and_time_kernel(
          stream_config,
          Q_size_k_alignment_necessary == 4
              ? efficient_attention_forward_decoder_ck_kernel<scalar_t, 4>
              : Q_size_k_alignment_necessary == 2
              ? efficient_attention_forward_decoder_ck_kernel<scalar_t, 2>
              : Q_size_k_alignment_necessary == 1
              ? efficient_attention_forward_decoder_ck_kernel<scalar_t, 1>
              : nullptr,
          argp->grid_dim,
          argp->block_dim,
          argp->lds_bytes,
          argp->XQ,
          argp->cache_K,
          argp->cache_V,
          argp->O,
          argp->seq_kv_lens,
          argp->XQ_stride_b,
          argp->XQ_stride_m,
          argp->XQ_stride_g,
          argp->XQ_stride_h,
          argp->K_stride_b,
          argp->K_stride_m,
          argp->K_stride_g,
          argp->K_stride_h,
          argp->Q_size_m,
          argp->Q_size_g,
          argp->Q_size_h,
          argp->Q_size_k,
          argp->K_size_m,
          argp->multiquery,
          argp->qk_scale);
    }
  };
};
} // namespace device
} // namespace tensor_operation
} // namespace ck
