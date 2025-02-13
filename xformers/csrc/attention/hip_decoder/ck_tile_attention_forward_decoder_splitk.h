#pragma once

#include <ck_tile/core.hpp>

#include "ck_tile_attention_inner_product.h"

namespace {

template <typename data_t, int32_t vec_size>
__device__ ck_tile::ext_vector_t<float, vec_size> scalar_scale_acc(
    ck_tile::ext_vector_t<float, vec_size> acc,
    ck_tile::ext_vector_t<data_t, vec_size> a,
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
    acc_u.arr[i] += ck_tile::type_convert<float>(a_u.arr[i]) * b;
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

} // namespace

namespace ck_tile {
template <typename scalar_t, typename compute_t>
struct ForwardDecoderSplitKArgument {
  const scalar_t* __restrict__ XQ;
  const scalar_t* __restrict__ cache_K;
  const scalar_t* __restrict__ cache_V;
  scalar_t* __restrict__ O;
  scalar_t* __restrict__ split_O;
  compute_t* __restrict__ split_max;
  compute_t* __restrict__ split_sumexp;
  const int32_t* __restrict__ seq_kv_lens;
  const ptrdiff_t XQ_stride_b;
  const ptrdiff_t XQ_stride_m;
  const ptrdiff_t XQ_stride_g;
  const ptrdiff_t XQ_stride_h;
  const ptrdiff_t K_stride_b;
  const ptrdiff_t K_stride_m;
  const ptrdiff_t K_stride_g;
  const ptrdiff_t K_stride_h;
  const ptrdiff_t O_stride_split;
  const int32_t Q_size_m;
  const int32_t Q_size_g;
  const int32_t Q_size_h;
  const int32_t Q_size_k;
  const int32_t K_size_m;
  const bool multiquery;
  const float qk_scale;
  const int32_t split_k;
};

template <typename scalar_t, int32_t vec_size = 4, typename compute_t = float>
struct ForwardDecoderSplitKReduceKernelImpl {
  CK_TILE_DEVICE void operator()(
      ForwardDecoderSplitKArgument<scalar_t, compute_t> arg) {
    // Each block handles a single batch and head and query and group
    const int32_t b = blockIdx.x / (arg.Q_size_m * arg.Q_size_g * arg.Q_size_h);
    const int32_t m =
        (blockIdx.x / (arg.Q_size_g * arg.Q_size_h)) % arg.Q_size_m;
    const int32_t g = (blockIdx.x / arg.Q_size_h) % arg.Q_size_g;
    const int32_t h = blockIdx.x % arg.Q_size_h;

    using data_t = scalar_t;
    using data_vec_t = ck_tile::ext_vector_t<data_t, vec_size>;
    using compute_vec_t = ck_tile::ext_vector_t<compute_t, vec_size>;

    union {
      data_vec_t vec;
      data_t arr[vec_size];
    } O_split_data;
    union {
      compute_vec_t vec;
      compute_t arr[vec_size];
    } O_split_compute;
    union {
      data_vec_t vec;
      data_t arr[vec_size];
    } global_O_data;
    union {
      compute_vec_t vec;
      compute_t arr[vec_size];
    } global_O_compute;

    global_O_compute.vec = 0;

    const int32_t lane_idx = threadIdx.x;
    const bool lane_active_for_io = lane_idx * vec_size < arg.Q_size_k;

    if (!lane_active_for_io) {
      return;
    }

    compute_t global_sumexp = 0;
    compute_t global_max = ck_tile::numeric<compute_t>::lowest();

    for (int32_t split_idx = 0; split_idx < arg.split_k; ++split_idx) {
      load_v<data_t, data_vec_t>(
          arg.split_O + b * arg.XQ_stride_b + m * arg.XQ_stride_m +
              g * arg.XQ_stride_g + h * arg.XQ_stride_h +
              split_idx * arg.O_stride_split,
          lane_idx,
          &O_split_data.vec);
#pragma unroll
      for (int32_t i = 0; i < vec_size; ++i) {
        O_split_compute.arr[i] =
            ck_tile::type_convert<compute_t>(O_split_data.arr[i]);
      }
      compute_t local_max =
          *(arg.split_max + blockIdx.x * arg.split_k + split_idx);
      compute_t local_sumexp =
          *(arg.split_sumexp + blockIdx.x * arg.split_k + split_idx);

      compute_t log_alpha = -std::abs(local_max - global_max);
      compute_t alpha =
          ck_tile::isnan(log_alpha) ? compute_t{1.} : ck_tile::exp(log_alpha);

      bool pick_new = local_max < global_max;
      compute_t pick_current_coef = pick_new ? 1. : alpha;
      compute_t pick_new_coef = pick_new ? alpha : 1.;

      global_sumexp =
          pick_current_coef * global_sumexp + pick_new_coef * local_sumexp;
      global_O_compute.vec = pick_current_coef * global_O_compute.vec +
          pick_new_coef * O_split_compute.vec;
      global_max = ck_tile::max(local_max, global_max);
    }
    global_O_compute.vec /= global_sumexp;
#pragma unroll
    for (int32_t i = 0; i < vec_size; ++i) {
      global_O_data.arr[i] =
          ck_tile::type_convert<data_t>(global_O_compute.arr[i]);
    }
    store_v<data_t, data_vec_t>(
        arg.O + b * arg.XQ_stride_b + m * arg.XQ_stride_m +
            g * arg.XQ_stride_g + h * arg.XQ_stride_h,
        lane_idx,
        global_O_data.vec);
  }
};

template <
    typename scalar_t,
    int32_t vec_size,
    int32_t n_loop_unroll,
    int32_t n_loop_unroll_tail,
    int32_t KV_M_MAX,
    typename compute_t>
struct ForwardDecoderSplitKAttnKernelImpl {
  CK_TILE_DEVICE void operator()(
      ForwardDecoderSplitKArgument<scalar_t, compute_t> arg) {
    static_assert(
        n_loop_unroll_tail < n_loop_unroll || n_loop_unroll_tail == 1,
        "tail unroll must be smaller than main loop untoll; pragma unroll 0 is illegal "
        "(and tail is no-op)");

    // Each block handles a single batch and head and query and group
    const int32_t b = blockIdx.x / (arg.Q_size_m * arg.Q_size_g * arg.Q_size_h);
    const int32_t m =
        (blockIdx.x / (arg.Q_size_g * arg.Q_size_h)) % arg.Q_size_m;
    const int32_t g = (blockIdx.x / arg.Q_size_h) % arg.Q_size_g;
    const int32_t h = blockIdx.x % arg.Q_size_h;
    const int32_t split_idx = blockIdx.y;

    // Note: this is decoding case where we attend to current and all previous
    // tokens.
    const int32_t t_max = arg.seq_kv_lens ? arg.seq_kv_lens[b] : arg.K_size_m;

    const int32_t lane_idx = threadIdx.x;
    const int32_t wavefront_idx = threadIdx.y;
    // TODO: `threads_per_wavefront` and `wavefronts_per_block` may be compile
    // time constants; investigate when optimizing
    const int32_t threads_per_wavefront = blockDim.x;
    const int32_t wavefronts_per_block = blockDim.y;
    const int32_t threads_per_block =
        threads_per_wavefront * wavefronts_per_block;
    const int32_t thread_linear_idx =
        lane_idx + wavefront_idx * threads_per_wavefront;
    // const auto* q_ = &(XQ_acc[b][m][g][h][0]);
    const auto XQO_base_offset = b * arg.XQ_stride_b + m * arg.XQ_stride_m +
        g * arg.XQ_stride_g + h * arg.XQ_stride_h;
    const auto* __restrict__ q_ = arg.XQ + XQO_base_offset;

    const auto cache_KV_base_offset = b * arg.K_stride_b + 0 * arg.K_stride_m +
        g * arg.K_stride_g + (arg.multiquery ? 0 : h * arg.K_stride_h);
    const auto* __restrict__ cache_K_base = arg.cache_K + cache_KV_base_offset;
    const auto* __restrict__ cache_V_base = arg.cache_V + cache_KV_base_offset;

    using data_t = scalar_t;
    using data_vec_t = std::conditional_t<
        vec_size == 1,
        data_t,
        ck_tile::ext_vector_t<data_t, vec_size>>;
    using compute_vec_t = ck_tile::ext_vector_t<compute_t, vec_size>;

    const bool lane_active_for_io = lane_idx * vec_size < arg.Q_size_k;

    extern __shared__ __align__(16) compute_t smem[];

    data_vec_t q_thread = 0;
    // Load Q into registers in all wavefronts.
    // Each thread handles `vec_size` D dimensions
    if (lane_active_for_io) {
      load_v<data_t, data_vec_t>(q_, lane_idx, &q_thread);
    }

    compute_t max_qk_acc = ck_tile::numeric<compute_t>::lowest();

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

    const auto dtt = wavefronts_per_block * n_loop_unroll;
    // only last split gets the tail.
    // the first (split_k - 1) splits have a number of iterations divisible by
    // `dtt`
    const auto n_unrolled_loops = t_max / dtt / arg.split_k; // +1?
    const int32_t tt_low =
        wavefront_idx * n_loop_unroll + n_unrolled_loops * dtt * split_idx;
    const int32_t tt_high = wavefront_idx * n_loop_unroll +
        n_unrolled_loops * dtt * (split_idx + 1);
    const int32_t dtt_tail = wavefronts_per_block * n_loop_unroll_tail;
    const int32_t tt_tail_low = wavefront_idx * n_loop_unroll_tail +
        n_unrolled_loops * dtt * (split_idx + 1);
    const int32_t tt_tail_high =
        (split_idx == arg.split_k - 1) ? t_max : tt_tail_low;

    for (auto tt = tt_low; tt < tt_high; tt += dtt) {
      if (lane_active_for_io) {
#pragma unroll n_loop_unroll
        for (auto ttt = 0; ttt < n_loop_unroll; ++ttt) {
          const int32_t t = tt + ttt;
          // load the K[b][t][g][h|0][:] row into registers
          load_v<data_t, data_vec_t>(
              cache_K_base + t * arg.K_stride_m, lane_idx, &k_loads[ttt]);
        }
      }
#pragma unroll n_loop_unroll
      for (auto ttt = 0; ttt < n_loop_unroll; ++ttt) {
        compute_t qk_acc = 0;
        ck_tile::inner_product<data_vec_t, data_vec_t, compute_t>(
            q_thread, k_loads[ttt], qk_acc);
        qk_acc *= arg.qk_scale;

        qk_acc = wavefrontReduce(qk_acc, [](auto a, auto b) { return a + b; });
        max_qk_acc = ck_tile::max(qk_acc, max_qk_acc);
        if (lane_idx == 0) {
          smem[tt + ttt - n_unrolled_loops * dtt * split_idx] = qk_acc;
        }
      }
    }

    for (auto tt = tt_tail_low; tt < tt_tail_high; tt += dtt_tail) {
      if (lane_active_for_io) {
#pragma unroll n_loop_unroll_tail
        for (auto ttt = 0; ttt < n_loop_unroll_tail; ++ttt) {
          const int32_t t = tt + ttt;
          if (t < t_max) {
            // load the K[b][t][g][h|0][:] row into registers
            load_v<data_t, data_vec_t>(
                cache_K_base + t * arg.K_stride_m, lane_idx, &k_loads[ttt]);
          }
        }
      }
#pragma unroll n_loop_unroll_tail
      for (auto ttt = 0; ttt < n_loop_unroll_tail; ++ttt) {
        compute_t qk_acc = 0;
        const int32_t t = tt + ttt;
        if (t < t_max) {
          ck_tile::inner_product<data_vec_t, data_vec_t, compute_t>(
              q_thread, k_loads[ttt], qk_acc);
          qk_acc *= arg.qk_scale;

          qk_acc =
              wavefrontReduce(qk_acc, [](auto a, auto b) { return a + b; });
          max_qk_acc = ck_tile::max(qk_acc, max_qk_acc);

          // write accumulated sums to smem.
          if (lane_idx == 0) {
            smem[t - n_unrolled_loops * dtt * split_idx] = qk_acc;
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
      max_qk_acc = ck_tile::max(max_qk_acc, smem[KV_M_MAX + lane_idx]);
    }
    // shared across all threads in block
    max_qk_acc = wavefrontReduce(
        max_qk_acc, [](auto a, auto b) { return a > b ? a : b; });

    if (wavefront_idx == 0 && lane_idx == 0) {
      arg.split_max[blockIdx.x * arg.split_k + split_idx] = max_qk_acc;
    }

    // each wavefront computes partial sum of exp.
    { // softmax reduce begin
      compute_t softmax_denominator = 0.0f;
      const int32_t t_low = n_unrolled_loops * dtt * split_idx;
      const int32_t t_high = (split_idx + 1 < arg.split_k)
          ? n_unrolled_loops * dtt * (split_idx + 1)
          : t_max;
      for (int32_t t = t_low + thread_linear_idx; t < t_high;
           t += threads_per_block) {
        const auto s = ck_tile::exp(smem[t - t_low] - max_qk_acc);
        softmax_denominator += s;
        smem[t - t_low] = s;
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

      if (wavefront_idx == 0 && lane_idx == 0) {
        arg.split_sumexp[blockIdx.x * arg.split_k + split_idx] =
            softmax_denominator;
      }
    } // softmax reduce end

    // Split T across wavefronts in a block
    // each wavefront compute sum(t_subset) P[t] * V[t_subset, d]
    // outputs are of size float[D]

    compute_t ps[n_loop_unroll] = {};
    compute_vec_t o_acc = 0;
    if (lane_active_for_io) {
      for (auto tt = tt_low; tt < tt_high; tt += dtt) {
#pragma unroll n_loop_unroll
        for (auto ttt = 0; ttt < n_loop_unroll; ++ttt) {
          const int32_t t = tt + ttt;
          // load the V[b][t][g][h|0][:] row into registers, reusing K register
          // storage
          load_v<data_t, data_vec_t>(
              cache_V_base + t * arg.K_stride_m, lane_idx, &k_loads[ttt]);
          ps[ttt] = smem[t - n_unrolled_loops * dtt * split_idx];
        }

#pragma unroll n_loop_unroll
        for (auto ttt = 0; ttt < n_loop_unroll; ++ttt) {
          o_acc =
              scalar_scale_acc<data_t, vec_size>(o_acc, k_loads[ttt], ps[ttt]);
        }
      }

      for (auto tt = tt_tail_low; tt < tt_tail_high; tt += dtt_tail) {
#pragma unroll n_loop_unroll_tail
        for (auto ttt = 0; ttt < n_loop_unroll_tail; ++ttt) {
          const int32_t t = tt + ttt;
          if (t < t_max) {
            // load the V[b][t][g][h|0][:] row into registers, reusing K
            // register storage
            load_v<data_t, data_vec_t>(
                cache_V_base + t * arg.K_stride_m, lane_idx, &k_loads[ttt]);
            ps[ttt] = smem[t - n_unrolled_loops * dtt * split_idx];
            o_acc = scalar_scale_acc<data_t, vec_size>(
                o_acc, k_loads[ttt], ps[ttt]);
          }
        }
      }
    }
    __syncthreads();

    // NB: needs sizeof(smem) >= `vec_size` * (sizeof(float)==4) *
    // threadsPerBlock
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
        bf_r.arr[i] = ck_tile::type_convert<data_t>(r.arr[i]);
      }
      // write output row O[b][m][g][h][:]
      data_t* __restrict__ o_ =
          arg.split_O + XQO_base_offset + split_idx * arg.O_stride_split;
      store_v<data_t, data_vec_t>(o_, lane_idx, bf_r.vec);
    }
  }
};

} // namespace ck_tile
