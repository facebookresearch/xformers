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

template <typename scalar_t, int32_t vec_size = 4, typename compute_t = float>
__global__ void efficient_attention_forward_decoder_splitk_reduce_ck_kernel(
    const scalar_t* __restrict__ O_splits,
    const compute_t* __restrict__ split_max,
    const compute_t* __restrict__ split_sumexp,
    scalar_t* __restrict__ O,
    const int32_t Q_size_m,
    const int32_t Q_size_g,
    const int32_t Q_size_h,
    const int32_t Q_size_k,
    const ptrdiff_t O_stride_split,
    const ptrdiff_t O_stride_b,
    const ptrdiff_t O_stride_m,
    const ptrdiff_t O_stride_g,
    const ptrdiff_t O_stride_h,
    const int32_t split_k) {
  // Each block handles a single batch and head and query and group
  const int32_t b = blockIdx.x / (Q_size_m * Q_size_g * Q_size_h);
  const int32_t m = (blockIdx.x / (Q_size_g * Q_size_h)) % Q_size_m;
  const int32_t g = (blockIdx.x / Q_size_h) % Q_size_g;
  const int32_t h = blockIdx.x % Q_size_h;

  using data_t = scalar_t;
  using data_vec_t = typename ck::vector_type<data_t, vec_size>::type;
  using compute_vec_t = typename ck::vector_type<compute_t, vec_size>::type;

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
  const bool lane_active_for_io = lane_idx * vec_size < Q_size_k;

  if (!lane_active_for_io) {
    return;
  }

  compute_t global_sumexp = 0;
  compute_t global_max = ck::NumericLimits<compute_t>::Lowest();

  for (int32_t split_idx = 0; split_idx < split_k; ++split_idx) {
    load_v<data_t, data_vec_t>(
        O_splits + b * O_stride_b + m * O_stride_m + g * O_stride_g +
            h * O_stride_h + split_idx * O_stride_split,
        lane_idx,
        &O_split_data.vec);
#pragma unroll
    for (int32_t i = 0; i < vec_size; ++i) {
      O_split_compute.arr[i] = ck::type_convert<compute_t>(O_split_data.arr[i]);
    }
    compute_t local_max = *(split_max + blockIdx.x * split_k + split_idx);
    compute_t local_sumexp = *(split_sumexp + blockIdx.x * split_k + split_idx);

    compute_t log_alpha = -std::abs(local_max - global_max);
    compute_t alpha =
        isnan(log_alpha) ? compute_t{1.} : ck::math::exp(log_alpha);

    bool pick_new = local_max < global_max;
    compute_t pick_current_coef = pick_new ? 1. : alpha;
    compute_t pick_new_coef = pick_new ? alpha : 1.;

    global_sumexp =
        pick_current_coef * global_sumexp + pick_new_coef * local_sumexp;
    global_O_compute.vec = pick_current_coef * global_O_compute.vec +
        pick_new_coef * O_split_compute.vec;
    global_max = ck::math::max(local_max, global_max);
  }
  global_O_compute.vec /= global_sumexp;
#pragma unroll
  for (int32_t i = 0; i < vec_size; ++i) {
    global_O_data.arr[i] = ck::type_convert<data_t>(global_O_compute.arr[i]);
  }
  store_v<data_t, data_vec_t>(
      O + b * O_stride_b + m * O_stride_m + g * O_stride_g + h * O_stride_h,
      lane_idx,
      global_O_data.vec);
}

template <
    typename scalar_t,
    int32_t vec_size,
    int32_t n_loop_unroll,
    int32_t n_loop_unroll_tail,
    int32_t KV_M_MAX,
    typename compute_t>
__global__ void efficient_attention_forward_decoder_splitk_ck_kernel(
    const scalar_t* __restrict__ XQ,
    const scalar_t* __restrict__ cache_K,
    const scalar_t* __restrict__ cache_V,
    scalar_t* __restrict__ O_splits,
    compute_t* __restrict__ split_max,
    compute_t* __restrict__ split_sumexp,
    const int32_t* __restrict__ seq_kv_lens,
    const ptrdiff_t XQ_stride_b,
    const ptrdiff_t XQ_stride_m,
    const ptrdiff_t XQ_stride_g,
    const ptrdiff_t XQ_stride_h,
    const ptrdiff_t K_stride_b,
    const ptrdiff_t K_stride_m,
    const ptrdiff_t K_stride_g,
    const ptrdiff_t K_stride_h,
    const ptrdiff_t O_stride_split,
    const int32_t Q_size_m,
    const int32_t Q_size_g,
    const int32_t Q_size_h,
    const int32_t Q_size_k,
    const int32_t K_size_m,
    const bool multiquery,
    const float qk_scale,
    const int32_t split_k) {
  static_assert(
      n_loop_unroll_tail < n_loop_unroll || n_loop_unroll_tail == 1,
      "tail unroll must be smaller than main loop untoll; pragma unroll 0 is illegal "
      "(and tail is no-op)");

  // Each block handles a single batch and head and query and group
  const int32_t b = blockIdx.x / (Q_size_m * Q_size_g * Q_size_h);
  const int32_t m = (blockIdx.x / (Q_size_g * Q_size_h)) % Q_size_m;
  const int32_t g = (blockIdx.x / Q_size_h) % Q_size_g;
  const int32_t h = blockIdx.x % Q_size_h;
  const int32_t split_idx = blockIdx.y;

  // Note: this is decoding case where we attend to current and all previous
  // tokens.
  const int32_t t_max = seq_kv_lens ? seq_kv_lens[b] : K_size_m;

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
  const auto XQO_base_offset =
      b * XQ_stride_b + m * XQ_stride_m + g * XQ_stride_g + h * XQ_stride_h;
  const auto* __restrict__ q_ = XQ + XQO_base_offset;

  const auto cache_KV_base_offset = b * K_stride_b + 0 * K_stride_m +
      g * K_stride_g + (multiquery ? 0 : h * K_stride_h);
  const auto* __restrict__ cache_K_base = cache_K + cache_KV_base_offset;
  const auto* __restrict__ cache_V_base = cache_V + cache_KV_base_offset;

  using data_t = scalar_t;
  using data_vec_t = typename ck::vector_type<data_t, vec_size>::type;
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

  const auto dtt = wavefronts_per_block * n_loop_unroll;
  // only last split gets the tail.
  // the first (split_k - 1) splits have a number of iterations divisible by
  // `dtt`
  const auto n_unrolled_loops = t_max / dtt / split_k; // +1?
  const int32_t tt_low =
      wavefront_idx * n_loop_unroll + n_unrolled_loops * dtt * split_idx;
  const int32_t tt_high =
      wavefront_idx * n_loop_unroll + n_unrolled_loops * dtt * (split_idx + 1);
  const int32_t dtt_tail = wavefronts_per_block * n_loop_unroll_tail;
  const int32_t tt_tail_low = wavefront_idx * n_loop_unroll_tail +
      n_unrolled_loops * dtt * (split_idx + 1);
  const int32_t tt_tail_high = (split_idx == split_k - 1) ? t_max : tt_tail_low;

  for (auto tt = tt_low; tt < tt_high; tt += dtt) {
    if (lane_active_for_io) {
#pragma unroll n_loop_unroll
      for (auto ttt = 0; ttt < n_loop_unroll; ++ttt) {
        const int32_t t = tt + ttt;
        // load the K[b][t][g][h|0][:] row into registers
        load_v<data_t, data_vec_t>(
            cache_K_base + t * K_stride_m, lane_idx, &k_loads[ttt]);
      }
    }
#pragma unroll n_loop_unroll
    for (auto ttt = 0; ttt < n_loop_unroll; ++ttt) {
      compute_t qk_acc = 0;
      ck::inner_product<data_vec_t, data_vec_t, compute_t>(
          q_thread, k_loads[ttt], qk_acc);
      qk_acc *= qk_scale;

      qk_acc = wavefrontReduce(qk_acc, [](auto a, auto b) { return a + b; });
      max_qk_acc = ck::math::max(qk_acc, max_qk_acc);
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
    max_qk_acc = ck::math::max(max_qk_acc, smem[KV_M_MAX + lane_idx]);
  }
  // shared across all threads in block
  max_qk_acc =
      wavefrontReduce(max_qk_acc, [](auto a, auto b) { return a > b ? a : b; });

  if (wavefront_idx == 0 && lane_idx == 0) {
    split_max[blockIdx.x * split_k + split_idx] = max_qk_acc;
  }

  // each wavefront computes partial sum of exp.
  { // softmax reduce begin
    compute_t softmax_denominator = 0.0f;
    const int32_t t_low = n_unrolled_loops * dtt * split_idx;
    const int32_t t_high = (split_idx + 1 < split_k)
        ? n_unrolled_loops * dtt * (split_idx + 1)
        : t_max;
    for (int32_t t = t_low + thread_linear_idx; t < t_high;
         t += threads_per_block) {
      const auto s = ck::math::exp(smem[t - t_low] - max_qk_acc);
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
      split_sumexp[blockIdx.x * split_k + split_idx] = softmax_denominator;
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
            cache_V_base + t * K_stride_m, lane_idx, &k_loads[ttt]);
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
          // load the V[b][t][g][h|0][:] row into registers, reusing K register
          // storage
          load_v<data_t, data_vec_t>(
              cache_V_base + t * K_stride_m, lane_idx, &k_loads[ttt]);
          ps[ttt] = smem[t - n_unrolled_loops * dtt * split_idx];
          o_acc =
              scalar_scale_acc<data_t, vec_size>(o_acc, k_loads[ttt], ps[ttt]);
        }
      }
    }
  }
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
    data_t* __restrict__ o_ =
        O_splits + XQO_base_offset + split_idx * O_stride_split;
    store_v<data_t, data_vec_t>(o_, lane_idx, bf_r.vec);
  }
}

} // namespace

namespace ck {
namespace tensor_operation {
namespace device {
template <
    typename scalar_t,
    int32_t KV_M_MAX,
    int32_t n_loop_unroll,
    int32_t n_loop_unroll_tail,
    typename compute_t>
struct FMHADecoderSplitKDeviceOp : public BaseOperator {
  using DeviceOp = FMHADecoderSplitKDeviceOp;
  struct Argument : public BaseArgument {
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

    const dim3 grid_dim;
    const dim3 block_dim;
    const size_t lds_bytes;

    Argument(
        const scalar_t* __restrict__ XQ,
        const scalar_t* __restrict__ cache_K,
        const scalar_t* __restrict__ cache_V,
        scalar_t* __restrict__ O,
        scalar_t* __restrict__ split_O,
        compute_t* __restrict__ split_max,
        compute_t* __restrict__ split_sumexp,
        const int32_t* __restrict__ seq_kv_lens,
        const ptrdiff_t XQ_stride_b,
        const ptrdiff_t XQ_stride_m,
        const ptrdiff_t XQ_stride_g,
        const ptrdiff_t XQ_stride_h,
        const ptrdiff_t K_stride_b,
        const ptrdiff_t K_stride_m,
        const ptrdiff_t K_stride_g,
        const ptrdiff_t K_stride_h,
        const ptrdiff_t O_stride_split,
        const int32_t Q_size_m,
        const int32_t Q_size_g,
        const int32_t Q_size_h,
        const int32_t Q_size_k,
        const int32_t K_size_m,
        const bool multiquery,
        const float qk_scale,
        const int32_t split_k,
        // launch params
        const dim3 grid_dim,
        const dim3 block_dim,
        const size_t lds_bytes)
        : XQ(XQ),
          cache_K(cache_K),
          cache_V(cache_V),
          O(O),
          split_O(split_O),
          split_max(split_max),
          split_sumexp(split_sumexp),
          seq_kv_lens(seq_kv_lens),
          XQ_stride_b(XQ_stride_b),
          XQ_stride_m(XQ_stride_m),
          XQ_stride_g(XQ_stride_g),
          XQ_stride_h(XQ_stride_h),
          K_stride_b(K_stride_b),
          K_stride_m(K_stride_m),
          K_stride_g(K_stride_g),
          K_stride_h(K_stride_h),
          O_stride_split(O_stride_split),
          Q_size_m(Q_size_m),
          Q_size_g(Q_size_g),
          Q_size_h(Q_size_h),
          Q_size_k(Q_size_k),
          K_size_m(K_size_m),
          multiquery(multiquery),
          qk_scale(qk_scale),
          split_k(split_k),
          // launch params
          grid_dim(grid_dim),
          block_dim(block_dim),
          lds_bytes(lds_bytes) {}

    std::string str() const {
      std::ostringstream oss;
      oss << "Argument { " << std::endl
          << "    XQ: " << XQ << std::endl
          << "    cache_K: " << cache_K << std::endl
          << "    cache_V: " << cache_V << std::endl
          << "    O: " << O << std::endl
          << "    split_O: " << split_O << std::endl
          << "    split_max: " << split_max << std::endl
          << "    split_sumexp: " << split_sumexp << std::endl
          << "    seq_kv_lens: " << seq_kv_lens << std::endl
          << "    XQ_stride_b: " << XQ_stride_b << std::endl
          << "    XQ_stride_m: " << XQ_stride_m << std::endl
          << "    XQ_stride_g: " << XQ_stride_g << std::endl
          << "    XQ_stride_h: " << XQ_stride_h << std::endl
          << "    K_stride_b: " << K_stride_b << std::endl
          << "    K_stride_m: " << K_stride_m << std::endl
          << "    K_stride_g: " << K_stride_g << std::endl
          << "    K_stride_h: " << K_stride_h << std::endl
          << "    O_stride_split: " << O_stride_split << std::endl
          << "    Q_size_m: " << Q_size_m << std::endl
          << "    Q_size_g: " << Q_size_g << std::endl
          << "    Q_size_h: " << Q_size_h << std::endl
          << "    Q_size_k: " << Q_size_k << std::endl
          << "    K_size_m: " << K_size_m << std::endl
          << "    multiquery: " << multiquery << std::endl
          << "    qk_scale: " << qk_scale << std::endl
          << "    split_k: " << split_k << std::endl
          << std::endl
          << "    grid_dim: " << grid_dim.x << "." << grid_dim.y << "."
          << grid_dim.z << std::endl
          << "    block_dim: " << block_dim.x << "." << block_dim.y << "."
          << block_dim.z << std::endl
          << "    lds_bytes: " << lds_bytes << std::endl
          << "}";
      return oss.str();
    }
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

      float split_attention_result = launch_and_time_kernel(
          stream_config,
          Q_size_k_alignment_necessary == 4
              ? efficient_attention_forward_decoder_splitk_ck_kernel<
                    scalar_t,
                    /* vec_size */ 4,
                    n_loop_unroll,
                    n_loop_unroll_tail,
                    KV_M_MAX,
                    compute_t>
              : Q_size_k_alignment_necessary == 2
              ? efficient_attention_forward_decoder_splitk_ck_kernel<
                    scalar_t,
                    /* vec_size */ 2,
                    n_loop_unroll,
                    n_loop_unroll_tail,
                    KV_M_MAX,
                    compute_t>
              : Q_size_k_alignment_necessary == 1
              ? efficient_attention_forward_decoder_splitk_ck_kernel<
                    scalar_t,
                    /* vec_size */ 1,
                    n_loop_unroll,
                    n_loop_unroll_tail,
                    KV_M_MAX,
                    compute_t>
              : nullptr,
          argp->grid_dim,
          argp->block_dim,
          argp->lds_bytes,
          argp->XQ,
          argp->cache_K,
          argp->cache_V,
          argp->split_O,
          argp->split_max,
          argp->split_sumexp,
          argp->seq_kv_lens,
          argp->XQ_stride_b,
          argp->XQ_stride_m,
          argp->XQ_stride_g,
          argp->XQ_stride_h,
          argp->K_stride_b,
          argp->K_stride_m,
          argp->K_stride_g,
          argp->K_stride_h,
          argp->O_stride_split,
          argp->Q_size_m,
          argp->Q_size_g,
          argp->Q_size_h,
          argp->Q_size_k,
          argp->K_size_m,
          argp->multiquery,
          argp->qk_scale,
          argp->split_k);

      const dim3 reduce_gridsize = {argp->grid_dim.x};
      const dim3 reduce_blocksize = {argp->block_dim.x};
      constexpr int32_t reduce_lds_bytes = 0;
      float reduce_result = launch_and_time_kernel(
          stream_config,
          Q_size_k_alignment_necessary == 4
              ? efficient_attention_forward_decoder_splitk_reduce_ck_kernel<
                    scalar_t,
                    4>
              : Q_size_k_alignment_necessary == 2
              ? efficient_attention_forward_decoder_splitk_reduce_ck_kernel<
                    scalar_t,
                    2>
              : Q_size_k_alignment_necessary == 1
              ? efficient_attention_forward_decoder_splitk_reduce_ck_kernel<
                    scalar_t,
                    1>
              : nullptr,
          reduce_gridsize,
          reduce_blocksize,
          reduce_lds_bytes,
          argp->split_O,
          argp->split_max,
          argp->split_sumexp,
          argp->O,
          argp->Q_size_m,
          argp->Q_size_g,
          argp->Q_size_h,
          argp->Q_size_k,
          argp->O_stride_split,
          argp->XQ_stride_b,
          argp->XQ_stride_m,
          argp->XQ_stride_g,
          argp->XQ_stride_h,
          argp->split_k);
      return split_attention_result + reduce_result;
    }
  };
};
} // namespace device
} // namespace tensor_operation
} // namespace ck
