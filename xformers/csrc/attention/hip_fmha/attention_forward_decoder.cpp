/*
  TODO: license header
*/

// #include <ck/ck.hpp>
#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <ck/host_utility/kernel_launch_hip.hpp>
#include <ck/stream_config.hpp>
#include <ck/tensor_operation/gpu/device/device_base.hpp>
#include <ck/utility/data_type.hpp>
#include <ck/utility/inner_product.hpp>
#include <torch/library.h>

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

constexpr int32_t kThreadsPerWavefront = 64;
constexpr int32_t kWavefrontsPerBlock = 16;
constexpr int32_t D_H = 4 * kThreadsPerWavefront;
constexpr int32_t T_MAX = 8192;

template <typename c10_t>
struct c10_to_data_t;

template <>
struct c10_to_data_t<float> {
  using type = float;
  using vec4 = ck::float4_t;
};

template <>
struct c10_to_data_t<c10::Half> {
  using type = ck::half_t;
  using vec4 = ck::half4_t;
};

template <>
struct c10_to_data_t<c10::BFloat16> {
  using type = ck::bhalf_t;
  using vec4 = ck::bhalf4_t;
};

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

template <typename F>
float __device__ __forceinline__ wavefrontReduce(float val, F f) {
#pragma unroll
  for (int32_t mask = kThreadsPerWavefront >> 1; mask > 0; mask >>= 1) {
    val = f(__shfl_xor(val, mask, kThreadsPerWavefront), val);
  }
  return val;
}

template <typename TDataPtr, typename TDataVec>
__forceinline__ __device__ void load_v(
    TDataPtr data_ptr,
    int32_t vector_offset,
    TDataVec* load_to) {
  *load_to = *(reinterpret_cast<const TDataVec*>(data_ptr) + vector_offset);
}

template <typename TDataPtr, typename TDataVec>
__forceinline__ __device__ void store_v(
    TDataPtr data_ptr,
    int32_t vector_offset,
    TDataVec value) {
  *(reinterpret_cast<TDataVec*>(data_ptr) + vector_offset) = value;
}

template <
    typename scalar_t,
    int32_t n_loop_unroll = 16,
    int32_t n_loop_unroll_tail = 2>
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
    const bool multiquery,
    const float qk_scale) {
  static_assert(n_loop_unroll_tail < n_loop_unroll, "");

  constexpr int32_t seq_positions_shift = 0;

  extern __shared__ __align__(16) float smem[];

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

  // Need D_H == 256 (NB: 128 in CUDA because of wavefront/warp sizes 64/32)
  // const auto* q_ = &(XQ_acc[b][0][h][0]);
  const auto XQO_base_offset = b * XQ_stride_0 + h * XQ_stride_2;
  const auto* q_ = XQ + XQO_base_offset;

  const auto cache_KV_base_offset =
      b * K_stride_0 + (multiquery ? 0 : h * K_stride_2);
  const auto* cache_K_base = cache_K + cache_KV_base_offset;
  const auto* cache_V_base = cache_V + cache_KV_base_offset;

  // Load Q into registers in all wavefronts.
  // Each thread handles 4 D dimensions
  using data_t = typename c10_to_data_t<scalar_t>::type;
  using data_vec4_t = typename c10_to_data_t<scalar_t>::vec4;
  data_vec4_t q_thread;
  load_v<decltype(q_), data_vec4_t>(q_, lane_idx, &q_thread);
  // Each block computes different B value
  float max_qk_acc = std::numeric_limits<float>::lowest();

  // Compute S[T_MAX] = for i in range(T): S[t] = sum(Q[d] * K[t, d])
  // Split T across wavefronts in a block, unroll loads to expose more
  // parallelism.

  data_vec4_t k_loads[n_loop_unroll];

  constexpr auto dtt = kWavefrontsPerBlock * n_loop_unroll;
  const int32_t t_max_unroll = (t_max / dtt) * dtt;

  for (auto tt = wavefront_idx * n_loop_unroll; tt < t_max_unroll; tt += dtt) {
#pragma unroll n_loop_unroll
    for (auto ttt = 0; ttt < n_loop_unroll; ++ttt) {
      const int32_t t = tt + ttt;
      // load the K[b][t][h|0][:] row into registers
      load_v<decltype(cache_K_base), data_vec4_t>(
          cache_K_base + t * K_stride_1, lane_idx, &k_loads[ttt]);
    }
    float qk_accs[n_loop_unroll] = {};
#pragma unroll n_loop_unroll
    for (auto ttt = 0; ttt < n_loop_unroll; ++ttt) {
      ck::inner_product<data_vec4_t, data_vec4_t, float>(
          q_thread, k_loads[ttt], qk_accs[ttt]);
      qk_accs[ttt] *= qk_scale;

      qk_accs[ttt] =
          wavefrontReduce(qk_accs[ttt], [](float a, float b) { return a + b; });
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
        load_v<decltype(cache_K_base), data_vec4_t>(
            cache_K_base + t * K_stride_1, lane_idx, &k_loads[ttt]);
      }
    }
#pragma unroll n_loop_unroll_tail
    for (auto ttt = 0; ttt < n_loop_unroll_tail; ++ttt) {
      float qk_acc = 0;
      const int32_t t = tt + ttt;
      if (t < t_max) {
        ck::inner_product<data_vec4_t, data_vec4_t, float>(
            q_thread, k_loads[ttt], qk_acc);
        qk_acc *= qk_scale;

        qk_acc =
            wavefrontReduce(qk_acc, [](float a, float b) { return a + b; });
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
      max_qk_acc, [](float a, float b) { return a > b ? a : b; });

  // each wavefront computes partial sum of exp.
  float softmax_denominator = 0.0f;
  for (int32_t t = thread_linear_idx; t < t_max; t += threads_per_block) {
    softmax_denominator += __expf(smem[t] - max_qk_acc);
  }
  softmax_denominator = wavefrontReduce(
      softmax_denominator, [](float a, float b) { return a + b; });

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
      softmax_denominator, [](float a, float b) { return a + b; });

  const float softmax_scale_factor = 1. / softmax_denominator;
  // now, compute the normalization across all threads.
  for (int32_t t = thread_linear_idx; t < t_max; t += threads_per_block) {
    smem[t] = __expf(smem[t] - max_qk_acc) * softmax_scale_factor;
  }
  __syncthreads();

  // Now, we can compute the softmax and write the outputs.

  // Split T across wavefronts in a block
  // each wavefront compute sum(t_subset) P[t] * V[t_subset, d]
  // outputs are of size float[D]

  float ps[n_loop_unroll];
  ck::float4_t o_acc = 0;
  for (auto tt = wavefront_idx * n_loop_unroll; tt < t_max_unroll; tt += dtt) {
#pragma unroll n_loop_unroll
    for (auto ttt = 0; ttt < n_loop_unroll; ++ttt) {
      const int32_t t = tt + ttt;
      // load the V[b][t][h|0][:] row into registers, reusing K register storage
      load_v<decltype(cache_V_base), data_vec4_t>(
          cache_V_base + t * K_stride_1, lane_idx, &k_loads[ttt]);
      ps[ttt] = smem[t];
    }

#pragma unroll n_loop_unroll
    for (auto ttt = 0; ttt < n_loop_unroll; ++ttt) {
      o_acc = scalar4_scale_acc<data_vec4_t>(o_acc, k_loads[ttt], ps[ttt]);
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
        load_v<decltype(cache_V_base), data_vec4_t>(
            cache_V_base + t * K_stride_1, lane_idx, &k_loads[ttt]);
        ps[ttt] = smem[t];
      }
    }

#pragma unroll n_loop_unroll_tail
    for (auto ttt = 0; ttt < n_loop_unroll_tail; ++ttt) {
      const int32_t t = tt + ttt;
      if (t < t_max) {
        o_acc = scalar4_scale_acc<data_vec4_t>(o_acc, k_loads[ttt], ps[ttt]);
      }
    }
  }
  // now, each thread has partial sums. Write to smem and get accumulated
  // results back.
  __syncthreads();

  // NB: needs sizeof(smem) >= 4 * (sizeof(float)==4) * threadsPerBlock
  store_v<float*, ck::float4_t>(&smem[0], thread_linear_idx, o_acc);

  __syncthreads();
  // sum up partial D rows from other wavefronts
  if (wavefront_idx == 0) {
    ck::float4_t r = 0;
    for (int32_t w = 0; w < wavefronts_per_block; ++w) {
      ck::float4_t partial_r;
      load_v<float*, ck::float4_t>(
          smem, w * threads_per_wavefront + lane_idx, &partial_r);
      r += partial_r;
    }
    // write output D row
    data_vec4_t bf_r;
    bf_r.x = ck::type_convert<data_t>(r.x);
    bf_r.y = ck::type_convert<data_t>(r.y);
    bf_r.z = ck::type_convert<data_t>(r.z);
    bf_r.w = ck::type_convert<data_t>(r.w);
    auto* o_ = O + XQO_base_offset;
    store_v<decltype(o_), data_vec4_t>(o_, lane_idx, bf_r);
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
          arg.multiquery,
          arg.qk_scale);
    }
  };
};
} // namespace device
} // namespace tensor_operation
} // namespace ck

namespace {

#define AT_DISPATCH_CASE_3(SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, ...) \
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)                           \
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)                           \
  AT_DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)

#define AT_DISPATCH_SWITCH_3(                               \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                       \
      TYPE,                                                 \
      NAME,                                                 \
      AT_DISPATCH_CASE_3(SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, __VA_ARGS__))

template <int32_t ThreadsPerWavefront, int32_t WavefrontsPerBlock>
at::Tensor& efficient_attention_forward_decoder_ck_out_impl(
    const at::Tensor& XQ, // [B, 1, H, D]
    const at::Tensor& cache_K, // [B, T_MAX, H or 1, D]
    const at::Tensor& cache_V, // [B, T_MAX, H or 1, D]
    const at::Tensor& seq_positions, // [B]
    double qk_scale,
    at::Tensor& O) {
  static_assert(4 * ThreadsPerWavefront == D_H, "");
  static_assert(WavefrontsPerBlock <= ThreadsPerWavefront, "");

  at::OptionalDeviceGuard guard(XQ.device());
  TORCH_CHECK(XQ.is_cuda());
  TORCH_CHECK(cache_K.is_cuda());
  TORCH_CHECK(cache_V.is_cuda());

  TORCH_CHECK(seq_positions.is_cuda());

  TORCH_CHECK(cache_K.size(1) <= T_MAX);
  TORCH_CHECK(cache_K.size(3) == D_H);

  auto B = XQ.size(0);
  auto H = XQ.size(2);
  dim3 blocks(B, H);
  dim3 threads(ThreadsPerWavefront, WavefrontsPerBlock);

  int32_t smem_softmax = T_MAX * sizeof(float) + threads.y * sizeof(float);
  int32_t smem_output = D_H * sizeof(float) *
      threads.y; // 4 * threadsPerBlock * sizeof(float) == sizeof(O[b][0][h][:])
  const size_t lds_bytes = max(smem_softmax, smem_output);
  auto stream = at::cuda::getCurrentHIPStream().stream();

  AT_DISPATCH_SWITCH_3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Float,
      XQ.scalar_type(),
      "efficient_attention_forward_decoder_ck",
      [&] {
        using device_op_t =
            ck::tensor_operation::device::FMHADecoderSeqlen1DeviceOp<scalar_t>;
        auto op = device_op_t{};

        auto XQ_acc =
            XQ.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>();
        auto K_acc =
            cache_K.packed_accessor64<scalar_t, 4, at::RestrictPtrTraits>();
        auto V_acc =
            cache_V.packed_accessor64<scalar_t, 4, at::RestrictPtrTraits>();
        auto O_acc = O.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>();
        auto seq_acc =
            seq_positions
                .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>();
        auto arg = device_op_t::Argument(
            XQ_acc.data(),
            K_acc.data(),
            V_acc.data(),
            O_acc.data(),
            seq_acc.data(),
            XQ_acc.stride(0),
            XQ_acc.stride(2),
            K_acc.stride(0),
            K_acc.stride(1),
            K_acc.stride(2),
            K_acc.size(2) == 1,
            qk_scale,
            blocks,
            threads,
            lds_bytes);

        auto invoker = device_op_t::Invoker{};
        (void)invoker.Run(arg, {stream});
      });

  return O;
}

#undef AT_DISPATCH_CASE_3
#undef AT_DISPATCH_SWITCH_3

template <int32_t ThreadsPerWavefront, int32_t WavefrontsPerBlock>
at::Tensor efficient_attention_forward_decoder_ck_impl(
    const at::Tensor& XQ, // [B, 1, H, D]
    const at::Tensor& cache_K, // [B, T_MAX, H or 1, D]
    const at::Tensor& cache_V, // [B, T_MAX, H or 1, D]
    const at::Tensor& seq_positions, // [B]
    double qk_scale) {
  auto O = at::empty_like(XQ);
  efficient_attention_forward_decoder_ck_out_impl<
      ThreadsPerWavefront,
      WavefrontsPerBlock>(XQ, cache_K, cache_V, seq_positions, qk_scale, O);
  return O;
}

at::Tensor efficient_attention_forward_decoder_ck(
    const at::Tensor& XQ, // [B, 1, H, D]
    const at::Tensor& cache_K, // [B, T_MAX, H or 1, D]
    const at::Tensor& cache_V, // [B, T_MAX, H or 1, D]
    const at::Tensor& seq_positions, // [B]
    double qk_scale) {
  return efficient_attention_forward_decoder_ck_impl<
      kThreadsPerWavefront,
      kWavefrontsPerBlock>(XQ, cache_K, cache_V, seq_positions, qk_scale);
}
} // namespace

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::efficient_attention_forward_decoder_ck"),
      TORCH_FN(efficient_attention_forward_decoder_ck));
}

#ifdef ATTN_FWD_DECODER_MAIN

#include <torch/torch.h>

// clang-format off

/*

(1) hipify
 > pip install -e /xformers

 For obtaining all the library paths needed for compilation below, add `--verbose`.
 For efficient utilization of CPU cores for compilation use MAX_JOBS env variable.

(2) compile
 > /opt/rocm/bin/hipcc \
-I/xformers/xformers/csrc \
-I/xformers/xformers/csrc/attention/hip_fmha \
-I/xformers/third_party/composable_kernel/include \
-I/xformers/third_party/composable_kernel/include/ck \
-I/xformers/third_party/composable_kernel/include/ck/tensor_operation/gpu/device \
-I/xformers/third_party/composable_kernel/include/ck/tensor_operation/gpu/device/impl \
-I/xformers/third_party/composable_kernel/include/ck/tensor_operation/gpu/element \
-I/opt/conda/envs/py_3.8/lib/python3.8/site-packages/torch/include \
-I/opt/conda/envs/py_3.8/lib/python3.8/site-packages/torch/include/torch/csrc/api/include \
-I/opt/conda/envs/py_3.8/lib/python3.8/site-packages/torch/include/TH \
-I/opt/conda/envs/py_3.8/lib/python3.8/site-packages/torch/include/THC \
-I/opt/conda/envs/py_3.8/lib/python3.8/site-packages/torch/include/THH \
-I/opt/rocm/include \
-I/opt/conda/envs/py_3.8/include/python3.8 \
-L/opt/conda/envs/py_3.8/lib/python3.8/site-packages/torch/lib \
-L/opt/conda/envs/py_3.8/lib \
-L/opt/rocm/lib \
-L/opt/rocm/hip/lib \
-fPIC \
-D__HIP_PLATFORM_HCC__=1 \
-DATTN_FWD_DECODER_MAIN \
-DUSE_ROCM=1 \
-DCUDA_HAS_FP16=1 \
-D__HIP_NO_HALF_OPERATORS__=1 \
-D__HIP_NO_HALF_CONVERSIONS__=1 \
-O3 \
-std=c++17 \
--offload-arch=gfx90a \
-U__CUDA_NO_HALF_OPERATORS__ \
-U__CUDA_NO_HALF_CONVERSIONS__ \
-DBUILD_PYTHON_PACKAGE \
-DTORCH_API_INCLUDE_EXTENSION_H \
'-DPYBIND11_COMPILER_TYPE="_gcc"' \
'-DPYBIND11_STDLIB="_libstdcpp"' \
'-DPYBIND11_BUILD_ABI="_cxxabi1013"' \
-DTORCH_EXTENSION_NAME=_C \
-D_GLIBCXX_USE_CXX11_ABI=1 \
-fno-gpu-rdc \
/xformers/xformers/csrc/attention/hip_fmha/attention_forward_decoder.hip \
-lc10_hip \
-ltorch_hip \
-lc10 \
-ltorch \
-ltorch_cpu \
-ltorch_python \
-lpython3.8 \
-lamdhip64 \
-o a.out

For assembly debugging, add `--save-temps -g`.

(3a) run correctness check
 >
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/envs/py_3.8/lib/python3.8/site-packages/torch/lib \
 ./a.out

(3b) run specific input shape
 >
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/envs/py_3.8/lib/python3.8/site-packages/torch/lib \
 ./a.out n_keys padding batch_size n_heads is_multiquery dtype n_wavefronts_per_block
*/

// clang-format on

static void do_correctness_check() {
  const int32_t D = 4 * kThreadsPerWavefront;
  const int32_t B = 1;
  const int32_t H = 4;
  auto options = torch::TensorOptions()
                     .dtype(torch::kFloat32)
                     .layout(torch::kStrided)
                     .device(torch::kCUDA, 1)
                     .requires_grad(false);
  auto int_options = options.dtype(torch::kInt);
  auto XQ = at::randn({B, 1, H, D}, options);
  auto K = at::randn({B, 4096, H, D}, options);
  auto V = at::randn({B, 4096, H, D}, options);
  auto seq = at::randint(63, 128, {B}, int_options);
  double qk_scale = 1. / sqrt(D);

  auto result = efficient_attention_forward_decoder_ck_impl<64, 1>(
      XQ, K, V, seq, qk_scale);
  auto gold_result = efficient_attention_forward_decoder_ck_impl<64, 2>(
      XQ, K, V, seq, qk_scale);
  auto mask = at::isclose(
      result, gold_result, /*atol*/ 1e-3, /*rtol*/ 1e-5, /*equal_nan*/ false);
  auto percent_match = at::sum(mask.to(torch::kFloat32)) / mask.numel();
  printf(
      "Mismatched elements percentage: %.2f\n",
      1. - percent_match.item<float>());
}

int main(int argc, char** argv) {
  if (argc == 1) {
    do_correctness_check();
  } else {
    const auto args = std::vector<std::string>(argv + 1, argv + argc);
    if (args.size() != 7) {
      std::cout
          << "Usage: ./a.out n_keys padding batch_size n_heads is_multiquery dtype n_wavefronts_per_block"
          << std::endl;
      return 0;
    }
    const int32_t n_keys = std::stoi(args[0]);
    const int32_t padding = std::stoi(args[1]);
    const int32_t batch_size = std::stoi(args[2]);
    const int32_t n_heads = std::stoi(args[3]);
    const int32_t multiquery = (args[4] == "mq");
    const auto dtype = (args[5] == "f32") ? torch::kFloat32
        : (args[5] == "f16")              ? torch::kFloat16
                                          : torch::kBFloat16;
    const int32_t n_wavefronts_per_block = std::stoi(args[6]);

    const int32_t dim_per_head = 4 * kThreadsPerWavefront;

    const auto options = torch::TensorOptions()
                             .dtype(dtype)
                             .layout(torch::kStrided)
                             .device(torch::kCUDA, 1)
                             .requires_grad(false);

    const auto int_options = options.dtype(torch::kInt);
    const auto Q = at::rand({batch_size, 1, n_heads, dim_per_head}, options);
    const auto K = multiquery
        ? at::rand({batch_size, padding, 1, dim_per_head}, options)
              .expand({batch_size, padding, n_heads, dim_per_head})
        : at::rand({batch_size, padding, n_heads, dim_per_head}, options);
    const auto V = at::rand_like(K);
    auto O = at::rand_like(Q);

    const auto seq = at::randint(1, n_keys, {batch_size}, int_options);
    const double qk_scale = 1. / sqrt(dim_per_head);
    auto call_ptr = decltype(&efficient_attention_forward_decoder_ck_out_impl<
                             kThreadsPerWavefront,
                             kWavefrontsPerBlock>){};

#define SWITCH_CASE_SET_CALLPTR(n)                               \
  case (n):                                                      \
    call_ptr = &efficient_attention_forward_decoder_ck_out_impl< \
        kThreadsPerWavefront,                                    \
        (n)>;                                                    \
    break;

    switch (n_wavefronts_per_block) {
      SWITCH_CASE_SET_CALLPTR(1);
      SWITCH_CASE_SET_CALLPTR(2);
      SWITCH_CASE_SET_CALLPTR(4);
      SWITCH_CASE_SET_CALLPTR(8);
      SWITCH_CASE_SET_CALLPTR(16);

      default:
        call_ptr = nullptr;
        break;
    }
#undef SWITCH_CASE_SET_CALLPTR

    if (call_ptr) {
      call_ptr(Q, K, V, seq, qk_scale, O);
    } else {
      std::cout << "Warning: no kernel was found for wavefronts_per_block="
                << n_wavefronts_per_block << std::endl;
    }
  }
  return 0;
}

#endif // MAIN