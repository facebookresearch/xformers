/* 
  TODO: license header
*/

// #include <ck/ck.hpp>
#include <ck/utility/data_type.hpp>
#include <ck/utility/inner_product.hpp>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>
#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/Tensor.h>

namespace ck {
template <>
__device__ void inner_product<bhalf_t, bhalf_t, float>(const bhalf_t& a, const bhalf_t& b, float& c)
{
    inner_product(type_convert<float>(a), type_convert<float>(b), c);
}

template <>
__device__ void inner_product<bhalf4_t, bhalf4_t, float>(const bhalf4_t& a, const bhalf4_t& b, float& c)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    inner_product(vector_type<bhalf_t, 4>{a}.AsType<bhalf_t>()[I0],
                  vector_type<bhalf_t, 4>{b}.AsType<bhalf_t>()[I0],
                  c);

    inner_product(vector_type<bhalf_t, 4>{a}.AsType<bhalf_t>()[I1],
                  vector_type<bhalf_t, 4>{b}.AsType<bhalf_t>()[I1],
                  c);

    inner_product(vector_type<bhalf_t, 4>{a}.AsType<bhalf_t>()[I2],
                  vector_type<bhalf_t, 4>{b}.AsType<bhalf_t>()[I2],
                  c);

    inner_product(vector_type<bhalf_t, 4>{a}.AsType<bhalf_t>()[I3],
                  vector_type<bhalf_t, 4>{b}.AsType<bhalf_t>()[I3],
                  c);
}
} // namespace ck

namespace {

constexpr int32_t kThreadsPerWavefront = 64;
constexpr int32_t kWavefrontsPerBlock = 16;
constexpr int32_t D_H = 256;
constexpr int32_t T_MAX = 8192;

// read 4 elements in one instruction
template<typename c10_t>
struct c10_to_read_t;

template<>
struct c10_to_read_t<float> {
  using type = uint4;
};

template<>
struct c10_to_read_t<c10::Half> {
  using type = uint2;
};

template<>
struct c10_to_read_t<c10::BFloat16> {
  using type = uint2;
};

template<typename c10_t>
struct c10_to_data_t;

template<>
struct c10_to_data_t<float> {
    using type = float_t;
    using vec4 = ck::float4_t;
};

template<>
struct c10_to_data_t<c10::Half> {
    using type = ck::half_t;
    using vec4 = ck::half4_t;
};

template<>
struct c10_to_data_t<c10::BFloat16> {
    using type = ck::bhalf_t;
    using vec4 = ck::bhalf4_t;
};

template<typename read_t, typename data_t>
__device__
float4 scalar4_scale_acc(float4 acc, const read_t* ra, float b);

template<>
__device__  
float4
scalar4_scale_acc<uint4, float>(float4 acc, const uint4* ra, float b) {
  const auto* a = reinterpret_cast<const float4*>(ra);
  acc.x += a->x * b;
  acc.y += a->y * b;
  acc.z += a->z * b;
  acc.w += a->w * b;
  return acc;
}

template<>
__device__  
float4
scalar4_scale_acc<uint2, ck::half_t>(float4 acc, const uint2* ra, float b) {
  const auto* a = reinterpret_cast<const ck::half4_t*>(ra);
  acc.x += a->x * b;
  acc.y += a->y * b;
  acc.z += a->z * b;
  acc.w += a->w * b;
  return acc;
}

template<>
__device__  
float4
scalar4_scale_acc<uint2, ck::bhalf_t>(float4 acc, const uint2* ra, float b) {
  const auto* a = reinterpret_cast<const ck::bhalf4_t*>(ra);
  acc.x += a->x * b;
  acc.y += a->y * b;
  acc.z += a->z * b;
  acc.w += a->w * b;
  return acc;
}

template <typename F>
float
__device__ wavefrontReduce(float val) {
  auto reducer = F();  
#pragma unroll
  for (uint mask = kThreadsPerWavefront >> 1; mask > 0; mask >>= 1) {
    val = reducer(val, __shfl_xor(val, mask, kThreadsPerWavefront));
  }
  return val;
}

template<typename scalar_t>
__global__ void
efficient_attention_forward_decoder_ck_kernel(
    at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> XQ,
    at::PackedTensorAccessor64<scalar_t, 4, at::RestrictPtrTraits> cache_K,
    at::PackedTensorAccessor64<scalar_t, 4, at::RestrictPtrTraits> cache_V,
    at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> O,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> seq_positions,
    float qk_scale
) {
  static_assert(4 * kThreadsPerWavefront == D_H, "");
  static_assert(kWavefrontsPerBlock <= kThreadsPerWavefront, "");

  constexpr int32_t seq_positions_shift = 0;

  extern __shared__ __align__(16) float smem[];

  // Each block handles a single batch and head
  int32_t b = blockIdx.x;
  int32_t h = blockIdx.y;

  // Note: this is decoding case where we attend to current and all previous
  // tokens.
  int32_t t_max = seq_positions[b] + seq_positions_shift;

  int32_t wavefront_idx = threadIdx.y;
  // need kWavefrontsPerBlock == blockDim.y;
  // Need D_H == 128
  const auto* q_ = &(XQ[b][0][h][0]);

  bool multiquery = cache_K.size(2) == 1;
  auto* cache_K_base = &cache_K[b][0][multiquery ? 0 : h][0];
  auto* cache_V_base = &cache_V[b][0][multiquery ? 0 : h][0];

  // Load Q into registers in all wavefronts.
  // Each thread handles 4 D dimensions
  using read_t = typename c10_to_read_t<scalar_t>::type;
  using data_t = typename c10_to_data_t<scalar_t>::type;
  using data_vec4_t = typename c10_to_data_t<scalar_t>::vec4;
  const read_t* q_thread = reinterpret_cast<const read_t*>(q_) + threadIdx.x;

  // Each block computes different B value
  float max_qk_acc = std::numeric_limits<float>::lowest();

  // Compute S[T_MAX] = for i in range(T): S[t] = sum(Q[d] * K[t, d])
  // Split T across wavefronts in a block, unroll loads to expose more
  // parallelism.

  constexpr int32_t kTimeUnroll = 1;
  const read_t* k_loads[kTimeUnroll];

  const int32_t t_max_unroll =
    (t_max / (kWavefrontsPerBlock * kTimeUnroll)) * (kWavefrontsPerBlock * kTimeUnroll);

  for (auto tt = wavefront_idx * kTimeUnroll; tt < t_max_unroll;
       tt += kWavefrontsPerBlock * kTimeUnroll) {
#pragma unroll kTimeUnroll
    for (auto ttt = 0; ttt < kTimeUnroll; ++ttt) {
      int32_t t = tt + ttt;
      auto* k_ = cache_K_base + t * cache_K.stride(1);
      // scalar4<scalar_t> k_thread;
      k_loads[ttt] =
          reinterpret_cast<const read_t*>(k_) + threadIdx.x;
    }
#pragma unroll kTimeUnroll
    for (auto ttt = 0; ttt < kTimeUnroll; ++ttt) {
      float qk_acc = 0;
      int32_t t = tt + ttt;

      ck::inner_product<data_vec4_t, data_vec4_t, float>(*reinterpret_cast<const data_vec4_t*>(q_thread), 
                                                         *reinterpret_cast<const data_vec4_t*>(k_loads[ttt]), 
                                                         qk_acc);
      qk_acc *= qk_scale;

      qk_acc = wavefrontReduce<std::plus<float>>(qk_acc);
      max_qk_acc = max(qk_acc, max_qk_acc);

      // write accumulated sums to smem.
      if (threadIdx.x == 0) {
        smem[t] = qk_acc;
      }
    }
  }

  constexpr int32_t kTimeUnroll1 = 1;
  for (auto tt = t_max_unroll + wavefront_idx; tt < t_max;
       tt += kWavefrontsPerBlock * kTimeUnroll1) {
#pragma unroll kTimeUnroll1
    for (auto ttt = 0; ttt < kTimeUnroll1; ++ttt) {
      int32_t t = tt + ttt;
      // &(cache_K[b][t][0][0]);
      auto* k_ = cache_K_base + t * cache_K.stride(1);
      // scalar4<scalar_t> k_thread;
      k_loads[ttt] =
          reinterpret_cast<const read_t*>(k_) + threadIdx.x;
    }
#pragma unroll kTimeUnroll1
    for (auto ttt = 0; ttt < kTimeUnroll1; ++ttt) {
      float qk_acc = 0;
      int32_t t = tt + ttt;
      ck::inner_product<data_vec4_t, data_vec4_t, float>(*reinterpret_cast<const data_vec4_t*>(q_thread), 
                                                         *reinterpret_cast<const data_vec4_t*>(k_loads[ttt]), 
                                                         qk_acc);
      qk_acc *= qk_scale;

      qk_acc = wavefrontReduce<std::plus<float>>(qk_acc);
      max_qk_acc = max(qk_acc, max_qk_acc);

      // write accumulated sums to smem.
      if (threadIdx.x == 0) {
        smem[t] = qk_acc;
      }
    }
  }

  // Use shared reduction to compute max and compute softmax on shared memory.
  // write max acc
  if (threadIdx.x == 0) {
    smem[T_MAX + wavefront_idx] = max_qk_acc;
  }
  __syncthreads();
  if (threadIdx.x < kWavefrontsPerBlock) {
    max_qk_acc = max(max_qk_acc, smem[T_MAX + threadIdx.x]);
  }
  // shared across all threads in block
  max_qk_acc = wavefrontReduce<std::greater<float>>(max_qk_acc);
  // each wavefront computes partial sum of exp.
  float softmax_denominator = 0.0f;
  for (int32_t t = threadIdx.x + wavefront_idx * kThreadsPerWavefront; t < t_max;
       t += kWavefrontsPerBlock * kThreadsPerWavefront) {
    softmax_denominator += __expf(smem[t] - max_qk_acc);
  }
  softmax_denominator = wavefrontReduce<std::plus<float>>(softmax_denominator);

  __syncthreads();
  if (threadIdx.x == 0) {
    smem[T_MAX + wavefront_idx] = softmax_denominator;
  }
  __syncthreads();
  
  // now, compute sum of exp(x - max(x)) over all intermediate results.
  softmax_denominator = 0.0;
  if (threadIdx.x < kWavefrontsPerBlock) {
    softmax_denominator = smem[T_MAX + threadIdx.x];
  }
  softmax_denominator = wavefrontReduce<std::plus<float>>(softmax_denominator);

  // now, compute the normalization across all threads.
  for (int32_t t = threadIdx.x + wavefront_idx * kThreadsPerWavefront; t < t_max;
       t += kWavefrontsPerBlock * kThreadsPerWavefront) {
    smem[t] = __expf(smem[t] - max_qk_acc) / softmax_denominator;
  }
  __syncthreads();

  // Now, we can comute the softmax and write the outputs.

  // Split T across wavefronts in a block
  // each wavefront compute sum(t_subset) P[t] * V[t_subset, d]
  // outputs are of size float[D]

  float ps[kTimeUnroll];
  float4 o_acc;
  for (auto tt = wavefront_idx * kTimeUnroll; tt < t_max_unroll;
       tt += kWavefrontsPerBlock * kTimeUnroll) {
#pragma unroll kTimeUnroll
    for (auto ttt = 0; ttt < kTimeUnroll; ++ttt) {
      int32_t t = tt + ttt;
      // &(cache_V[b][t][0][0]);
      auto* v_ = cache_V_base + t * cache_V.stride(1);
      //   scalar4<scalar_t> v_thread;
      k_loads[ttt] =
          reinterpret_cast<const read_t*>(v_) + threadIdx.x;
      ps[ttt] = smem[t];
    }

#pragma unroll kTimeUnroll
    for (auto ttt = 0; ttt < kTimeUnroll; ++ttt) {
      o_acc = scalar4_scale_acc<read_t, data_t>(o_acc, k_loads[ttt], ps[ttt]);
    }
  }


  for (auto tt = t_max_unroll + wavefront_idx; tt < t_max;
       tt += kWavefrontsPerBlock * kTimeUnroll1) {
#pragma unroll kTimeUnroll1
    for (auto ttt = 0; ttt < kTimeUnroll1; ++ttt) {
      int32_t t = tt + ttt;
      // &(cache_V[b][t][0][0]);
      auto* v_ = cache_V_base + t * cache_V.stride(1);
      //   scalar4<scalar_t> v_thread;
      k_loads[ttt] =
          reinterpret_cast<const read_t*>(v_) + threadIdx.x;
      ps[ttt] = smem[t];
    }

#pragma unroll kTimeUnroll1
    for (auto ttt = 0; ttt < kTimeUnroll1; ++ttt) {
      o_acc = scalar4_scale_acc<read_t, data_t>(o_acc, k_loads[ttt], ps[ttt]);
    }
  }
  // now, each thread has partial sums. Write to smem and get accumulated
  // results back.
  __syncthreads();

  *(reinterpret_cast<float4*>(smem) + wavefront_idx * kThreadsPerWavefront +
    threadIdx.x) = o_acc;
  __syncthreads();
  // sum up partial D rows from other wavefronts
  if (wavefront_idx == 0) {
    float4 r = make_float4(0, 0, 0, 0);
    for (int32_t w = 0; w < kWavefrontsPerBlock; ++w) {
      auto partial_r = *(
          reinterpret_cast<float4*>(smem) + w * kThreadsPerWavefront + threadIdx.x);
      r.x += partial_r.x;
      r.y += partial_r.y;
      r.z += partial_r.z;
      r.w += partial_r.w;
    }
    // write output D row
    auto* o_ = reinterpret_cast<read_t*>(&O[b][0][h][0]);
    typename c10_to_data_t<scalar_t>::vec4 bf_r;
    bf_r.x = r.x;
    bf_r.y = r.y;
    bf_r.z = r.z;
    bf_r.w = r.w;
    o_[threadIdx.x] =
        *reinterpret_cast<const read_t*>(&bf_r);
  }
}

#define AT_DISPATCH_CASE_3(SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, ...) \
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__) \
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__) \
  AT_DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__) 
  
#define AT_DISPATCH_SWITCH_3(SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                               \
      TYPE,                                                         \
      NAME,                                                         \
      AT_DISPATCH_CASE_3(SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, __VA_ARGS__))

at::Tensor
efficient_attention_forward_decoder_ck(
    const at::Tensor& XQ, // [B, 1, H, D]
    const at::Tensor& cache_K, // [B, T_MAX, H or 1, D]
    const at::Tensor& cache_V, // [B, T_MAX, H or 1, D]
    const at::Tensor& seq_positions, // [B]
    double qk_scale) {

  at::OptionalDeviceGuard guard(XQ.device());
  TORCH_CHECK(XQ.is_cuda());
  TORCH_CHECK(cache_K.is_cuda());
  TORCH_CHECK(cache_V.is_cuda());

  TORCH_CHECK(seq_positions.is_cuda());

  TORCH_CHECK(cache_K.size(1) <= T_MAX);
  TORCH_CHECK(cache_K.size(3) == D_H);

  auto O = at::empty_like(XQ);
  auto B = XQ.size(0);
  auto H = XQ.size(2);
  dim3 blocks(B, H);
  dim3 threads(kThreadsPerWavefront, kWavefrontsPerBlock);

  int32_t smem_softmax = T_MAX * sizeof(float) + kWavefrontsPerBlock * sizeof(float);
  int32_t smem_output = D_H * sizeof(float) * kWavefrontsPerBlock;
  int32_t smem = max(smem_softmax, smem_output);
  auto stream = at::cuda::getCurrentHIPStream().stream();

  AT_DISPATCH_SWITCH_3(at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::Float, 
    XQ.scalar_type(), "efficient_attention_forward_decoder_ck", [&] {
      auto* kernel = &efficient_attention_forward_decoder_ck_kernel<scalar_t>;
      if (smem > 48 * 1024) {
        C10_CUDA_CHECK(hipFuncSetAttribute(
            reinterpret_cast<void*&>(kernel),
            hipFuncAttributeMaxDynamicSharedMemorySize,
            smem));
      }
      kernel
          <<<blocks, threads, smem, stream>>>(
              XQ.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(),
              cache_K.packed_accessor64<scalar_t, 4, at::RestrictPtrTraits>(),
              cache_V.packed_accessor64<scalar_t, 4, at::RestrictPtrTraits>(),
              O.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(),
              seq_positions
                  .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
              qk_scale);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
  });

  return O;
}  

#undef AT_DISPATCH_CASE_3
#undef AT_DISPATCH_SWITCH_3

} // namespace

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::efficient_attention_forward_decoder_ck"),
      TORCH_FN(efficient_attention_forward_decoder_ck));
}