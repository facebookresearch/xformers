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
constexpr int32_t kWavefrontsPerBlock = 1;
constexpr int32_t D_H = 256;
constexpr int32_t T_MAX = 8192;

template<typename c10_t>
struct c10_to_data_t;

template<>
struct c10_to_data_t<float> {
    using type = float;
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

template<typename data4_t>
__device__
ck::float4_t scalar4_scale_acc(ck::float4_t acc, data4_t a, float b);

template<>
__device__  
ck::float4_t
scalar4_scale_acc<ck::float4_t>(ck::float4_t acc, ck::float4_t a, float b) {
  return acc + a * b;
}

template<>
__device__  
ck::float4_t
scalar4_scale_acc<ck::half4_t>(ck::float4_t acc, ck::half4_t a, float b) {
  acc.x += ck::type_convert<float>(a.x) * b;
  acc.y += ck::type_convert<float>(a.y) * b;
  acc.z += ck::type_convert<float>(a.z) * b;
  acc.w += ck::type_convert<float>(a.w) * b;
  return acc;
}

template<>
__device__  
ck::float4_t
scalar4_scale_acc<ck::bhalf4_t>(ck::float4_t acc, ck::bhalf4_t a, float b) {
  acc.x += ck::type_convert<float>(a.x) * b;
  acc.y += ck::type_convert<float>(a.y) * b;
  acc.z += ck::type_convert<float>(a.z) * b;
  acc.w += ck::type_convert<float>(a.w) * b;
  return acc;
}

template <typename F>
float
__device__ __forceinline__ wavefrontReduce(float val) {
  auto reducer = F();  
#pragma unroll
  for (int32_t mask = kThreadsPerWavefront >> 1; mask > 0; mask >>= 1) {
    val = reducer(val, __shfl_xor(val, mask, kThreadsPerWavefront));
  }
  return val;
}

template <typename TDataPtr, typename TDataVec>
__device__ TDataVec load_v(TDataPtr data_ptr, int32_t vector_offset) {
  return *(reinterpret_cast<const TDataVec*>(data_ptr) + vector_offset);
}

template <typename TDataPtr, typename TDataVec>
__device__ void store_v(TDataPtr data_ptr, int32_t vector_offset, TDataVec value) {
  *(reinterpret_cast<TDataVec*>(data_ptr) + vector_offset) = value;
}

template<typename scalar_t>
__global__ void
efficient_attention_forward_decoder_ck_kernel(
    at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> XQ,
    at::PackedTensorAccessor64<scalar_t, 4, at::RestrictPtrTraits> cache_K,
    at::PackedTensorAccessor64<scalar_t, 4, at::RestrictPtrTraits> cache_V,
    at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> O,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> seq_positions,
    const float qk_scale
) {
  static_assert(4 * kThreadsPerWavefront == D_H, "");
  static_assert(kWavefrontsPerBlock <= kThreadsPerWavefront, "");

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
  const int32_t threads_per_block = threads_per_wavefront * wavefronts_per_block;
  const int32_t thread_linear_idx = lane_idx + wavefront_idx * threads_per_wavefront;

  // Need D_H == 256 (NB: 128 in CUDA because of wavefront/warp sizes 64/32)
  const auto* q_ = &(XQ[b][0][h][0]);

  const bool multiquery = cache_K.size(2) == 1;
  const auto* cache_K_base = &cache_K[b][0][multiquery ? 0 : h][0];
  const auto* cache_V_base = &cache_V[b][0][multiquery ? 0 : h][0];

  // Load Q into registers in all wavefronts.
  // Each thread handles 4 D dimensions
  using data_t = typename c10_to_data_t<scalar_t>::type;
  using data_vec4_t = typename c10_to_data_t<scalar_t>::vec4;
  const data_vec4_t q_thread = load_v<decltype(q_), data_vec4_t>(q_, lane_idx);
  // Each block computes different B value
  float max_qk_acc = std::numeric_limits<float>::lowest();

  // Compute S[T_MAX] = for i in range(T): S[t] = sum(Q[d] * K[t, d])
  // Split T across wavefronts in a block, unroll loads to expose more
  // parallelism.

  constexpr int32_t kTimeUnroll = 1;
  data_vec4_t k_loads[kTimeUnroll];

  const auto dtt = wavefronts_per_block * kTimeUnroll;
  const int32_t t_max_unroll =
    (t_max / dtt) * dtt;

  for (auto tt = wavefront_idx; tt < t_max_unroll; tt += dtt) {
#pragma unroll kTimeUnroll
    for (auto ttt = 0; ttt < kTimeUnroll; ++ttt) {
      const int32_t t = tt + ttt;
      // &(cache_K[b][t][0][0]);
      auto* k_ = cache_K_base + t * cache_K.stride(1);
      // scalar4<scalar_t> k_thread;
      k_loads[ttt] = load_v<decltype(k_), data_vec4_t>(k_, lane_idx);
    }
#pragma unroll kTimeUnroll
    for (auto ttt = 0; ttt < kTimeUnroll; ++ttt) {
      float qk_acc = 0;
      const int32_t t = tt + ttt;

      ck::inner_product<data_vec4_t, data_vec4_t, float>(q_thread, 
                                                         k_loads[ttt], 
                                                         qk_acc);
      qk_acc *= qk_scale;

      qk_acc = wavefrontReduce<std::plus<float>>(qk_acc);
      max_qk_acc = max(qk_acc, max_qk_acc);

      // write accumulated sums to smem.
      if (lane_idx == 0) {
        smem[t] = qk_acc;
      }
    }
  }

  constexpr int32_t kTimeUnroll1 = 1;
  for (auto tt = t_max_unroll + wavefront_idx; tt < t_max;
       tt += wavefronts_per_block * kTimeUnroll1) {
#pragma unroll kTimeUnroll1
    for (auto ttt = 0; ttt < kTimeUnroll1; ++ttt) {
      const int32_t t = tt + ttt;
      // &(cache_K[b][t][0][0]);
      auto* k_ = cache_K_base + t * cache_K.stride(1);
      // scalar4<scalar_t> k_thread;
      k_loads[ttt] = load_v<decltype(k_), data_vec4_t>(k_, lane_idx);
    }
#pragma unroll kTimeUnroll1
    for (auto ttt = 0; ttt < kTimeUnroll1; ++ttt) {
      float qk_acc = 0;
      const int32_t t = tt + ttt;
      ck::inner_product<data_vec4_t, data_vec4_t, float>(q_thread, 
                                                         k_loads[ttt], 
                                                         qk_acc);
      qk_acc *= qk_scale;

      qk_acc = wavefrontReduce<std::plus<float>>(qk_acc);
      max_qk_acc = max(qk_acc, max_qk_acc);

      // write accumulated sums to smem.
      if (lane_idx == 0) {
        smem[t] = qk_acc;
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
  max_qk_acc = wavefrontReduce<std::greater<float>>(max_qk_acc);
  // each wavefront computes partial sum of exp.
  float softmax_denominator = 0.0f;
  for (int32_t t = thread_linear_idx; t < t_max; t += threads_per_block) {
    softmax_denominator += expf(smem[t] - max_qk_acc);
  }
  softmax_denominator = wavefrontReduce<std::plus<float>>(softmax_denominator);

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
  softmax_denominator = wavefrontReduce<std::plus<float>>(softmax_denominator);
  
  // now, compute the normalization across all threads.
  for (int32_t t = thread_linear_idx; t < t_max; t += threads_per_block) {
    smem[t] = expf(smem[t] - max_qk_acc) / softmax_denominator;
  }
  __syncthreads();

  // Now, we can compute the softmax and write the outputs.

  // Split T across wavefronts in a block
  // each wavefront compute sum(t_subset) P[t] * V[t_subset, d]
  // outputs are of size float[D]

  float ps[kTimeUnroll];
  ck::float4_t o_acc = 0;
  for (auto tt = wavefront_idx; tt < t_max_unroll; tt += dtt) {
#pragma unroll kTimeUnroll
    for (auto ttt = 0; ttt < kTimeUnroll; ++ttt) {
      const int32_t t = tt + ttt;
      // &(cache_V[b][t][0][0]);
      auto* v_ = cache_V_base + t * cache_V.stride(1);
      //   scalar4<scalar_t> v_thread;
      k_loads[ttt] = load_v<decltype(v_), data_vec4_t>(v_, lane_idx);

      ps[ttt] = smem[t];
    }

#pragma unroll kTimeUnroll
    for (auto ttt = 0; ttt < kTimeUnroll; ++ttt) {
      o_acc = scalar4_scale_acc<data_vec4_t>(o_acc, k_loads[ttt], ps[ttt]);
    }
  }

  for (auto tt = t_max_unroll + wavefront_idx; tt < t_max; tt += wavefronts_per_block * kTimeUnroll1) {
#pragma unroll kTimeUnroll1
    for (auto ttt = 0; ttt < kTimeUnroll1; ++ttt) {
      const int32_t t = tt + ttt;
      // &(cache_V[b][t][0][0]);
      auto* v_ = cache_V_base + t * cache_V.stride(1);
      //   scalar4<scalar_t> v_thread;
      k_loads[ttt] = load_v<decltype(v_), data_vec4_t>(v_, lane_idx);

      ps[ttt] = smem[t];
    }

#pragma unroll kTimeUnroll1
    for (auto ttt = 0; ttt < kTimeUnroll1; ++ttt) {
      o_acc = scalar4_scale_acc<data_vec4_t>(o_acc, k_loads[ttt], ps[ttt]);
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
      auto partial_r = load_v<float*, ck::float4_t>(smem, w * threads_per_wavefront + lane_idx);
      r += partial_r;
    }
    // write output D row
    data_vec4_t bf_r;
    bf_r.x = ck::type_convert<data_t>(r.x);
    bf_r.y = ck::type_convert<data_t>(r.y);
    bf_r.z = ck::type_convert<data_t>(r.z);
    bf_r.w = ck::type_convert<data_t>(r.w);
    auto* o_ = &O[b][0][h][0];
    store_v<decltype(o_), data_vec4_t>(o_, lane_idx, bf_r);
  }
}

void update_max_dynamic_shared_memory_size_bytes(void* kernel_func, int32_t new_value) {
  hipFuncAttributes attributes;
  C10_CUDA_CHECK(hipFuncGetAttributes(
      &attributes, 
      kernel_func));

  const auto default_value = attributes.maxDynamicSharedSizeBytes;

  // printf("Default smem size: %d\n", default_value);

  if (new_value > default_value) {
    C10_CUDA_CHECK(hipFuncSetAttribute(
        kernel_func,
        hipFuncAttributeMaxDynamicSharedMemorySize,
        new_value));
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

template<int32_t ThreadsPerWavefront, int32_t WavefrontsPerBlock>
at::Tensor
efficient_attention_forward_decoder_ck_impl(
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
  dim3 threads(ThreadsPerWavefront, WavefrontsPerBlock);

  int32_t smem_softmax = T_MAX * sizeof(float) + threads.y * sizeof(float);
  int32_t smem_output = D_H * sizeof(float) * threads.y; // 4 * threadsPerBlock * sizeof(float) == sizeof(O[b][0][h][:])
  int32_t smem_size = max(smem_softmax, smem_output);
  auto stream = at::cuda::getCurrentHIPStream().stream();

  AT_DISPATCH_SWITCH_3(at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::Float, 
    XQ.scalar_type(), "efficient_attention_forward_decoder_ck", [&] {
      auto* kernel = &efficient_attention_forward_decoder_ck_kernel<scalar_t>;
      update_max_dynamic_shared_memory_size_bytes(reinterpret_cast<void*&>(kernel), smem_size);
      kernel
          <<<blocks, threads, smem_size, stream>>>(
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

at::Tensor
efficient_attention_forward_decoder_ck(
    const at::Tensor& XQ, // [B, 1, H, D]
    const at::Tensor& cache_K, // [B, T_MAX, H or 1, D]
    const at::Tensor& cache_V, // [B, T_MAX, H or 1, D]
    const at::Tensor& seq_positions, // [B]
    double qk_scale) {
  return efficient_attention_forward_decoder_ck_impl<kThreadsPerWavefront, kWavefrontsPerBlock> (
    XQ, cache_K, cache_V, seq_positions, qk_scale
  );
}
} // namespace

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::efficient_attention_forward_decoder_ck"),
      TORCH_FN(efficient_attention_forward_decoder_ck));
}

#ifdef ATTN_FWD_DECODER_MAIN

#include <torch/torch.h>

/*

(1) hipify
 > pip install -e /xformers
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

(3) run
 > LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/envs/py_3.8/lib/python3.8/site-packages/torch/lib ./a.out
*/

int main(int argc, char** argv) {
  const int32_t D = 256;
  const int32_t B = 4;
  const int32_t H = 8;
  auto options = torch::TensorOptions()
    .dtype(torch::kFloat32)
    .layout(torch::kStrided)
    .device(torch::kCUDA, 1)
    .requires_grad(false);
  auto int_options = options.dtype(torch::kInt);
  auto XQ = at::randn({B, 1, H, D}, options);
  auto K = at::randn({B, T_MAX / 2, H, D}, options);
  auto V = at::randn({B, T_MAX / 2, H, D}, options);
  auto seq = at::randint(1, 32, {B}, int_options);
  double qk_scale = 1. / sqrt(D);

  auto result = efficient_attention_forward_decoder_ck_impl<64, 1>(XQ, K, V, seq, qk_scale);
  auto gold_result = efficient_attention_forward_decoder_ck_impl<64, 2>(XQ, K, V, seq, qk_scale);
  auto mask = at::isclose(result, gold_result, 1e-2, 1e-2, false);
  auto percent_match = at::sum(mask.to(torch::kFloat32)) / mask.numel();
  printf("Mismatched elements percentage: %.2f\n", 1. - percent_match.item<float>());
  return 0;
}

#endif // MAIN