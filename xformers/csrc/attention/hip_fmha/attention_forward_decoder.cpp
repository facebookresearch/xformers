/*
  TODO: license header
*/

// #include <ck/ck.hpp>
#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include "ck_attention_forward_decoder.h"

namespace {
  constexpr int32_t kThreadsPerWavefront = 64;
  constexpr int32_t kWavefrontsPerBlock = 16;
  constexpr int32_t D_H = 4 * kThreadsPerWavefront;
}

namespace {

template <typename c10_t>
struct c10_to_data_t;
template <>
struct c10_to_data_t<float> {
  using type = float;
};

template <>
struct c10_to_data_t<c10::Half> {
  using type = ck::half_t;
};

template <>
struct c10_to_data_t<c10::BFloat16> {
  using type = ck::bhalf_t;
};
}

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

template <int32_t ThreadsPerWavefront, int32_t WavefrontsPerBlock,
          int32_t T_MAX = 8192,
          int32_t D_H = 256>
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
  TORCH_CHECK(cache_K.size(3) <= D_H);

  auto B = XQ.size(0);
  auto M = XQ.size(1);
  auto H = XQ.size(2);

  TORCH_CHECK(B <= 1024);
  TORCH_CHECK(M <= 1024);
  TORCH_CHECK(H <= 1024);

  dim3 blocks(B, H, M);
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
        using ck_data_t = c10_to_data_t<scalar_t>::type;
        using device_op_t =
            ck::tensor_operation::device::FMHADecoderSeqlen1DeviceOp<ck_data_t>;
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
            reinterpret_cast<const ck_data_t* __restrict__>(XQ_acc.data()),
            reinterpret_cast<const ck_data_t* __restrict__>(K_acc.data()),
            reinterpret_cast<const ck_data_t* __restrict__>(V_acc.data()),
            reinterpret_cast<ck_data_t* __restrict__>(O_acc.data()),
            seq_acc.data(),
            XQ_acc.stride(0),
            XQ_acc.stride(1),
            XQ_acc.stride(2),
            K_acc.stride(0),
            K_acc.stride(1),
            K_acc.stride(2),
            K_acc.size(3),
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