/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include "ck_attention_forward_decoder.h"

namespace {
constexpr int32_t kThreadsPerWavefront = 64;
constexpr int32_t kWavefrontsPerBlock = 16;
constexpr int32_t K_MAX = 4 * kThreadsPerWavefront;
} // namespace

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
} // namespace

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

template <
    int32_t ThreadsPerWavefront,
    int32_t WavefrontsPerBlock,
    int32_t KV_M_MAX = 8192,
    int32_t K_MAX = 256>
at::Tensor& efficient_attention_forward_decoder_ck_out_impl(
    const at::Tensor& XQ, // [B, 1, G, H, D]
    const at::Tensor& cache_K, // [B, KV_M_MAX, G, H or 1, D]
    const at::Tensor& cache_V, // [B, KV_M_MAX, G, H or 1, D]
    at::optional<at::Tensor> seq_kv_lens, // [B]
    double qk_scale,
    at::Tensor& O) {
  static_assert(4 * ThreadsPerWavefront == K_MAX, "");
  static_assert(WavefrontsPerBlock <= ThreadsPerWavefront, "");

  at::OptionalDeviceGuard guard(XQ.device());
  TORCH_CHECK(XQ.is_cuda());
  TORCH_CHECK(cache_K.is_cuda());
  TORCH_CHECK(cache_V.is_cuda());

  TORCH_CHECK(!seq_kv_lens || seq_kv_lens->is_cuda());

  TORCH_CHECK(cache_K.size(1) <= KV_M_MAX);
  TORCH_CHECK(cache_K.size(4) <= K_MAX);

  constexpr auto rank = 5;

  auto B = XQ.size(0);
  auto M = XQ.size(1);
  auto G = XQ.size(2);
  auto H = XQ.size(3);

  TORCH_CHECK(B <= 1024);
  TORCH_CHECK(M <= 1024);
  TORCH_CHECK(H <= 1024);

  dim3 blocks(B * H * M * G);
  dim3 threads(ThreadsPerWavefront, WavefrontsPerBlock);

  int32_t smem_softmax = KV_M_MAX * sizeof(float) + threads.y * sizeof(float);
  int32_t smem_output = K_MAX * sizeof(float) *
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
            XQ.packed_accessor32<scalar_t, rank, at::RestrictPtrTraits>();
        auto K_acc =
            cache_K.packed_accessor64<scalar_t, rank, at::RestrictPtrTraits>();
        auto V_acc =
            cache_V.packed_accessor64<scalar_t, rank, at::RestrictPtrTraits>();
        auto O_acc =
            O.packed_accessor32<scalar_t, rank, at::RestrictPtrTraits>();
        auto seq_acc = seq_kv_lens
            ? seq_kv_lens
                  ->packed_accessor32<int32_t, 1, at::RestrictPtrTraits>()
                  .data()
            : nullptr;
        auto arg = device_op_t::Argument(
            reinterpret_cast<const ck_data_t* __restrict__>(XQ_acc.data()),
            reinterpret_cast<const ck_data_t* __restrict__>(K_acc.data()),
            reinterpret_cast<const ck_data_t* __restrict__>(V_acc.data()),
            reinterpret_cast<ck_data_t* __restrict__>(O_acc.data()),
            seq_acc,
            XQ_acc.stride(0),
            XQ_acc.stride(1),
            XQ_acc.stride(2),
            XQ_acc.stride(3),
            K_acc.stride(0),
            K_acc.stride(1),
            K_acc.stride(2),
            K_acc.stride(3),
            XQ_acc.size(1),
            XQ_acc.size(2),
            XQ_acc.size(3),
            XQ_acc.size(4),
            K_acc.size(1),
            K_acc.size(3) == 1,
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
    const at::Tensor& XQ, // [B, 1, G, H, D]
    const at::Tensor& cache_K, // [B, KV_M_MAX, G, H or 1, D]
    const at::Tensor& cache_V, // [B, KV_M_MAX, H or 1, D]
    at::optional<at::Tensor> seq_kv_lens, // [B]
    double qk_scale) {
  auto O = at::empty_like(XQ);
  efficient_attention_forward_decoder_ck_out_impl<
      ThreadsPerWavefront,
      WavefrontsPerBlock>(XQ, cache_K, cache_V, seq_kv_lens, qk_scale, O);
  return O;
}

at::Tensor efficient_attention_forward_decoder_ck(
    const at::Tensor& XQ, // [B, 1, G, H, D]
    const at::Tensor& cache_K, // [B, KV_M_MAX, G, H or 1, D]
    const at::Tensor& cache_V, // [B, KV_M_MAX, G, H or 1, D]
    at::optional<at::Tensor> seq_kv_lens, // [B]
    double qk_scale) {
  return efficient_attention_forward_decoder_ck_impl<
      kThreadsPerWavefront,
      kWavefrontsPerBlock>(XQ, cache_K, cache_V, seq_kv_lens, qk_scale);
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
 > mkdir build
 > cd build
 > cmake /xformers/xformers/csrc/attention/hip_fmha/ \
       -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
       -D CMAKE_PREFIX_PATH=/opt/rocm \
       -D CMAKE_BUILD_TYPE=Debug \
       -D GPU_TARGETS="native" 
  > make

(3a) run correctness check
 > ./attention_forward_decoder_main

(3b) run specific input shape
 > ./attention_forward_decoder_main n_keys padding batch_size n_heads is_multiquery dtype n_wavefronts_per_block
*/

// clang-format on

static void do_correctness_check() {
  const int32_t D = 4 * kThreadsPerWavefront;
  const int32_t B = 1;
  const int32_t H = 4;
  const int32_t G = 1;
  auto options = torch::TensorOptions()
                     .dtype(torch::kFloat32)
                     .layout(torch::kStrided)
                     .device(torch::kCUDA, 1)
                     .requires_grad(false);
  auto int_options = options.dtype(torch::kInt);
  auto XQ = at::randn({B, 1, G, H, D}, options);
  auto K = at::randn({B, 4096, G, H, D}, options);
  auto V = at::randn({B, 4096, G, H, D}, options);
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
          << "Usage: ./a.out n_keys padding batch_size n_heads is_multiquery dtype "
             "n_wavefronts_per_block"
          << std::endl;
      return 0;
    }
    const int32_t n_keys = std::stoi(args[0]);
    const int32_t padding = std::stoi(args[1]);
    const int32_t batch_size = std::stoi(args[2]);
    const int32_t n_heads = std::stoi(args[3]);
    const int32_t n_groups = 1;
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
    const auto Q =
        at::rand({batch_size, 1, n_groups, n_heads, dim_per_head}, options);
    const auto K = multiquery
        ? at::rand({batch_size, padding, n_groups, 1, dim_per_head}, options)
              .expand({batch_size, padding, n_groups, n_heads, dim_per_head})
        : at::rand(
              {batch_size, padding, n_groups, n_heads, dim_per_head}, options);
    const auto V = at::rand_like(K);
    auto O = at::empty_like(Q);

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