#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include "ck_attention_forward_decoder_splitk.h"

namespace {
  constexpr int32_t kThreadsPerWavefront = 64;
  constexpr int32_t kWavefrontsPerBlock = 16;
  constexpr int32_t K_MAX = 4 * kThreadsPerWavefront;
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

namespace {

// at::Tensor efficient_attention_forward_decoder_splitk_ck(
//     const at::Tensor& XQ, // [B, 1, G, H, D]
//     const at::Tensor& cache_K, // [B, KV_M_MAX, G, H or 1, D]
//     const at::Tensor& cache_V, // [B, KV_M_MAX, G, H or 1, D]
//     at::optional<at::Tensor> seq_kv_lens, // [B]
//     double qk_scale,
//     at::Tensor& O,
//     int64_t split_k) {
    
//     at::OptionalDeviceGuard guard(XQ.device());

//     TORCH_CHECK(XQ.is_cuda());
//     TORCH_CHECK(cache_K.is_cuda());
//     TORCH_CHECK(cache_V.is_cuda());

//     TORCH_CHECK(seq_positions.is_cuda());

//     auto M = XQ.size(1);
//     auto B = XQ.size(0);
//     auto G = XQ.size(2);
//     auto H = XQ.size(3);
//     auto K_q = XQ.size(4);
//     auto M_k = cache_K.size(1);

//     constexpr auto BLOCK_M = 16;
//     auto M_ceil = (M + BLOCK_M - 1) / BLOCK_M * BLOCK_M;

//     constexpr auto kThreadsPerWarp = 64;
//     constexpr auto kWarpsPerBlock = 2; // original uses 2 warps

//     const auto options = at::TensorOptions()
//                              .dtype(XQ.dtype())
//                              .layout(at::kStrided)
//                              .device(XQ.device())
//                              .requires_grad(false);

//     auto O_splitk = at::empty({B * G * H, split_k, M_ceil, K_q}, options);
//     auto metadata = at::empty({B * G * H, 2, split_k, M_ceil}, options);

//     dim3 attention_grid = {static_cast<uint32_t>(M / BLOCK_M), static_cast<uint32_t>(B * G * H), static_cast<uint32_t>(split_k)};
//     dim3 reduce_grid = {static_cast<uint32_t>(B * G * H), static_cast<uint32_t>(M)};

//     dim3 threads = {kThreadsPerWarp * kWarpsPerBlock};
    
//     auto O = at::empty_like(XQ);

//     return O;
// }

template <int32_t ThreadsPerWavefront, int32_t WavefrontsPerBlock,
          int32_t KV_M_MAX = 8192,
          int32_t K_MAX = 256>
at::Tensor& efficient_attention_forward_decoder_splitk_ck_out_impl(
    const at::Tensor& XQ, // [B, 1, G, H, D]
    const at::Tensor& cache_K, // [B, KV_M_MAX, G, H or 1, D]
    const at::Tensor& cache_V, // [B, KV_M_MAX, G, H or 1, D]
    at::optional<at::Tensor> seq_kv_lens, // [B]
    double qk_scale,
    int64_t split_k,
    at::Tensor& split_max,
    at::Tensor& split_sumexp,
    at::Tensor& split_O,
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

  dim3 blocks(B * H * M * G, split_k);
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
      "efficient_attention_forward_decoder_splitk_ck",
      [&] {
        using ck_data_t = c10_to_data_t<scalar_t>::type;
        using device_op_t =
            ck::tensor_operation::device::FMHADecoderSplitKDeviceOp<ck_data_t>;
        auto op = device_op_t{};

        auto XQ_acc =
            XQ.packed_accessor32<scalar_t, rank, at::RestrictPtrTraits>();
        auto K_acc =
            cache_K.packed_accessor64<scalar_t, rank, at::RestrictPtrTraits>();
        auto V_acc =
            cache_V.packed_accessor64<scalar_t, rank, at::RestrictPtrTraits>();
        auto split_O_acc = split_O.packed_accessor32<scalar_t, 1 + rank, at::RestrictPtrTraits>();
        auto O_acc = O.packed_accessor32<scalar_t, rank, at::RestrictPtrTraits>();
        auto seq_acc = seq_kv_lens ?
            seq_kv_lens->packed_accessor32<int32_t, 1, at::RestrictPtrTraits>().data() : nullptr;
        auto split_max_acc = split_max.packed_accessor32<float, rank, at::RestrictPtrTraits>();
        auto split_sumexp_acc = split_sumexp.packed_accessor32<float, rank, at::RestrictPtrTraits>();
        auto arg = device_op_t::Argument(
            reinterpret_cast<const ck_data_t* __restrict__>(XQ_acc.data()),
            reinterpret_cast<const ck_data_t* __restrict__>(K_acc.data()),
            reinterpret_cast<const ck_data_t* __restrict__>(V_acc.data()),
            reinterpret_cast<ck_data_t* __restrict__>(O_acc.data()),
            reinterpret_cast<ck_data_t* __restrict__>(split_O_acc.data()),
            split_max_acc.data(),
            split_sumexp_acc.data(),
            seq_acc,
            XQ_acc.stride(0),
            XQ_acc.stride(1),
            XQ_acc.stride(2),
            XQ_acc.stride(3),
            K_acc.stride(0),
            K_acc.stride(1),
            K_acc.stride(2),
            K_acc.stride(3),
            O_acc.stride(2),
            XQ_acc.size(1),
            XQ_acc.size(2),
            XQ_acc.size(3),
            XQ_acc.size(4),
            K_acc.size(1),
            K_acc.size(3) == 1,
            qk_scale,
            split_k,
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
at::Tensor efficient_attention_forward_decoder_splitk_ck_impl(
    const at::Tensor& XQ, // [B, 1, G, H, D]
    const at::Tensor& cache_K, // [B, KV_M_MAX, G, H or 1, D]
    const at::Tensor& cache_V, // [B, KV_M_MAX, H or 1, D]
    at::optional<at::Tensor> seq_kv_lens, // [B]
    int64_t split_k,
    double qk_scale) {
  auto O = at::empty_like(XQ);
  constexpr auto splitk_dim = 0;
  // auto O_unsqueeze = at::unsqueeze(O, splitk_dim);
  auto O_splits = at::stack(O, splitk_dim);

  TORCH_CHECK(XQ.dim() == 5);
  TORCH_CHECK(cache_K.dim() == 5);
  TORCH_CHECK(cache_V.dim() == 5);
  TORCH_CHECK(O_splits.dim() == 6);

  auto B = XQ.size(0);
  auto M = XQ.size(1);
  auto G = XQ.size(2);
  auto H = XQ.size(3);

  auto split_max = at::empty({B, M, G, H, split_k}, XQ.options().dtype(at::kFloat));
  auto split_sumexp = at::empty_like(split_max);

  efficient_attention_forward_decoder_splitk_ck_out_impl<
      ThreadsPerWavefront,
      WavefrontsPerBlock>(XQ, cache_K, cache_V, seq_kv_lens, qk_scale, split_k, split_max, split_sumexp, O_splits, O);
  return O;
}

at::Tensor efficient_attention_forward_decoder_splitk_ck(
    const at::Tensor& XQ, // [B, 1, G, H, D]
    const at::Tensor& cache_K, // [B, KV_M_MAX, G, H or 1, D]
    const at::Tensor& cache_V, // [B, KV_M_MAX, G, H or 1, D]
    at::optional<at::Tensor> seq_kv_lens, // [B]
    double qk_scale,
    int64_t split_k) {
  return efficient_attention_forward_decoder_splitk_ck_impl<
      kThreadsPerWavefront,
      kWavefrontsPerBlock>(XQ, cache_K, cache_V, seq_kv_lens, qk_scale, split_k);
}
} // namespace


TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::efficient_attention_forward_decoder_splitk_ck"),
      TORCH_FN(efficient_attention_forward_decoder_splitk_ck));
}

#undef AT_DISPATCH_CASE_3
#undef AT_DISPATCH_SWITCH_3