#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include <ck_tile/core.hpp>
#include <ck_tile/host/kernel_launch.hpp>
#include <ck_tile/host/stream_config.hpp>

#include "ck_tile_attention_forward_decoder_splitk.h"

namespace {
constexpr int32_t kThreadsPerWavefront = 64;
constexpr int32_t kWavefrontsPerBlock = 4;
constexpr int32_t kMaxHeadDimension = 4 * kThreadsPerWavefront;
constexpr int32_t kMaxKVSequenceLength = 4096;
constexpr int32_t kLoopUnroll = 16;
constexpr int32_t kLoopUnrollTail = 2;
using compute_t = float;
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
  using type = ck_tile::fp16_t;
};

template <>
struct c10_to_data_t<c10::BFloat16> {
  using type = ck_tile::bf16_t;
};
} // namespace

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

template <typename ck_data_t, typename compute_t, int32_t vec_size>
void instantiate_and_launch_kernels(
    typename ck_tile::ForwardDecoderSplitKArgument<ck_data_t, compute_t> arg,
    dim3 attn_grid_size,
    dim3 attn_block_size,
    int32_t lds_bytes,
    dim3 reduce_grid_size,
    dim3 reduce_block_size,
    hipStream_t stream) {
  auto attn_kernel_impl = ck_tile::ForwardDecoderSplitKAttnKernelImpl<
      ck_data_t,
      vec_size,
      kLoopUnroll,
      kLoopUnrollTail,
      kMaxKVSequenceLength,
      compute_t>{};
  auto reduce_kernel_impl = ck_tile::
      ForwardDecoderSplitKReduceKernelImpl<ck_data_t, vec_size, compute_t>{};

  (void)ck_tile::launch_kernel(
      ck_tile::stream_config{stream, /* benchmark */ false},
      ck_tile::make_kernel(
          attn_kernel_impl, attn_grid_size, attn_block_size, lds_bytes, arg));

  (void)ck_tile::launch_kernel(
      ck_tile::stream_config{stream, /* benchmark */ false},
      ck_tile::make_kernel(
          reduce_kernel_impl,
          reduce_grid_size,
          reduce_block_size,
          0 /* lds_bytes */,
          arg));
}

template <
    int32_t ThreadsPerWavefront,
    int32_t WavefrontsPerBlock>
at::Tensor& efficient_attention_forward_decoder_splitk_ck_out_impl(
    const at::Tensor& XQ, // [B, 1, G, H, D]
    const at::Tensor& cache_K, // [B, kMaxKVSequenceLength, G, H or 1, D]
    const at::Tensor& cache_V, // [B, kMaxKVSequenceLength, G, H or 1, D]
    at::optional<at::Tensor> seq_kv_lens, // [B]
    float qk_scale,
    int32_t split_k,
    at::Tensor& split_max,
    at::Tensor& split_sumexp,
    at::Tensor& split_O,
    at::Tensor& O) {
  static_assert(4 * ThreadsPerWavefront == kMaxHeadDimension, "");
  static_assert(WavefrontsPerBlock <= ThreadsPerWavefront, "");

  at::OptionalDeviceGuard guard(XQ.device());
  TORCH_CHECK(XQ.is_cuda());
  TORCH_CHECK(cache_K.is_cuda());
  TORCH_CHECK(cache_V.is_cuda());

  TORCH_CHECK(!seq_kv_lens || seq_kv_lens->is_cuda());

  TORCH_CHECK(cache_K.size(1) / split_k <= kMaxKVSequenceLength);
  TORCH_CHECK(cache_K.size(4) <= kMaxHeadDimension);

  constexpr auto rank = 5;

  auto B = XQ.size(0);
  auto M = XQ.size(1);
  auto G = XQ.size(2);
  auto H = XQ.size(3);
  auto HDim = XQ.size(4);

  TORCH_CHECK(B <= 1024);
  TORCH_CHECK(M <= 1024);
  TORCH_CHECK(H <= 1024);

  const dim3 attn_grid_size(B * H * M * G, split_k);
  const dim3 attn_block_size(ThreadsPerWavefront, WavefrontsPerBlock);

  const dim3 reduce_grid_size = {attn_grid_size.x};
  const dim3 reduce_block_size = {attn_block_size.x};

  int32_t smem_softmax = kMaxKVSequenceLength * sizeof(compute_t) +
      WavefrontsPerBlock * sizeof(compute_t);
  int32_t smem_output = kMaxHeadDimension * sizeof(compute_t) *
      WavefrontsPerBlock; // 4 * threadsPerBlock * sizeof(float) ==
                          // sizeof(O[b][0][h][:])
  const size_t attn_lds_bytes = max(smem_softmax, smem_output);
  auto stream = at::hip::getCurrentHIPStream().stream();

  AT_DISPATCH_SWITCH_3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Float,
      XQ.scalar_type(),
      "efficient_attention_forward_decoder_splitk_ck",
      [&] {
        using ck_data_t = c10_to_data_t<scalar_t>::type;

        auto XQ_acc =
            XQ.packed_accessor32<scalar_t, rank, at::RestrictPtrTraits>();
        auto K_acc =
            cache_K.packed_accessor64<scalar_t, rank, at::RestrictPtrTraits>();
        auto V_acc =
            cache_V.packed_accessor64<scalar_t, rank, at::RestrictPtrTraits>();
        auto split_O_acc =
            split_O
                .packed_accessor32<scalar_t, 1 + rank, at::RestrictPtrTraits>();
        auto O_acc =
            O.packed_accessor32<scalar_t, rank, at::RestrictPtrTraits>();
        auto seq_acc_ptr = seq_kv_lens
            ? seq_kv_lens
                  ->packed_accessor32<int32_t, 1, at::RestrictPtrTraits>()
                  .data()
            : nullptr;
        auto split_max_acc =
            split_max.packed_accessor32<float, rank, at::RestrictPtrTraits>();
        auto split_sumexp_acc =
            split_sumexp
                .packed_accessor32<float, rank, at::RestrictPtrTraits>();
        auto arg = ck_tile::ForwardDecoderSplitKArgument<ck_data_t, compute_t>{
            reinterpret_cast<const ck_data_t* __restrict__>(XQ_acc.data()),
            reinterpret_cast<const ck_data_t* __restrict__>(K_acc.data()),
            reinterpret_cast<const ck_data_t* __restrict__>(V_acc.data()),
            reinterpret_cast<ck_data_t* __restrict__>(O_acc.data()),
            reinterpret_cast<ck_data_t* __restrict__>(split_O_acc.data()),
            split_max_acc.data(),
            split_sumexp_acc.data(),
            seq_acc_ptr,
            XQ_acc.stride(0),
            XQ_acc.stride(1),
            XQ_acc.stride(2),
            XQ_acc.stride(3),
            K_acc.stride(0),
            K_acc.stride(1),
            K_acc.stride(2),
            K_acc.stride(3),
            split_O_acc.stride(0),
            static_cast<int32_t>(XQ_acc.size(1)),
            static_cast<int32_t>(XQ_acc.size(2)),
            static_cast<int32_t>(XQ_acc.size(3)),
            static_cast<int32_t>(XQ_acc.size(4)),
            static_cast<int32_t>(K_acc.size(1)),
            K_acc.size(3) == 1,
            qk_scale,
            split_k};

        auto required_vec_size = 0;

        for (auto vec_size : {4, 2, 1}) {
          if (arg.Q_size_k <= vec_size * ThreadsPerWavefront) {
            required_vec_size = vec_size;
          }
        }

        TORCH_CHECK(required_vec_size > 0);

        switch (required_vec_size) {
          case 4:
            instantiate_and_launch_kernels<ck_data_t, compute_t, 4>(
                arg,
                attn_grid_size,
                attn_block_size,
                attn_lds_bytes,
                reduce_grid_size,
                reduce_block_size,
                stream);
            break;
          case 2:
            instantiate_and_launch_kernels<ck_data_t, compute_t, 2>(
                arg,
                attn_grid_size,
                attn_block_size,
                attn_lds_bytes,
                reduce_grid_size,
                reduce_block_size,
                stream);
            break;
          case 1:
            instantiate_and_launch_kernels<ck_data_t, compute_t, 1>(
                arg,
                attn_grid_size,
                attn_block_size,
                attn_lds_bytes,
                reduce_grid_size,
                reduce_block_size,
                stream);
            break;
          default:
            break;
        }
      });

  return O;
}

template <int32_t ThreadsPerWavefront, int32_t WavefrontsPerBlock>
at::Tensor efficient_attention_forward_decoder_splitk_ck_impl(
    const at::Tensor& XQ, // [B, 1, G, H, D]
    const at::Tensor& cache_K, // [B, kMaxKVSequenceLength, G, H or 1, D]
    const at::Tensor& cache_V, // [B, kMaxKVSequenceLength, H or 1, D]
    at::optional<at::Tensor> seq_kv_lens, // [B]
    float qk_scale,
    int32_t split_k) {
  auto O = at::empty_like(XQ);
  constexpr auto rank = 5;

  TORCH_CHECK(XQ.dim() == rank);
  TORCH_CHECK(cache_K.dim() == rank);
  TORCH_CHECK(cache_V.dim() == rank);

  auto B = XQ.size(0);
  auto M = XQ.size(1);
  auto G = XQ.size(2);
  auto H = XQ.size(3);
  auto K = XQ.size(4);

  auto O_splits = at::empty({split_k, B, M, G, H, K}, XQ.options());
  auto split_max =
      at::empty({B, M, G, H, split_k}, XQ.options().dtype(at::kFloat));
  auto split_sumexp = at::empty_like(split_max);

  efficient_attention_forward_decoder_splitk_ck_out_impl<
      ThreadsPerWavefront,
      WavefrontsPerBlock>(
      XQ,
      cache_K,
      cache_V,
      seq_kv_lens,
      qk_scale,
      split_k,
      split_max,
      split_sumexp,
      O_splits,
      O);

  return O;
}

at::Tensor efficient_attention_forward_decoder_splitk_ck(
    const at::Tensor& XQ, // [B, 1, G, H, D]
    const at::Tensor& cache_K, // [B, kMaxKVSequenceLength, G, H or 1, D]
    const at::Tensor& cache_V, // [B, kMaxKVSequenceLength, G, H or 1, D]
    at::optional<at::Tensor> seq_kv_lens, // [B]
    double qk_scale,
    int64_t split_k) {
  return efficient_attention_forward_decoder_splitk_ck_impl<
      kThreadsPerWavefront,
      kWavefrontsPerBlock>(
      XQ,
      cache_K,
      cache_V,
      seq_kv_lens,
      static_cast<float>(qk_scale),
      static_cast<int32_t>(split_k));
}
} // namespace

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME(
          "xformers::efficient_attention_forward_decoder_splitk_ck"),
      TORCH_FN(efficient_attention_forward_decoder_splitk_ck));
}

#undef AT_DISPATCH_CASE_3
#undef AT_DISPATCH_SWITCH_3
