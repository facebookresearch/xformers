#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

namespace {

at::Tensor efficient_attention_forward_decoder_splitk_ck(
    const at::Tensor& XQ, // [B, 1, H, D]
    const at::Tensor& cache_K, // [B, T_MAX, H or 1, D]
    const at::Tensor& cache_V, // [B, T_MAX, H or 1, D]
    const at::Tensor& seq_positions, // [B]
    double qk_scale,
    int64_t split_k) {
    
    at::OptionalDeviceGuard guard(XQ.device());

    TORCH_CHECK(XQ.is_cuda());
    TORCH_CHECK(cache_K.is_cuda());
    TORCH_CHECK(cache_V.is_cuda());

    TORCH_CHECK(seq_positions.is_cuda());

    auto B = XQ.size(0);
    auto M = XQ.size(1);
    auto G = XQ.size(2);
    auto H = XQ.size(3);
    auto K_q = XQ.size(4);
    auto M_k = cache_K.size(1);

    constexpr auto BLOCK_M = 16;
    auto M_ceil = (M + BLOCK_M - 1) / BLOCK_M * BLOCK_M;

    constexpr auto kThreadsPerWarp = 64;
    constexpr auto kWarpsPerBlock = 2; // original uses 2 warps

    const auto options = at::TensorOptions()
                             .dtype(XQ.dtype())
                             .layout(at::kStrided)
                             .device(XQ.device())
                             .requires_grad(false);

    auto O_splitk = at::empty({B * G * H, split_k, M_ceil, K_q}, options);
    auto metadata = at::empty({B * G * H, 2, split_k, M_ceil}, options);

    dim3 attention_grid = {static_cast<uint32_t>(M / BLOCK_M), static_cast<uint32_t>(B * G * H), static_cast<uint32_t>(split_k)};
    dim3 reduce_grid = {static_cast<uint32_t>(B * G * H), static_cast<uint32_t>(M)};

    dim3 threads = {kThreadsPerWarp * kWarpsPerBlock};
    
    auto O = at::empty_like(XQ);

    return O;
}
}

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::efficient_attention_forward_decoder_splitk_ck"),
      TORCH_FN(efficient_attention_forward_decoder_splitk_ck));
}