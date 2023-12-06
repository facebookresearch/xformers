/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/core/TensorOptions.h>
#include <torch/library.h>
#include <torch/types.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>

#include <ck/ck.hpp>
#include <ck/tensor_operation/gpu/device/gemm_specialization.hpp>
#include <ck/tensor_operation/gpu/device/tensor_specialization.hpp>
#include "ck/tensor_operation/gpu/device/impl/device_batched_dropout.hpp"

#include "ck_fmha_util.h"

namespace {

/**
 * generate a tensor with random uniform values. only used for testing, not much
 * attention is paid to performance
 */
at::Tensor
rand_uniform_int(double dropout_prob,
                 const at::Tensor& out_pattern) // [Batches, num_head, query_len, key_len]
{
    int B         = out_pattern.size(0);
    int num_heads = out_pattern.size(1);
    int M         = out_pattern.size(2);
    int N         = out_pattern.size(3);

    // at::cuda::CUDAGuard device_guard(out_pattern.device());
    hipStream_t stream = at::cuda::getCurrentHIPStream().stream();

    at::CUDAGeneratorImpl* gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());

    at::PhiloxCudaState rng_engine_inputs;
    {
        std::lock_guard<std::mutex> lock(gen->mutex_);
        rng_engine_inputs = gen->philox_cuda_state(B * num_heads * M * N);
    }

    const auto seeds = at::cuda::philox::unpack(rng_engine_inputs);

    int64_t philox_seed   = std::get<0>(seeds);
    int64_t philox_offset = std::get<1>(seeds);

    at::Tensor randvals;

    randvals = at::empty({B, num_heads, M, N}, out_pattern.options().dtype(at::ScalarType::Int));

    static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKOPadding;

    static constexpr auto TensorSpecA = ck::tensor_operation::device::TensorSpecialization::Default;
    static constexpr auto TensorSpecB0 =
        ck::tensor_operation::device::TensorSpecialization::Default;
    static constexpr auto TensorSpecB1 =
        ck::tensor_operation::device::TensorSpecialization::Default;
    static constexpr auto TensorSpecC = ck::tensor_operation::device::TensorSpecialization::Default;

    using DeviceOpInstance = ck::tensor_operation::device::DeviceBatchedDropout<2, // NumDimG
                                                                                ck::half_t,
                                                                                int,
                                                                                ck::half_t,
                                                                                GemmSpec,
                                                                                TensorSpecA,
                                                                                TensorSpecB0,
                                                                                TensorSpecB1,
                                                                                TensorSpecC,
                                                                                256, // BlockSize
                                                                                64,  // MPerBlock
                                                                                128, // NPerBlock
                                                                                32,  // KPerBlock
                                                                                8,   // AK1
                                                                                8,   // BK1
                                                                                32,  // MPerXDL
                                                                                32,  // NPerXDL
                                                                                2,   // MXdlPerWave
                                                                                1>;  // NXdlPerWave

    const uint64_t seed   = 1;
    const uint64_t offset = 0;

    std::vector<ck::index_t> z_gs_ms_ns_lengths = {B, num_heads, M, N};
    std::vector<ck::index_t> z_gs_ms_ns_strides = {static_cast<int>(randvals.stride(0)),
                                                   static_cast<int>(randvals.stride(1)),
                                                   static_cast<int>(randvals.stride(2)),
                                                   static_cast<int>(randvals.stride(3))};

    auto dropout_op      = DeviceOpInstance();
    auto dropout_invoker = dropout_op.MakeInvoker();

    auto dropout_arg = dropout_op.MakeArgument(static_cast<int*>(randvals.data_ptr()),
                                               z_gs_ms_ns_lengths,
                                               z_gs_ms_ns_strides,
                                               {philox_seed, philox_offset});

    dropout_invoker.Run(dropout_arg, StreamConfig{stream, false});
    (void)hipStreamSynchronize(stream);

    return randvals;
} // namespace

} // namespace

TORCH_LIBRARY_IMPL(xformers, CUDA, m)
{
    m.impl(TORCH_SELECTIVE_NAME("xformers::_ck_rand_uniform"), TORCH_FN(rand_uniform_int));
}
