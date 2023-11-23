#pragma once

#include <sstream>
#include <stdexcept>

#include <ck/ck.hpp>
#include <ck/tensor_operation/gpu/device/gemm_specialization.hpp>
#include <ck/tensor_operation/gpu/device/impl/device_batched_mha_bwd_xdl_cshuffle_qloop_v1.hpp>
#include <ck/tensor_operation/gpu/device/impl/device_batched_mha_bwd_xdl_cshuffle_qloop_v2.hpp>
#include <ck/tensor_operation/gpu/device/tensor_specialization.hpp>
#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>

#include "ck_align_switch.h"
#include "ck_fmha_backward_gemm_constants.h"
#include "ck_fmha_common_gemm_constants.h"
#include "ck_fmha_op_helper.h"
#include "ck_fmha_params.h"

template <
    typename scalar_t,
    int32_t custom_mask_type,
    bool has_attn_bias,
    bool use_fp32_qkv_grad>
struct batched_backward_masktype_attnbias_dispatched {
  using PassThrough = ck::tensor_operation::element_wise::PassThrough;
  using Scale = ck::tensor_operation::element_wise::Scale;

  using QKVElementOp = PassThrough;
  using YElementOp = PassThrough;

  using InputDataType = scalar_t;
  using OutputDataType =
      typename std::conditional<use_fp32_qkv_grad, F32, scalar_t>::type;
  using GemmDataType = scalar_t;
  using AccDataType = F32;
  using ShuffleDataType = F32;
  using LSEDataType = F32;
  using ZDataType = unsigned short;
  using Acc0BiasDataType =
      typename std::conditional<has_attn_bias, scalar_t, void>::type;
  using Acc1BiasDataType = void;

  static constexpr auto GemmSpec =
      ck::tensor_operation::device::GemmSpecialization::MNKOPadding;
  static constexpr auto MaskingSpec =
      static_cast<ck::tensor_operation::device::MaskingSpecialization>(
          custom_mask_type);

  static constexpr bool Deterministic = true;

  static constexpr ck::index_t kAcc0BiasTransferSrcScalarPerVector = 1;

#ifndef BATCHED_BACKWARD_V1_HEADDIM_SWITCH
#define BATCHED_BACKWARD_V1_HEADDIM_SWITCH(HEAD_DIM1, HEAD_DIM2, ...) \
  [&] {                                                               \
    if (HEAD_DIM1 <= 32 && HEAD_DIM2 <= 32) {                         \
      constexpr ck::index_t kGemm1NPerBlock = 32;                     \
      constexpr ck::index_t kGemm1NXdlPerWave = 1;                    \
      constexpr ck::index_t kCShuffleNXdlPerWavePerShuffle = 1;       \
      using kCShuffleBlockTransferClusterLengths = S<1, 64, 1, 4>;    \
      __VA_ARGS__();                                                  \
    } else {                                                          \
      constexpr ck::index_t kGemm1NPerBlock = 64;                     \
      constexpr ck::index_t kGemm1NXdlPerWave = 2;                    \
      constexpr ck::index_t kCShuffleNXdlPerWavePerShuffle = 2;       \
      using kCShuffleBlockTransferClusterLengths = S<1, 32, 1, 8>;    \
      __VA_ARGS__();                                                  \
    };                                                                \
  }()
#endif

  // clang-format off
  template <
      ck::index_t kGemm1NPerBlock,
      ck::index_t kGemm1NXdlPerWave,
      ck::index_t kCShuffleNXdlPerWavePerShuffle,
      typename kCShuffleBlockTransferClusterLengths,
      ck::index_t kABBlockTransferSrcScalarPerVector,
      ck::index_t kCShuffleBlockTransferScalarPerVector>
  using DeviceOpInstanceTemp_V1 = ck::tensor_operation::device::DeviceBatchedMultiheadAttentionBackward_Qloop_Xdl_CShuffle_V1<
          GemmOpConstantsCommon::NumDimG,
          GemmOpConstantsCommon::NumDimM,
          GemmOpConstantsCommon::NumDimN,
          GemmOpConstantsCommon::NumDimK,
          GemmOpConstantsCommon::NumDimO,
          InputDataType,
          OutputDataType,
          GemmDataType,
          ZDataType,
          LSEDataType,
          Acc0BiasDataType,
          Acc1BiasDataType,
          AccDataType,
          ShuffleDataType,
          QKVElementOp,
          QKVElementOp,
          Scale,
          QKVElementOp,
          YElementOp,
          GemmSpec,
          GemmOpConstantsCommon::TensorSpecA,
          GemmOpConstantsCommon::TensorSpecB0,
          GemmOpConstantsCommon::TensorSpecB1,
          GemmOpConstantsCommon::TensorSpecC,
          GemmOpConstantsBatchedBackward_V1::NumGemmKPrefetchStage,
          GemmOpConstantsBatchedBackward_V1::BlockSize,
          GemmOpConstantsBatchedBackward_V1::MPerBlock,
          GemmOpConstantsBatchedBackward_V1::NPerBlock,
          kGemm1NPerBlock, // KPerBlock == kGemm1NPerBlock required
          kGemm1NPerBlock,
          GemmOpConstantsBatchedBackward_V1::Gemm1KPerBlock,
          GemmOpConstantsBatchedBackward_V1::Gemm2KPerBlock,
          GemmOpConstantsBatchedBackward_V1::AK1,
          GemmOpConstantsBatchedBackward_V1::BK1,
          GemmOpConstantsBatchedBackward_V1::B1K1,
          GemmOpConstantsBatchedBackward_V1::MPerXDL,
          GemmOpConstantsBatchedBackward_V1::NPerXDL,
          GemmOpConstantsBatchedBackward_V1::MXdlPerWave,
          GemmOpConstantsBatchedBackward_V1::NXdlPerWave,
          kGemm1NXdlPerWave,
          GemmOpConstantsBatchedBackward_V1::Gemm2NXdlPerWave,
          GemmOpConstantsBatchedBackward_V1::ABlockTransferThreadClusterLengths_AK0_M_AK1,
          GemmOpConstantsBatchedBackward_V1::ABlockTransferThreadClusterArrangeOrder,
          GemmOpConstantsBatchedBackward_V1::ABlockTransferSrcAccessOrder,
          GemmOpConstantsBatchedBackward_V1::ABlockTransferSrcVectorDim,
          kABBlockTransferSrcScalarPerVector,
          GemmOpConstantsBatchedBackward_V1::ABlockTransferDstScalarPerVector_AK1,
          GemmOpConstantsBatchedBackward_V1::ABlockLdsExtraM,
          GemmOpConstantsBatchedBackward_V1::BBlockTransferThreadClusterLengths_BK0_N_BK1,
          GemmOpConstantsBatchedBackward_V1::BBlockTransferThreadClusterArrangeOrder,
          GemmOpConstantsBatchedBackward_V1::BBlockTransferSrcAccessOrder,
          GemmOpConstantsBatchedBackward_V1::BBlockTransferSrcVectorDim,
          kABBlockTransferSrcScalarPerVector,
          GemmOpConstantsBatchedBackward_V1::BBlockTransferDstScalarPerVector_BK1,
          GemmOpConstantsBatchedBackward_V1::BBlockLdsExtraN,
          kAcc0BiasTransferSrcScalarPerVector,
          GemmOpConstantsBatchedBackward_V1::CShuffleMXdlPerWavePerShuffle,
          kCShuffleNXdlPerWavePerShuffle,
          kCShuffleBlockTransferClusterLengths,
          kCShuffleBlockTransferScalarPerVector,
          MaskingSpec,
          Deterministic>;
  // clang-format on

  // clang-format off
  template <
      ck::index_t kGemm1NPerBlock,
      ck::index_t kGemm1NXdlPerWave,
      ck::index_t kCShuffleNXdlPerWavePerShuffle,
      typename kCShuffleBlockTransferClusterLengths,
      ck::index_t kABBlockTransferSrcScalarPerVector,
      ck::index_t kB1BlockTransferSrcScalarPerVector,
      ck::index_t kCShuffleBlockTransferScalarPerVector>
  using DeviceOpInstanceTemp_V2 = ck::tensor_operation::device::DeviceBatchedMultiheadAttentionBackward_Qloop_Xdl_CShuffle_V2<
          GemmOpConstantsCommon::NumDimG,
          GemmOpConstantsCommon::NumDimM,
          GemmOpConstantsCommon::NumDimN,
          GemmOpConstantsCommon::NumDimK,
          GemmOpConstantsCommon::NumDimO,
          InputDataType,
          OutputDataType,
          GemmDataType,
          ZDataType,
          LSEDataType,
          Acc0BiasDataType,
          Acc1BiasDataType,
          AccDataType,
          ShuffleDataType,
          QKVElementOp,
          QKVElementOp,
          Scale,
          QKVElementOp,
          YElementOp,
          GemmSpec,
          GemmOpConstantsCommon::TensorSpecA,
          GemmOpConstantsCommon::TensorSpecB0,
          GemmOpConstantsCommon::TensorSpecB1,
          GemmOpConstantsCommon::TensorSpecC,
          GemmOpConstantsBatchedBackward_V2::NumGemmKPrefetchStage,
          GemmOpConstantsBatchedBackward_V2::BlockSize,
          GemmOpConstantsBatchedBackward_V2::MPerBlock,
          GemmOpConstantsBatchedBackward_V2::NPerBlock,
          GemmOpConstantsBatchedBackward_V2::KPerBlock,
          kGemm1NPerBlock,
          GemmOpConstantsBatchedBackward_V2::Gemm1KPerBlock,
          GemmOpConstantsBatchedBackward_V2::Gemm2KPerBlock,
          GemmOpConstantsBatchedBackward_V2::AK1,
          GemmOpConstantsBatchedBackward_V2::BK1,
          GemmOpConstantsBatchedBackward_V2::B1K1,
          GemmOpConstantsBatchedBackward_V2::MPerXDL,
          GemmOpConstantsBatchedBackward_V2::NPerXDL,
          GemmOpConstantsBatchedBackward_V2::MXdlPerWave,
          GemmOpConstantsBatchedBackward_V2::NXdlPerWave,
          kGemm1NXdlPerWave,
          GemmOpConstantsBatchedBackward_V2::Gemm2NXdlPerWave,
          GemmOpConstantsBatchedBackward_V2::ABlockTransferThreadClusterLengths_AK0_M_AK1,
          GemmOpConstantsBatchedBackward_V2::ABlockTransferThreadClusterArrangeOrder,
          GemmOpConstantsBatchedBackward_V2::ABlockTransferSrcAccessOrder,
          GemmOpConstantsBatchedBackward_V2::ABlockTransferSrcVectorDim,
          kABBlockTransferSrcScalarPerVector,
          GemmOpConstantsBatchedBackward_V2::ABlockTransferDstScalarPerVector_AK1,
          GemmOpConstantsBatchedBackward_V2::ABlockLdsExtraM,
          GemmOpConstantsBatchedBackward_V2::BBlockTransferThreadClusterLengths_BK0_N_BK1,
          GemmOpConstantsBatchedBackward_V2::BBlockTransferThreadClusterArrangeOrder,
          GemmOpConstantsBatchedBackward_V2::BBlockTransferSrcAccessOrder,
          GemmOpConstantsBatchedBackward_V2::BBlockTransferSrcVectorDim,
          kABBlockTransferSrcScalarPerVector,
          GemmOpConstantsBatchedBackward_V2::BBlockTransferDstScalarPerVector_BK1,
          GemmOpConstantsBatchedBackward_V2::BBlockLdsExtraN,
          kAcc0BiasTransferSrcScalarPerVector,
          GemmOpConstantsBatchedBackward_V2::B1BlockTransferThreadClusterLengths_BK0_N_BK1,
          GemmOpConstantsBatchedBackward_V2::B1BlockTransferThreadClusterArrangeOrder,
          GemmOpConstantsBatchedBackward_V2::B1BlockTransferSrcAccessOrder,
          GemmOpConstantsBatchedBackward_V2::B1BlockTransferSrcVectorDim,
          kB1BlockTransferSrcScalarPerVector,
          GemmOpConstantsBatchedBackward_V2::B1BlockTransferDstScalarPerVector_BK1,
          GemmOpConstantsBatchedBackward_V2::B1BlockLdsExtraN,
          GemmOpConstantsBatchedBackward_V2::CShuffleMXdlPerWavePerShuffle,
          kCShuffleNXdlPerWavePerShuffle,
          kCShuffleBlockTransferClusterLengths,
          kCShuffleBlockTransferScalarPerVector,
          MaskingSpec,
          Deterministic>;
  // clang-format on

  static constexpr auto I1 = ck::Number<1>{};
  static constexpr auto I2 = ck::Number<2>{};
  static constexpr auto I3 = ck::Number<3>{};

  static void Run(BatchedBackwardParams& param, hipStream_t stream) {
    using ck::math::min;

    if (param.K <= 64 && param.Kv <= 64) {
      // compile-time constants which don't depend on head-dim switching
      constexpr ck::index_t thread_slice_length_ak1 =
          GemmOpConstantsBatchedBackward_V1::AK1 /
          GemmOpConstantsBatchedBackward_V1::
              ABlockTransferThreadClusterLengths_AK0_M_AK1::At(I2);
      constexpr ck::index_t thread_slice_length_bk1 =
          GemmOpConstantsBatchedBackward_V1::BK1 /
          GemmOpConstantsBatchedBackward_V1::
              BBlockTransferThreadClusterLengths_BK0_N_BK1::At(I2);

      static_assert(
          thread_slice_length_ak1 == thread_slice_length_bk1,
          "ABlockTransfer and BBlockTransfer should use completely same K1 sizes and ThreadClusterLengths!");

      constexpr ck::index_t kABBlockTransferSrcScalarPerVector_max =
          min(2, thread_slice_length_ak1);

      BATCHED_BACKWARD_V1_HEADDIM_SWITCH(param.K, param.Kv, [&] {
        constexpr ck::index_t thread_slice_length_cshuflle_n =
            (kCShuffleNXdlPerWavePerShuffle * kGemm1NPerBlock /
             kGemm1NXdlPerWave) /
            kCShuffleBlockTransferClusterLengths::At(I3);

        constexpr ck::index_t kCShuffleBlockTransferScalarPerVector_max =
            min(2, thread_slice_length_cshuflle_n);

        ALIGN_SWITCH_2(
            kABBlockTransferSrcScalarPerVector_max,
            kABBlockTransferSrcScalarPerVector,
            param.K,
            kCShuffleBlockTransferScalarPerVector_max,
            kCShuffleBlockTransferScalarPerVector,
            param.Kv,
            [&] {
              using DeviceOpInstance = DeviceOpInstanceTemp_V1<
                  kGemm1NPerBlock,
                  kGemm1NXdlPerWave,
                  kCShuffleNXdlPerWavePerShuffle,
                  kCShuffleBlockTransferClusterLengths,
                  kABBlockTransferSrcScalarPerVector,
                  kCShuffleBlockTransferScalarPerVector>;

              RunWithDeviceOp<DeviceOpInstance>(param, stream);
            });
      });
    } else {
      constexpr ck::index_t kGemm1NPerBlock = 128;
      constexpr ck::index_t kGemm1NXdlPerWave = 4;
      constexpr ck::index_t kCShuffleNXdlPerWavePerShuffle = 4;
      using kCShuffleBlockTransferClusterLengths = S<1, 32, 1, 8>;

      constexpr ck::index_t thread_slice_length_ak1 =
          GemmOpConstantsBatchedBackward_V2::AK1 /
          GemmOpConstantsBatchedBackward_V2::
              ABlockTransferThreadClusterLengths_AK0_M_AK1::At(I2);
      constexpr ck::index_t thread_slice_length_bk1 =
          GemmOpConstantsBatchedBackward_V2::BK1 /
          GemmOpConstantsBatchedBackward_V2::
              BBlockTransferThreadClusterLengths_BK0_N_BK1::At(I2);

      static_assert(
          thread_slice_length_ak1 == thread_slice_length_bk1,
          "ABlockTransfer and BBlockTransfer should use completely same K1 sizes and ThreadClusterLengths!");

      constexpr ck::index_t kABBlockTransferSrcScalarPerVector_max =
          min(2, thread_slice_length_ak1);

      constexpr ck::index_t thread_slice_length_gemm1n = kGemm1NPerBlock /
          GemmOpConstantsBatchedBackward_V2::
              B1BlockTransferThreadClusterLengths_BK0_N_BK1::At(I1);
      constexpr ck::index_t kB1BlockTransferSrcScalarPerVector_max =
          min(2, thread_slice_length_gemm1n);

      constexpr ck::index_t thread_slice_length_cshuflle_n =
          (kCShuffleNXdlPerWavePerShuffle * kGemm1NPerBlock /
           kGemm1NXdlPerWave) /
          kCShuffleBlockTransferClusterLengths::At(I3);

      constexpr ck::index_t kCShuffleBlockTransferScalarPerVector_max =
          min(2, thread_slice_length_cshuflle_n);

      if constexpr (
          kB1BlockTransferSrcScalarPerVector_max >=
          kCShuffleBlockTransferScalarPerVector_max) {
        ALIGN_SWITCH_2(
            kABBlockTransferSrcScalarPerVector_max,
            kABBlockTransferSrcScalarPerVector,
            param.K,
            kB1BlockTransferSrcScalarPerVector_max,
            kB1BlockTransferSrcScalarPerVector,
            param.Kv,
            [&] {
              constexpr ck::index_t kCShuffleBlockTransferScalarPerVector =
                  min(kB1BlockTransferSrcScalarPerVector,
                      kCShuffleBlockTransferScalarPerVector_max);

              static_assert(
                  kB1BlockTransferSrcScalarPerVector > 0,
                  "kB1BlockTransferSrcScalarPerVector must be positive");

              using DeviceOpInstance = DeviceOpInstanceTemp_V2<
                  kGemm1NPerBlock,
                  kGemm1NXdlPerWave,
                  kCShuffleNXdlPerWavePerShuffle,
                  kCShuffleBlockTransferClusterLengths,
                  kABBlockTransferSrcScalarPerVector,
                  kB1BlockTransferSrcScalarPerVector,
                  kCShuffleBlockTransferScalarPerVector>;

              RunWithDeviceOp<DeviceOpInstance>(param, stream);
            });
      } else {
        ALIGN_SWITCH_2(
            kABBlockTransferSrcScalarPerVector_max,
            kABBlockTransferSrcScalarPerVector,
            param.K,
            kCShuffleBlockTransferScalarPerVector_max,
            kCShuffleBlockTransferScalarPerVector,
            param.Kv,
            [&] {
              constexpr ck::index_t kB1BlockTransferSrcScalarPerVector =
                  min(kCShuffleBlockTransferScalarPerVector,
                      kB1BlockTransferSrcScalarPerVector_max);

              static_assert(
                  kB1BlockTransferSrcScalarPerVector > 0,
                  "kB1BlockTransferSrcScalarPerVector must be positive");

              using DeviceOpInstance = DeviceOpInstanceTemp_V2<
                  kGemm1NPerBlock,
                  kGemm1NXdlPerWave,
                  kCShuffleNXdlPerWavePerShuffle,
                  kCShuffleBlockTransferClusterLengths,
                  kABBlockTransferSrcScalarPerVector,
                  kB1BlockTransferSrcScalarPerVector,
                  kCShuffleBlockTransferScalarPerVector>;

              RunWithDeviceOp<DeviceOpInstance>(param, stream);
            });
      };
    };
  };

  template <typename DeviceOpInstance>
  static void RunWithDeviceOp(
      BatchedBackwardParams& param,
      hipStream_t stream) {
    std::vector<ck::index_t> q_gs_ms_ks_lengths{
        param.B, param.Hq, param.M, param.K};
    std::vector<ck::index_t> q_gs_ms_ks_strides{
        param.q_strides[0],
        param.q_strides[2],
        param.q_strides[1],
        param.q_strides[3]};

    std::vector<ck::index_t> k_gs_ns_ks_lengths{
        param.B, param.Hkv, param.N, param.K};
    std::vector<ck::index_t> k_gs_ns_ks_strides{
        param.k_strides[0],
        param.k_strides[2],
        param.k_strides[1],
        param.k_strides[3]};

    std::vector<ck::index_t> kgrad_gs_ns_ks_lengths = {
        param.B, param.Hq, param.N, param.K};
    std::vector<ck::index_t> kgrad_gs_ns_ks_strides = {
        param.tmp_grad_k_strides[0],
        param.tmp_grad_k_strides[2],
        param.tmp_grad_k_strides[1],
        param.tmp_grad_k_strides[3]};

    std::vector<ck::index_t> v_gs_os_ns_lengths{
        param.B, param.Hkv, param.Kv, param.N};
    std::vector<ck::index_t> v_gs_os_ns_strides{
        param.v_strides[0],
        param.v_strides[2],
        param.v_strides[3],
        param.v_strides[1]};

    std::vector<ck::index_t> vgrad_gs_os_ns_lengths = {
        param.B, param.Hq, param.Kv, param.N};
    std::vector<ck::index_t> vgrad_gs_os_ns_strides = {
        param.tmp_grad_v_strides[0],
        param.tmp_grad_v_strides[2],
        param.tmp_grad_v_strides[3],
        param.tmp_grad_v_strides[1]};

    std::vector<ck::index_t> y_gs_ms_os_lengths{
        param.B, param.Hq, param.M, param.Kv};
    std::vector<ck::index_t> y_gs_ms_os_strides{
        param.out_strides[0],
        param.out_strides[2],
        param.out_strides[1],
        param.out_strides[3]};

    std::vector<ck::index_t> lse_gs_ms_lengths{param.B, param.Hq, param.M};

    std::vector<ck::index_t> d_gs_ms_ns_lengths;
    std::vector<ck::index_t> d_gs_ms_ns_strides;

    if constexpr (has_attn_bias) {
      d_gs_ms_ns_lengths = {param.B, param.Hq, param.M, param.N};
      d_gs_ms_ns_strides = {
          param.attn_bias_strides[0],
          param.attn_bias_strides[1],
          param.attn_bias_strides[2],
          param.attn_bias_strides[3]};
    } else {
      d_gs_ms_ns_lengths = {1, 1, 1, 1};
      d_gs_ms_ns_strides = {0, 0, 0, 0};
    };

    float alpha = param.scale;

    auto op = DeviceOpInstance{};
    auto invoker = op.MakeInvoker();

    auto arg_ptr = op.MakeArgumentPointer(
        param.q_ptr,
        param.k_ptr,
        nullptr, // p_z_grid
        param.v_ptr,
        param.out_ptr,
        param.logsumexp_ptr,
        param.grad_out_ptr,
        param.grad_q_ptr,
        param.grad_k_ptr,
        param.grad_v_ptr,
        param.has_attn_bias ? param.attn_bias_ptr : nullptr,
        nullptr, // p_acc1_bias
        param.bias_has_grad ? param.grad_bias_ptr : nullptr,
        nullptr,
        q_gs_ms_ks_lengths, // q, dQ should have same shape
        q_gs_ms_ks_strides,
        k_gs_ns_ks_lengths, // k, dK should have same shape
        k_gs_ns_ks_strides,
        {1, 1, 1, 1}, // z_gs_ms_ns_lengths
        {0, 0, 0, 0}, // z_gs_ms_ns_strides
        v_gs_os_ns_lengths, // v, dV should have same shape
        v_gs_os_ns_strides,
        y_gs_ms_os_lengths, // y, dY should have same shape
        y_gs_ms_os_strides,
        lse_gs_ms_lengths,
        param.is_mqa_gqa ? kgrad_gs_ns_ks_lengths : k_gs_ns_ks_lengths,
        param.is_mqa_gqa ? kgrad_gs_ns_ks_strides : k_gs_ns_ks_strides,
        param.is_mqa_gqa ? vgrad_gs_os_ns_lengths : v_gs_os_ns_lengths,
        param.is_mqa_gqa ? vgrad_gs_os_ns_strides : v_gs_os_ns_strides,
        d_gs_ms_ns_lengths, // bias, grad_bias should have same shape
        d_gs_ms_ns_strides,
        {}, // acc1_biases_gs_ms_os_lengths
        {}, // acc1_biases_gs_ms_os_strides
        QKVElementOp{},
        QKVElementOp{},
        Scale{alpha},
        QKVElementOp{},
        YElementOp{},
        param.dropout_prob,
        std::tuple<int64_t, int64_t>(param.philox_seed, param.philox_offset));

    if (!op.IsSupportedArgument(arg_ptr.get())) {
      std::ostringstream ostr;
      ostr << op.GetTypeString() << " does not support this problem";

      throw std::runtime_error(ostr.str());
    }

    (void)invoker.Run(arg_ptr.get(), StreamConfig{stream, false});
  };
};

template <
    typename scalar_t,
    int32_t custom_mask_type,
    bool has_attn_bias,
    bool use_fp32_qkv_grad>
void run_batched_backward_masktype_attnbias_dispatched(
    BatchedBackwardParams& param,
    hipStream_t stream) {
  batched_backward_masktype_attnbias_dispatched<
      scalar_t,
      custom_mask_type,
      has_attn_bias,
      use_fp32_qkv_grad>::Run(param, stream);
};
