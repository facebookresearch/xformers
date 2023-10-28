#pragma once

#include <sstream>
#include <stdexcept>

#include <ck/ck.hpp>
#include <ck/tensor_operation/gpu/device/gemm_specialization.hpp>
#include <ck/tensor_operation/gpu/device/impl/device_grouped_mha_bwd_xdl_cshuffle_qloop_v1.hpp>
#include <ck/tensor_operation/gpu/device/impl/device_grouped_mha_bwd_xdl_cshuffle_qloop_v2.hpp>
#include <ck/tensor_operation/gpu/device/tensor_specialization.hpp>
#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>
#include <ck/utility/math.hpp>
#include <ck/utility/number.hpp>

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
struct grouped_backward_masktype_attnbias_dispatched {
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

#ifndef GROUPED_BACKWARD_V1_HEADDIM_SWITCH
#define GROUPED_BACKWARD_V1_HEADDIM_SWITCH(HEAD_DIM1, HEAD_DIM2, ...) \
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
  using DeviceOpInstanceTemp_V1 = ck::tensor_operation::device::DeviceGroupedMultiheadAttentionBackward_Qloop_Xdl_CShuffle_V1<
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
          GemmOpConstantsGroupedBackward_V1::NumGemmKPrefetchStage,
          GemmOpConstantsGroupedBackward_V1::BlockSize,
          GemmOpConstantsGroupedBackward_V1::MPerBlock,
          GemmOpConstantsGroupedBackward_V1::NPerBlock,
          kGemm1NPerBlock, // KPerBlock = kGemm1NerBlock
          kGemm1NPerBlock,
          GemmOpConstantsGroupedBackward_V1::Gemm1KPerBlock,
          GemmOpConstantsGroupedBackward_V1::Gemm2KPerBlock,
          GemmOpConstantsGroupedBackward_V1::AK1,
          GemmOpConstantsGroupedBackward_V1::BK1,
          GemmOpConstantsGroupedBackward_V1::B1K1,
          GemmOpConstantsGroupedBackward_V1::MPerXDL,
          GemmOpConstantsGroupedBackward_V1::NPerXDL,
          GemmOpConstantsGroupedBackward_V1::MXdlPerWave,
          GemmOpConstantsGroupedBackward_V1::NXdlPerWave,
          kGemm1NXdlPerWave,
          GemmOpConstantsGroupedBackward_V1::Gemm2NXdlPerWave,
          GemmOpConstantsGroupedBackward_V1::ABlockTransferThreadClusterLengths_AK0_M_AK1,
          GemmOpConstantsGroupedBackward_V1::ABlockTransferThreadClusterArrangeOrder,
          GemmOpConstantsGroupedBackward_V1::ABlockTransferSrcAccessOrder,
          GemmOpConstantsGroupedBackward_V1::ABlockTransferSrcVectorDim,
          kABBlockTransferSrcScalarPerVector,
          GemmOpConstantsGroupedBackward_V1::ABlockTransferDstScalarPerVector_AK1,
          GemmOpConstantsGroupedBackward_V1::ABlockLdsExtraM,
          GemmOpConstantsGroupedBackward_V1::BBlockTransferThreadClusterLengths_BK0_N_BK1,
          GemmOpConstantsGroupedBackward_V1::BBlockTransferThreadClusterArrangeOrder,
          GemmOpConstantsGroupedBackward_V1::BBlockTransferSrcAccessOrder,
          GemmOpConstantsGroupedBackward_V1::BBlockTransferSrcVectorDim,
          kABBlockTransferSrcScalarPerVector,
          GemmOpConstantsGroupedBackward_V1::BBlockTransferDstScalarPerVector_BK1,
          GemmOpConstantsGroupedBackward_V1::BBlockLdsExtraN,
          kAcc0BiasTransferSrcScalarPerVector,
          GemmOpConstantsGroupedBackward_V2::CShuffleMXdlPerWavePerShuffle,
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
  using DeviceOpInstanceTemp_V2 = ck::tensor_operation::device::DeviceGroupedMultiheadAttentionBackward_Qloop_Xdl_CShuffle_V2<
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
          GemmOpConstantsGroupedBackward_V2::NumGemmKPrefetchStage,
          GemmOpConstantsGroupedBackward_V2::BlockSize,
          GemmOpConstantsGroupedBackward_V2::MPerBlock,
          GemmOpConstantsGroupedBackward_V2::NPerBlock,
          GemmOpConstantsGroupedBackward_V2::KPerBlock,
          kGemm1NPerBlock,
          GemmOpConstantsGroupedBackward_V2::Gemm1KPerBlock,
          GemmOpConstantsGroupedBackward_V2::Gemm2KPerBlock,
          GemmOpConstantsGroupedBackward_V2::AK1,
          GemmOpConstantsGroupedBackward_V2::BK1,
          GemmOpConstantsGroupedBackward_V2::B1K1,
          GemmOpConstantsGroupedBackward_V2::MPerXDL,
          GemmOpConstantsGroupedBackward_V2::NPerXDL,
          GemmOpConstantsGroupedBackward_V2::MXdlPerWave,
          GemmOpConstantsGroupedBackward_V2::NXdlPerWave,
          kGemm1NXdlPerWave,
          GemmOpConstantsBatchedBackward_V2::Gemm2NXdlPerWave,
          GemmOpConstantsGroupedBackward_V2::ABlockTransferThreadClusterLengths_AK0_M_AK1,
          GemmOpConstantsGroupedBackward_V2::ABlockTransferThreadClusterArrangeOrder,
          GemmOpConstantsGroupedBackward_V2::ABlockTransferSrcAccessOrder,
          GemmOpConstantsGroupedBackward_V2::ABlockTransferSrcVectorDim,
          kABBlockTransferSrcScalarPerVector,
          GemmOpConstantsGroupedBackward_V2::ABlockTransferDstScalarPerVector_AK1,
          GemmOpConstantsGroupedBackward_V2::ABlockLdsExtraM,
          GemmOpConstantsGroupedBackward_V2::BBlockTransferThreadClusterLengths_BK0_N_BK1,
          GemmOpConstantsGroupedBackward_V2::BBlockTransferThreadClusterArrangeOrder,
          GemmOpConstantsGroupedBackward_V2::BBlockTransferSrcAccessOrder,
          GemmOpConstantsGroupedBackward_V2::BBlockTransferSrcVectorDim,
          kABBlockTransferSrcScalarPerVector,
          GemmOpConstantsGroupedBackward_V2::BBlockTransferDstScalarPerVector_BK1,
          GemmOpConstantsGroupedBackward_V2::BBlockLdsExtraN,
          kAcc0BiasTransferSrcScalarPerVector,
          GemmOpConstantsGroupedBackward_V2::B1BlockTransferThreadClusterLengths_BK0_N_BK1,
          GemmOpConstantsGroupedBackward_V2::B1BlockTransferThreadClusterArrangeOrder,
          GemmOpConstantsGroupedBackward_V2::B1BlockTransferSrcAccessOrder,
          GemmOpConstantsGroupedBackward_V2::B1BlockTransferSrcVectorDim,
          kB1BlockTransferSrcScalarPerVector,
          GemmOpConstantsGroupedBackward_V2::B1BlockTransferDstScalarPerVector_BK1,
          GemmOpConstantsGroupedBackward_V2::B1BlockLdsExtraN,
          GemmOpConstantsGroupedBackward_V2::CShuffleMXdlPerWavePerShuffle,
          kCShuffleNXdlPerWavePerShuffle,
          kCShuffleBlockTransferClusterLengths,
          kCShuffleBlockTransferScalarPerVector,
          MaskingSpec,
          Deterministic>;
  // clang-format on

  static constexpr auto I1 = ck::Number<1>{};
  static constexpr auto I2 = ck::Number<2>{};
  static constexpr auto I3 = ck::Number<3>{};

  static void Run(GroupedBackwardParams& param, hipStream_t stream) {
    using ck::math::min;

    if (param.K <= 64 && param.Kv <= 64) {
      // compile-time constants which don't depend on head-dim switching
      constexpr ck::index_t thread_slice_length_ak1 =
          GemmOpConstantsGroupedBackward_V1::AK1 /
          GemmOpConstantsGroupedBackward_V1::
              ABlockTransferThreadClusterLengths_AK0_M_AK1::At(I2);
      constexpr ck::index_t thread_slice_length_bk1 =
          GemmOpConstantsGroupedBackward_V1::BK1 /
          GemmOpConstantsGroupedBackward_V1::
              BBlockTransferThreadClusterLengths_BK0_N_BK1::At(I2);

      static_assert(
          thread_slice_length_ak1 == thread_slice_length_bk1,
          "ABlockTransfer and BBlockTransfer should use completely same K1 sizes and ThreadClusterLengths!");

      constexpr ck::index_t kABBlockTransferSrcScalarPerVector_max =
          min(2, thread_slice_length_ak1);

      GROUPED_BACKWARD_V1_HEADDIM_SWITCH(param.K, param.Kv, [&] {
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
          GemmOpConstantsGroupedBackward_V2::AK1 /
          GemmOpConstantsGroupedBackward_V2::
              ABlockTransferThreadClusterLengths_AK0_M_AK1::At(I2);
      constexpr ck::index_t thread_slice_length_bk1 =
          GemmOpConstantsGroupedBackward_V2::BK1 /
          GemmOpConstantsGroupedBackward_V2::
              BBlockTransferThreadClusterLengths_BK0_N_BK1::At(I2);

      static_assert(
          thread_slice_length_ak1 == thread_slice_length_bk1,
          "ABlockTransfer and BBlockTransfer should use completely same K1 sizes and ThreadClusterLengths!");

      constexpr ck::index_t kABBlockTransferSrcScalarPerVector_max =
          min(2, thread_slice_length_ak1);

      constexpr ck::index_t thread_slice_length_gemm1n = kGemm1NPerBlock /
          GemmOpConstantsGroupedBackward_V2::
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
      GroupedBackwardParams& param,
      hipStream_t stream) {
    // Tunables
    std::vector<typename DeviceOpInstance::ProblemDesc> problem_descs;

    for (std::size_t i = 0; i < param.num_batches; i++) {
      int M =
          param.host_seqstart_q[i + 1] - param.host_seqstart_q[i]; // seqlen Q
      int N = param.host_seqlen_k.empty()
          ? param.host_seqstart_k[i + 1] - param.host_seqstart_k[i]
          : param.host_seqlen_k[i];
      int K = param.K;
      int Kv = param.Kv;
      int G1 = param.num_heads;

      std::vector<ck::index_t> q_gs_ms_ks_lengths{1, G1, M, K};
      std::vector<ck::index_t> q_gs_ms_ks_strides{
          0, param.q_strides[1], param.q_strides[0], param.q_strides[2]};

      std::vector<ck::index_t> k_gs_ns_ks_lengths{1, G1, N, K};
      std::vector<ck::index_t> k_gs_ns_ks_strides{
          0, param.k_strides[1], param.k_strides[0], param.k_strides[2]};

      // ToDo: support multi-query and group-query attention
      std::vector<ck::index_t> kgrad_gs_ns_ks_lengths = k_gs_ns_ks_lengths;
      std::vector<ck::index_t> kgrad_gs_ns_ks_strides = k_gs_ns_ks_strides;

      // to be changed to v_gs_ns_os_lengths
      std::vector<ck::index_t> v_gs_os_ns_lengths{1, G1, Kv, N};
      std::vector<ck::index_t> v_gs_os_ns_strides{
          0, param.v_strides[1], param.v_strides[2], param.v_strides[0]};

      // ToDo: support multi-query and group-query attention
      std::vector<ck::index_t> vgrad_gs_os_ns_lengths = v_gs_os_ns_lengths;
      std::vector<ck::index_t> vgrad_gs_os_ns_strides = v_gs_os_ns_strides;

      std::vector<ck::index_t> y_gs_ms_os_lengths{1, G1, M, Kv};
      std::vector<ck::index_t> y_gs_ms_os_strides{
          0, param.out_strides[1], param.out_strides[0], param.out_strides[2]};

      std::vector<ck::index_t> lse_gs_ms_lengths{1, G1, M};
      std::vector<ck::index_t> lse_gs_ms_strides{0, param.max_seqlen_q, 1};

      std::vector<ck::index_t> d_gs_ms_ns_lengths;
      std::vector<ck::index_t> d_gs_ms_ns_strides;

      if constexpr (has_attn_bias) {
        d_gs_ms_ns_lengths = {1, G1, M, N};
        d_gs_ms_ns_strides = {
            0,
            param.attn_bias_strides[0],
            param.attn_bias_strides[1],
            param.attn_bias_strides[2]};

      } else {
        d_gs_ms_ns_lengths = {1, 1, 1, 1};
        d_gs_ms_ns_strides = {0, 0, 0, 0};
      };

      problem_descs.push_back({
          q_gs_ms_ks_lengths, // q, dQ should have same shape
          q_gs_ms_ks_strides,
          k_gs_ns_ks_lengths, // k, dK should have same shape
          k_gs_ns_ks_strides,
          {1, 1, 1, 1},
          {0, 0, 0, 0},
          v_gs_os_ns_lengths, // v, dV should have same shape
          v_gs_os_ns_strides,
          y_gs_ms_os_lengths, // y, dY should have same shape
          y_gs_ms_os_strides,
          lse_gs_ms_lengths,
          lse_gs_ms_strides,
          kgrad_gs_ns_ks_lengths,
          kgrad_gs_ns_ks_strides,
          vgrad_gs_os_ns_lengths,
          vgrad_gs_os_ns_strides,
          d_gs_ms_ns_lengths, // bias, grad_bias should have same shape
          d_gs_ms_ns_strides,
          {}, // acc1_biases_gs_ms_os_lengths
          {}, // acc1_biases_gs_ms_os_strides
      });
    }

    float alpha = param.scale;

    auto op = DeviceOpInstance{};
    auto invoker = op.MakeInvoker();

    auto arg_ptr = op.MakeArgumentPointer(
        param.q_ptrs,
        param.k_ptrs,
        param.randvals_ptrs,
        param.v_ptrs,
        param.out_ptrs,
        param.logsumexp_ptrs,
        param.grad_out_ptrs,
        param.grad_q_ptrs,
        param.grad_k_ptrs,
        param.grad_v_ptrs,
        param.attn_bias_ptrs,
        {}, // p_acc1_bias_vec;
        param.grad_bias_ptrs,
        {},
        problem_descs,
        QKVElementOp{},
        QKVElementOp{},
        Scale{alpha},
        QKVElementOp{},
        YElementOp{},
        param.dropout_prob,
        std::tuple<int64_t, int64_t>(param.philox_seed, param.philox_offset));

    SimpleDeviceMem workspace(op.GetWorkSpaceSize(arg_ptr.get()));

    op.SetWorkSpacePointer(arg_ptr.get(), workspace.GetDeviceBuffer());

    if (!op.IsSupportedArgument(arg_ptr.get())) {
      std::ostringstream ostr;
      ostr << op.GetTypeString() << " does not support this problem";

      throw std::runtime_error(ostr.str());
    }

    (void)invoker.Run(arg_ptr.get(), StreamConfig{stream, false});
  };
};
