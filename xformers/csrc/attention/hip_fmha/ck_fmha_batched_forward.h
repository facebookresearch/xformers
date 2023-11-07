#pragma once

#include <sstream>
#include <stdexcept>

#include <ck/ck.hpp>
#include <ck/tensor_operation/gpu/device/gemm_specialization.hpp>
#include <ck/tensor_operation/gpu/device/impl/device_batched_mha_fwd_xdl_cshuffle_v2.hpp>
#include <ck/tensor_operation/gpu/device/tensor_specialization.hpp>
#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>
#include <ck/utility/math.hpp>
#include <ck/utility/number.hpp>

#include "ck_align_switch.h"
#include "ck_fmha_common_gemm_constants.h"
#include "ck_fmha_forward_gemm_constants.h"
#include "ck_fmha_op_helper.h"
#include "ck_fmha_params.h"

template <typename scalar_t, int32_t custom_mask_type, bool has_attn_bias>
struct batched_forward_masktype_attnbias_dispatched {
  using PassThrough = ck::tensor_operation::element_wise::PassThrough;

  using GemmDataType = scalar_t;
  using ADataType = scalar_t;
  using B0DataType = scalar_t;
  using B1DataType = scalar_t;
  using AccDataType = F32;
  using CShuffleDataType = F32;
  using CDataType = scalar_t;
  using ZDataType = unsigned short;
  using LSEDataType = F32;
  using Acc0BiasDataType =
      typename std::conditional<has_attn_bias, scalar_t, void>::type;
  using Acc1BiasDataType = void;

  static constexpr ck::index_t NumDimG = 2;
  static constexpr ck::index_t NumDimM = 1;
  static constexpr ck::index_t NumDimN = 1;
  static constexpr ck::index_t NumDimK = 1;
  static constexpr ck::index_t NumDimO = 1;

  using AElementOp = PassThrough;
  using B0ElementOp = PassThrough;
  using Acc0ElementOp = ck::tensor_operation::element_wise::Scale;
  using B1ElementOp = PassThrough;
  using CElementOp = PassThrough;

  static constexpr auto GemmSpec =
      ck::tensor_operation::device::GemmSpecialization::MNKOPadding;
  static constexpr auto MaskingSpec =
      static_cast<ck::tensor_operation::device::MaskingSpecialization>(
          custom_mask_type);

  static constexpr ck::index_t kAcc0BiasTransferSrcScalarPerVector = 1;

#ifndef BATCHED_FORWARD_HEADDIM_SWITCH
#define BATCHED_FORWARD_HEADDIM_SWITCH(HEAD_DIM1, HEAD_DIM2, ...) \
  [&] {                                                           \
    if (HEAD_DIM1 <= 32 && HEAD_DIM2 <= 32) {                     \
      constexpr ck::index_t kGemm1NPerBlock = 32;                 \
      constexpr ck::index_t kGemm1NXdlPerWave = 1;                \
      constexpr ck::index_t kCShuffleNXdlPerWavePerShuffle = 1;   \
      __VA_ARGS__();                                              \
    } else if (HEAD_DIM1 <= 64 && HEAD_DIM2 <= 64) {              \
      constexpr ck::index_t kGemm1NPerBlock = 64;                 \
      constexpr ck::index_t kGemm1NXdlPerWave = 2;                \
      constexpr ck::index_t kCShuffleNXdlPerWavePerShuffle = 2;   \
      __VA_ARGS__();                                              \
    } else {                                                      \
      constexpr ck::index_t kGemm1NPerBlock = 128;                \
      constexpr ck::index_t kGemm1NXdlPerWave = 4;                \
      constexpr ck::index_t kCShuffleNXdlPerWavePerShuffle = 4;   \
      __VA_ARGS__();                                              \
    }                                                             \
  }()
#endif

  // clang-format off
  template <
      ck::index_t kGemm1NPerBlock,
      ck::index_t kGemm1NXdlPerWave,
      ck::index_t kCShuffleNXdlPerWavePerShuffle,
      ck::index_t kABBlockTransferSrcScalarPerVector,
      ck::index_t kB1BlockTransferSrcScalarPerVector,
      ck::index_t kCShuffleBlockTransferScalarPerVector>
  using DeviceOpInstanceTemp = ck::tensor_operation::device::DeviceBatchedMultiheadAttentionForward_Xdl_CShuffle_V2<
          GemmOpConstantsCommon::NumDimG,
          GemmOpConstantsCommon::NumDimM,
          GemmOpConstantsCommon::NumDimN,
          GemmOpConstantsCommon::NumDimK,
          GemmOpConstantsCommon::NumDimO,
          ADataType,
          B0DataType,
          B1DataType,
          CDataType,
          GemmDataType,
          ZDataType,
          LSEDataType,
          Acc0BiasDataType,
          Acc1BiasDataType,
          AccDataType,
          CShuffleDataType,
          AElementOp,
          B0ElementOp,
          Acc0ElementOp,
          B1ElementOp,
          CElementOp,
          GemmSpec,
          GemmOpConstantsCommon::TensorSpecA,
          GemmOpConstantsCommon::TensorSpecB0,
          GemmOpConstantsCommon::TensorSpecB1,
          GemmOpConstantsCommon::TensorSpecC,
          GemmOpConstantsBatchedForward::NumGemmKPrefetchStage,
          GemmOpConstantsBatchedForward::BlockSize,
          GemmOpConstantsBatchedForward::MPerBlock,
          GemmOpConstantsBatchedForward::NPerBlock,
          GemmOpConstantsBatchedForward::KPerBlock,
          kGemm1NPerBlock,
          GemmOpConstantsBatchedForward::Gemm1KPerBlock,
          GemmOpConstantsBatchedForward::AK1,
          GemmOpConstantsBatchedForward::BK1,
          GemmOpConstantsBatchedForward::B1K1,
          GemmOpConstantsBatchedForward::MPerXDL,
          GemmOpConstantsBatchedForward::NPerXDL,
          GemmOpConstantsBatchedForward::MXdlPerWave,
          GemmOpConstantsBatchedForward::NXdlPerWave,
          kGemm1NXdlPerWave,
          GemmOpConstantsBatchedForward::DropoutStep,
          GemmOpConstantsBatchedForward::ABlockTransferThreadClusterLengths_AK0_M_AK1,
          GemmOpConstantsBatchedForward::ABlockTransferThreadClusterArrangeOrder,
          GemmOpConstantsBatchedForward::ABlockTransferSrcAccessOrder,
          GemmOpConstantsBatchedForward::ABlockTransferSrcVectorDim,
          kABBlockTransferSrcScalarPerVector,
          GemmOpConstantsBatchedForward::ABlockTransferDstScalarPerVector_AK1,
          GemmOpConstantsBatchedForward::ABlockLdsExtraM,
          GemmOpConstantsBatchedForward::BBlockTransferThreadClusterLengths_BK0_N_BK1,
          GemmOpConstantsBatchedForward::BBlockTransferThreadClusterArrangeOrder,
          GemmOpConstantsBatchedForward::BBlockTransferSrcAccessOrder,
          GemmOpConstantsBatchedForward::BBlockTransferSrcVectorDim,
          kABBlockTransferSrcScalarPerVector,
          GemmOpConstantsBatchedForward::BBlockTransferDstScalarPerVector_BK1,
          GemmOpConstantsBatchedForward::BBlockLdsExtraN,
          kAcc0BiasTransferSrcScalarPerVector,
          GemmOpConstantsBatchedForward::B1BlockTransferThreadClusterLengths_BK0_N_BK1,
          GemmOpConstantsBatchedForward::B1BlockTransferThreadClusterArrangeOrder,
          GemmOpConstantsBatchedForward::B1BlockTransferSrcAccessOrder,
          GemmOpConstantsBatchedForward::B1BlockTransferSrcVectorDim,
          kB1BlockTransferSrcScalarPerVector,
          GemmOpConstantsBatchedForward::B1BlockTransferDstScalarPerVector_BK1,
          GemmOpConstantsBatchedForward::B1BlockLdsExtraN,
          GemmOpConstantsBatchedForward::CShuffleMXdlPerWavePerShuffle,
          kCShuffleNXdlPerWavePerShuffle,
          GemmOpConstantsBatchedForward::CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          kCShuffleBlockTransferScalarPerVector,
          GemmOpConstantsBatchedForward::Acc1BiasTransferSrcScalarPerVector,
          MaskingSpec>;
  // clang-format on

  static constexpr auto I1 = ck::Number<1>{};
  static constexpr auto I2 = ck::Number<2>{};
  static constexpr auto I3 = ck::Number<3>{};

  static void Run(BatchedForwardParams& param, hipStream_t stream) {
    using ck::math::min;

    // compile-time constants which don't depend on head-dim switching
    constexpr ck::index_t thread_slice_length_ak1 =
        GemmOpConstantsBatchedForward::AK1 /
        GemmOpConstantsBatchedForward::
            ABlockTransferThreadClusterLengths_AK0_M_AK1::At(I2);
    constexpr ck::index_t thread_slice_length_bk1 =
        GemmOpConstantsBatchedForward::BK1 /
        GemmOpConstantsBatchedForward::
            BBlockTransferThreadClusterLengths_BK0_N_BK1::At(I2);

    static_assert(
        thread_slice_length_ak1 == thread_slice_length_bk1,
        "ABlockTransfer and BBlockTransfer should use completely same K1 sizes and ThreadClusterLengths!");

    constexpr ck::index_t kABBlockTransferSrcScalarPerVector_max =
        min(4, thread_slice_length_ak1);

    BATCHED_FORWARD_HEADDIM_SWITCH(param.K, param.Kv, [&] {
      constexpr ck::index_t thread_slice_length_gemm1n = kGemm1NPerBlock /
          GemmOpConstantsBatchedForward::
              B1BlockTransferThreadClusterLengths_BK0_N_BK1::At(I1);
      constexpr ck::index_t kB1BlockTransferSrcScalarPerVector_max =
          min(2, thread_slice_length_gemm1n);

      constexpr ck::index_t thread_slice_length_cshuflle_n =
          (kCShuffleNXdlPerWavePerShuffle * kGemm1NPerBlock /
           kGemm1NXdlPerWave) /
          GemmOpConstantsBatchedForward::
              CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock ::
                  At(I3);

      constexpr ck::index_t kCShuffleBlockTransferScalarPerVector_max =
          min(1, thread_slice_length_cshuflle_n);

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
              using DeviceOpInstance = DeviceOpInstanceTemp<
                  kGemm1NPerBlock,
                  kGemm1NXdlPerWave,
                  kCShuffleNXdlPerWavePerShuffle,
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
              using DeviceOpInstance = DeviceOpInstanceTemp<
                  kGemm1NPerBlock,
                  kGemm1NXdlPerWave,
                  kCShuffleNXdlPerWavePerShuffle,
                  kABBlockTransferSrcScalarPerVector,
                  kB1BlockTransferSrcScalarPerVector,
                  kCShuffleBlockTransferScalarPerVector>;

              RunWithDeviceOp<DeviceOpInstance>(param, stream);
            });
      };
    });
  };

  template <typename DeviceOpInstance>
  static void RunWithDeviceOp(BatchedForwardParams& param, hipStream_t stream) {
    std::vector<ck::index_t> a_gs_ms_ks_lengths{
        param.B, param.Hq, param.M, param.K};
    std::vector<ck::index_t> a_gs_ms_ks_strides{
        param.q_strides[0],
        param.q_strides[2],
        param.q_strides[1],
        param.q_strides[3]};

    std::vector<ck::index_t> b0_gs_ns_ks_lengths{
        param.B, param.Hkv, param.N, param.K};
    std::vector<ck::index_t> b0_gs_ns_ks_strides{
        param.k_strides[0],
        param.k_strides[2],
        param.k_strides[1],
        param.k_strides[3]};

    // to be changed to b1_gs_ns_os_lengths
    std::vector<ck::index_t> b1_gs_os_ns_lengths{
        param.B, param.Hkv, param.Kv, param.N};
    std::vector<ck::index_t> b1_gs_os_ns_strides{
        param.v_strides[0],
        param.v_strides[2],
        param.v_strides[3],
        param.v_strides[1]};

    std::vector<ck::index_t> c_gs_ms_os_lengths{
        param.B, param.Hq, param.M, param.Kv};
    std::vector<ck::index_t> c_gs_ms_os_strides{
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

    auto a_element_op = AElementOp{};
    auto b0_element_op = B0ElementOp{};
    auto acc0_element_op = Acc0ElementOp{alpha};
    auto b1_element_op = B1ElementOp{};
    auto c_element_op = CElementOp{};

    auto op = DeviceOpInstance{};
    auto invoker = op.MakeInvoker();

    auto arg_ptr = op.MakeArgumentPointer(
        param.q_ptr,
        param.k_ptr,
        param.v_ptr,
        param.out_ptr,
        nullptr,
        param.logsumexp_ptr,
        param.has_attn_bias ? param.attn_bias_ptr : nullptr,
        {}, // p_acc1_biases;
        a_gs_ms_ks_lengths,
        a_gs_ms_ks_strides,
        b0_gs_ns_ks_lengths,
        b0_gs_ns_ks_strides,
        b1_gs_os_ns_lengths,
        b1_gs_os_ns_strides,
        c_gs_ms_os_lengths,
        c_gs_ms_os_strides,
        {1, 1, 1, 1},
        {0, 0, 0, 0},
        lse_gs_ms_lengths,
        d_gs_ms_ns_lengths,
        d_gs_ms_ns_strides,
        {}, // acc1_biases_gs_ms_os_lengths
        {}, // acc1_biases_gs_ms_os_strides,
        a_element_op,
        b0_element_op,
        acc0_element_op,
        b1_element_op,
        c_element_op,
        param.use_dropout ? param.dropout_prob : 0.0f, // dropout ratio
        std::tuple<int64_t, int64_t>(
            param.philox_seed,
            param.philox_offset)); // dropout random seed and offset

    SimpleDeviceMem workspace(op.GetWorkSpaceSize(arg_ptr.get()));

    op.SetWorkSpacePointer(arg_ptr.get(), workspace.GetDeviceBuffer());

    if (!op.IsSupportedArgument(arg_ptr.get())) {
      std::ostringstream ostr;

      ostr << op.GetTypeString() << " does not support this problem";

      throw std::runtime_error(ostr.str());
    }

    invoker.Run(arg_ptr.get(), StreamConfig{stream, false});
  };
};
