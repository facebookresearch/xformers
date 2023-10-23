#pragma once

#include <sstream>
#include <stdexcept>

#include <ck/ck.hpp>
#include <ck/tensor_operation/gpu/device/gemm_specialization.hpp>
#include <ck/tensor_operation/gpu/device/tensor_specialization.hpp>
#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>
#include "ck/tensor_operation/gpu/device/impl/device_batched_mha_infer_xdl_cshuffle.hpp"

#include "ck_fmha_device_gemm_constants.h"
#include "ck_fmha_op_helper.h"
#include "ck_fmha_params.h"

template <typename scalar_t, int32_t custom_mask_type, bool has_attn_bias>
struct batched_infer_masktype_attnbias_dispatched {
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

  static constexpr ck::index_t kABBlockTransferSrcScalarPerVector = 1;
  static constexpr ck::index_t kB1BlockTransferSrcScalarPerVector = 1;
  static constexpr ck::index_t kCShuffleBlockTransferScalarPerVector = 1;
  static constexpr ck::index_t kAcc0BiasTransferSrcScalarPerVector = 1;

  template <
      ck::index_t kGemm1NPerBlock,
      ck::index_t kGemm1NXdlPerWave,
      ck::index_t kCShuffleNXdlPerWavePerShuffle>
  using DeviceOpInstanceTemp = ck::tensor_operation::device::
      DeviceBatchedMultiheadAttentionInfer_Xdl_CShuffle<
          GemmOpConstantsCommon::NumDimG,
          GemmOpConstantsCommon::NumDimM,
          GemmOpConstantsCommon::NumDimN,
          GemmOpConstantsCommon::NumDimK,
          GemmOpConstantsCommon::NumDimO,
          ADataType,
          B0DataType,
          B1DataType,
          CDataType,
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
          1,
          GemmOpConstantsBatchedInfer::BlockSize,
          GemmOpConstantsBatchedInfer::MPerBlock,
          GemmOpConstantsBatchedInfer::NPerBlock,
          GemmOpConstantsBatchedInfer::KPerBlock,
          kGemm1NPerBlock,
          GemmOpConstantsBatchedInfer::Gemm1KPerBlock,
          GemmOpConstantsBatchedInfer::AK1,
          GemmOpConstantsBatchedInfer::BK1,
          GemmOpConstantsBatchedInfer::B1K1,
          GemmOpConstantsBatchedInfer::MPerXDL,
          GemmOpConstantsBatchedInfer::NPerXDL,
          GemmOpConstantsBatchedInfer::MXdlPerWave,
          GemmOpConstantsBatchedInfer::NXdlPerWave,
          kGemm1NXdlPerWave,
          GemmOpConstantsBatchedInfer::
              ABlockTransferThreadClusterLengths_AK0_M_AK1,
          GemmOpConstantsBatchedInfer::ABlockTransferThreadClusterArrangeOrder,
          GemmOpConstantsBatchedInfer::ABlockTransferSrcAccessOrder,
          GemmOpConstantsBatchedInfer::ABlockTransferSrcVectorDim,
          kABBlockTransferSrcScalarPerVector,
          GemmOpConstantsBatchedInfer::ABlockTransferDstScalarPerVector_AK1,
          GemmOpConstantsBatchedInfer::ABlockLdsExtraM,
          GemmOpConstantsBatchedInfer::
              BBlockTransferThreadClusterLengths_BK0_N_BK1,
          GemmOpConstantsBatchedInfer::BBlockTransferThreadClusterArrangeOrder,
          GemmOpConstantsBatchedInfer::BBlockTransferSrcAccessOrder,
          GemmOpConstantsBatchedInfer::BBlockTransferSrcVectorDim,
          kABBlockTransferSrcScalarPerVector,
          GemmOpConstantsBatchedInfer::BBlockTransferDstScalarPerVector_BK1,
          GemmOpConstantsBatchedInfer::BBlockLdsExtraN,
          kAcc0BiasTransferSrcScalarPerVector,
          GemmOpConstantsBatchedInfer::
              B1BlockTransferThreadClusterLengths_BK0_N_BK1,
          GemmOpConstantsBatchedInfer::B1BlockTransferThreadClusterArrangeOrder,
          GemmOpConstantsBatchedInfer::B1BlockTransferSrcAccessOrder,
          GemmOpConstantsBatchedInfer::B1BlockTransferSrcVectorDim,
          kB1BlockTransferSrcScalarPerVector,
          GemmOpConstantsBatchedInfer::B1BlockTransferDstScalarPerVector_BK1,
          GemmOpConstantsBatchedInfer::B1BlockLdsExtraN,
          GemmOpConstantsBatchedInfer::CShuffleMXdlPerWavePerShuffle,
          kCShuffleNXdlPerWavePerShuffle,
          GemmOpConstantsBatchedInfer::
              CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          kCShuffleBlockTransferScalarPerVector,
          MaskingSpec>;

  static void Run(BatchedForwardParams& param, hipStream_t stream) {
    if (param.K <= 32 && param.Kv <= 32) {
      constexpr ck::index_t kGemm1NPerBlock = 32;
      constexpr ck::index_t kGemm1NXdlPerWave = 1;
      constexpr ck::index_t kCShuffleNXdlPerWavePerShuffle = 1;

      using DeviceOpInstance = DeviceOpInstanceTemp<
          kGemm1NPerBlock,
          kGemm1NXdlPerWave,
          kCShuffleNXdlPerWavePerShuffle>;

      RunWithDeviceOp<DeviceOpInstance>(param, stream);
    } else if (param.K <= 64 && param.Kv <= 64) {
      constexpr ck::index_t kGemm1NPerBlock = 64;
      constexpr ck::index_t kGemm1NXdlPerWave = 2;
      constexpr ck::index_t kCShuffleNXdlPerWavePerShuffle = 2;

      using DeviceOpInstance = DeviceOpInstanceTemp<
          kGemm1NPerBlock,
          kGemm1NXdlPerWave,
          kCShuffleNXdlPerWavePerShuffle>;

      RunWithDeviceOp<DeviceOpInstance>(param, stream);
    } else {
      constexpr ck::index_t kGemm1NPerBlock = 128;
      constexpr ck::index_t kGemm1NXdlPerWave = 4;
      constexpr ck::index_t kCShuffleNXdlPerWavePerShuffle = 4;

      using DeviceOpInstance = DeviceOpInstanceTemp<
          kGemm1NPerBlock,
          kGemm1NXdlPerWave,
          kCShuffleNXdlPerWavePerShuffle>;

      RunWithDeviceOp<DeviceOpInstance>(param, stream);
    };
  };

  template <typename DeviceOpInstance>
  static void RunWithDeviceOp(BatchedForwardParams& param, hipStream_t stream) {
    std::vector<ck::index_t> a_gs_ms_ks_lengths{
        param.B, param.num_heads, param.M, param.K};
    std::vector<ck::index_t> a_gs_ms_ks_strides{
        param.q_strides[0],
        param.q_strides[2],
        param.q_strides[1],
        param.q_strides[3]};

    std::vector<ck::index_t> b0_gs_ns_ks_lengths{
        param.B, param.num_heads, param.N, param.K};
    std::vector<ck::index_t> b0_gs_ns_ks_strides{
        param.k_strides[0],
        param.k_strides[2],
        param.k_strides[1],
        param.k_strides[3]};

    // to be changed to b1_gs_ns_os_lengths
    std::vector<ck::index_t> b1_gs_os_ns_lengths{
        param.B, param.num_heads, param.Kv, param.N};
    std::vector<ck::index_t> b1_gs_os_ns_strides{
        param.v_strides[0],
        param.v_strides[2],
        param.v_strides[3],
        param.v_strides[1]};

    std::vector<ck::index_t> c_gs_ms_os_lengths{
        param.B, param.num_heads, param.M, param.Kv};
    std::vector<ck::index_t> c_gs_ms_os_strides{
        param.out_strides[0],
        param.out_strides[2],
        param.out_strides[1],
        param.out_strides[3]};

    std::vector<ck::index_t> lse_gs_ms_lengths{
        param.B, param.num_heads, param.M};

    std::vector<ck::index_t> d_gs_ms_ns_lengths;
    std::vector<ck::index_t> d_gs_ms_ns_strides;

    if constexpr (has_attn_bias) {
      d_gs_ms_ns_lengths = {param.B, param.num_heads, param.M, param.N};
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
        d_gs_ms_ns_lengths,
        d_gs_ms_ns_strides,
        {}, // acc1_biases_gs_ms_os_lengths
        {}, // acc1_biases_gs_ms_os_strides,
        a_element_op,
        b0_element_op,
        acc0_element_op,
        b1_element_op,
        c_element_op);

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
