#pragma once

#include <sstream>
#include <stdexcept>

#include <ck/ck.hpp>
#include <ck/tensor_operation/gpu/device/gemm_specialization.hpp>
#include <ck/tensor_operation/gpu/device/impl/device_grouped_mha_fwd_xdl_cshuffle_v2.hpp>
#include <ck/tensor_operation/gpu/device/tensor_specialization.hpp>
#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>
#include <ck/utility/sequence.hpp>

#include "ck_fmha_op_helper.h"
#include "ck_fmha_params.h"

template <typename scalar_t, int32_t custom_mask_type, bool has_attn_bias>
void grouped_forward_masktype_attnbias_dispatched(
    GroupedForwardParams& param,
    hipStream_t stream) {
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

  static constexpr auto TensorSpecA =
      ck::tensor_operation::device::TensorSpecialization::Default;
  static constexpr auto TensorSpecB0 =
      ck::tensor_operation::device::TensorSpecialization::Default;
  static constexpr auto TensorSpecB1 =
      ck::tensor_operation::device::TensorSpecialization::Default;
  static constexpr auto TensorSpecC =
      ck::tensor_operation::device::TensorSpecialization::Default;
  static constexpr bool Deterministic = true;

  // Tunables
  static constexpr ck::index_t ABBlockTransferSrcScalarPerVector = 1;
  static constexpr ck::index_t B1CShuffleBlockTransferScalarPerVector = 1;
  static constexpr ck::index_t Acc0BiasTransferSrcScalarPerVector = 1;

  using DeviceOpInstance = ck::tensor_operation::device::
      DeviceGroupedMultiheadAttentionForward_Xdl_CShuffle_V2<
          NumDimG,
          NumDimM,
          NumDimN,
          NumDimK,
          NumDimO,
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
          TensorSpecA,
          TensorSpecB0,
          TensorSpecB1,
          TensorSpecC,
          1,
          256,
          128, // MPerBlock
          128, // NPerBlock
          32, // KPerBlock
          128, // Gemm1NPerBlock
          32, // Gemm1KPerBlock
          8, // AK1
          8, // BK1
          2, // B1K1
          32, // MPerXDL
          32, // NPerXDL
          1, // MXdlPerWave
          4, // NXdlPerWave
          4, // Gemm1NXdlPerWave
          1, // DropoutStep
          S<4, 64, 1>, // ABlockTransfer
          S<1, 0, 2>,
          S<1, 0, 2>,
          2,
          ABBlockTransferSrcScalarPerVector, // TUNABLE
          8,
          true,
          S<4, 64, 1>, // BBlockTransfer
          S<1, 0, 2>,
          S<1, 0, 2>,
          2,
          ABBlockTransferSrcScalarPerVector, // TUNABLE
          8,
          true,
          Acc0BiasTransferSrcScalarPerVector,
          S<8, 32, 1>, // B1BlockTransfer
          S<0, 2, 1>,
          S<0, 2, 1>,
          1,
          B1CShuffleBlockTransferScalarPerVector, // TUNABLE
          2,
          false,
          1, // CShuffleMXdlPerWavePerShuffle
          2, // CShuffleNXdlPerWavePerShuffle
          S<1,
            32,
            1,
            8>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
          B1CShuffleBlockTransferScalarPerVector, // TUNABLE
          1,
          MaskingSpec, // MaskingSpecialization
          Deterministic>;

  std::vector<typename DeviceOpInstance::ProblemDesc> problem_descs;

  for (std::size_t i = 0; i < param.num_batches; i++) {
    int M = param.host_seqstart_q[i + 1] - param.host_seqstart_q[i];
    int N = param.host_seqlen_k.empty()
        ? param.host_seqstart_k[i + 1] - param.host_seqstart_k[i]
        : param.host_seqlen_k[i];
    int K = param.K;
    int Kv = param.Kv;
    int G1 = param.num_heads;

    std::vector<ck::index_t> a_gs_ms_ks_lengths{1, G1, M, K};
    std::vector<ck::index_t> a_gs_ms_ks_strides{
        0, param.q_strides[1], param.q_strides[0], param.q_strides[2]};

    std::vector<ck::index_t> b0_gs_ns_ks_lengths{1, G1, N, K};
    std::vector<ck::index_t> b0_gs_ns_ks_strides{
        0, param.k_strides[1], param.k_strides[0], param.k_strides[2]};

    // to be changed to b1_gs_ns_os_lengths
    std::vector<ck::index_t> b1_gs_os_ns_lengths{1, G1, Kv, N};
    std::vector<ck::index_t> b1_gs_os_ns_strides{
        0, param.v_strides[1], param.v_strides[2], param.v_strides[0]};

    std::vector<ck::index_t> c_gs_ms_os_lengths{1, G1, M, Kv};
    std::vector<ck::index_t> c_gs_ms_os_strides{
        0, param.out_strides[1], param.out_strides[0], param.out_strides[2]};

    std::vector<ck::index_t> z_gs_ms_ns_lengths{1, G1, M, N};
    std::vector<ck::index_t> z_gs_ms_ns_strides{
        0,
        param.randvals_strides[0],
        param.randvals_strides[1],
        param.randvals_strides[2]};

    std::vector<ck::index_t> lse_gs_ms_lengths{1, G1, M};
    std::vector<ck::index_t> lse_gs_ms_strides{0, param.M, 1};

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

    problem_descs.push_back(
        {a_gs_ms_ks_lengths,
         a_gs_ms_ks_strides,
         b0_gs_ns_ks_lengths,
         b0_gs_ns_ks_strides,
         b1_gs_os_ns_lengths,
         b1_gs_os_ns_strides,
         c_gs_ms_os_lengths,
         c_gs_ms_os_strides,
         z_gs_ms_ns_lengths,
         z_gs_ms_ns_strides,
         lse_gs_ms_lengths,
         lse_gs_ms_strides,
         d_gs_ms_ns_lengths,
         d_gs_ms_ns_strides,
         {}, // acc1_bias_gs_ms_os_lengths
         {}}); // acc1_bias_gs_ms_os_strides
  }

  float alpha = param.scale;

  auto a_element_op = AElementOp{};
  auto b0_element_op = B0ElementOp{};
  auto acc0_element_op = Acc0ElementOp{alpha};
  auto b1_element_op = B1ElementOp{};
  auto c_element_op = CElementOp{};

  auto op = DeviceOpInstance{};
  auto invoker = op.MakeInvoker();

  auto arg_ptr = op.MakeArgumentPointer(
      param.q_ptrs,
      param.k_ptrs,
      param.v_ptrs,
      param.out_ptrs,
      param.randvals_ptrs,
      param.logsumexp_ptrs,
      param.attn_bias_ptrs,
      {}, // p_acc1_biases
      problem_descs,
      a_element_op,
      b0_element_op,
      acc0_element_op,
      b1_element_op,
      c_element_op,
      param.use_dropout ? param.dropout_prob : 0.0f, // dropout ratio
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
