#pragma once

#include <sstream>
#include <stdexcept>

#include <ck/ck.hpp>
#include <ck/tensor_operation/gpu/device/gemm_specialization.hpp>
#include <ck/tensor_operation/gpu/device/impl/device_grouped_mha_bwd_xdl_cshuffle_kloop_v1.hpp>
#include <ck/tensor_operation/gpu/device/impl/device_grouped_mha_bwd_xdl_cshuffle_kloop_v2.hpp>
#include <ck/tensor_operation/gpu/device/tensor_specialization.hpp>
#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>
#include <ck/utility/sequence.hpp>

#include "ck_fmha_util.h"

template <typename scalar_t, int32_t custom_mask_type, bool has_attn_bias>
void grouped_backward_masktype_attnbias_dispatched(
    GroupedBackwardParams& param,
    hipStream_t stream) {
  using PassThrough = ck::tensor_operation::element_wise::PassThrough;
  using Scale = ck::tensor_operation::element_wise::Scale;

  using QKVElementOp = PassThrough;
  using YElementOp = PassThrough;

  using InputDataType = scalar_t;
  using OutputDataType = scalar_t;
  using GemmDataType = scalar_t;
  using AccDataType = F32;
  using ShuffleDataType = F32;
  using LSEDataType = F32;
  using ZDataType = unsigned short;
  using Acc0BiasDataType = ck::Tuple<>;
  using Acc1BiasDataType = ck::Tuple<>;

  static constexpr ck::index_t NumDimG = 2;
  static constexpr ck::index_t NumDimM = 1;
  static constexpr ck::index_t NumDimN = 1;
  static constexpr ck::index_t NumDimK = 1;
  static constexpr ck::index_t NumDimO = 1;

  static constexpr ck::index_t CShuffleBlockTransferScalarPerVector_NPerBlock =
      MaxVectorSizeForType<scalar_t>::value;

  static constexpr auto GemmSpec =
      ck::tensor_operation::device::GemmSpecialization::MNKOPadding;
  static constexpr auto MaskingSpec =
      static_cast<ck::tensor_operation::device::MaskingSpecialization>(
          custom_mask_type);

  static constexpr auto TensorSpecQ =
      ck::tensor_operation::device::TensorSpecialization::Default;
  static constexpr auto TensorSpecK =
      ck::tensor_operation::device::TensorSpecialization::Default;
  static constexpr auto TensorSpecV =
      ck::tensor_operation::device::TensorSpecialization::Default;
  static constexpr auto TensorSpecY =
      ck::tensor_operation::device::TensorSpecialization::Default;
  static constexpr bool Deterministic = false;

  using DeviceOpInstance = ck::tensor_operation::device::
      DeviceGroupedMultiheadAttentionBackward_Kloop_Xdl_CShuffle_V1<
          NumDimG,
          NumDimM,
          NumDimN,
          NumDimK,
          NumDimO,
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
          TensorSpecQ,
          TensorSpecK,
          TensorSpecV,
          TensorSpecY,
          1,
          256,
          128, // MPerBlock
          128, // NPerBlock
          64, // KPerBlock
          64, // Gemm1NPerBlock
          32, // Gemm1KPerBlock
          8, // AK1
          8, // BK1
          2, // B1K1
          32, // MPerXDL
          32, // NPerXDL
          1, // MXdlPerWave
          4, // NXdlPerWave
          2, // Gemm1NXdlPerWave
          2, // Gemm2NXdlPerWave
          S<4, 64, 1>, // ABlockTransfer
          S<1, 0, 2>,
          S<1, 0, 2>,
          2,
          8,
          8,
          true,
          S<4, 64, 1>, // BBlockTransfer
          S<1, 0, 2>,
          S<1, 0, 2>,
          2,
          8,
          8,
          true,
          1, // CShuffleMXdlPerWavePerShuffle
          2, // CShuffleNXdlPerWavePerShuffle
          S<1,
            32,
            1,
            8>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
          CShuffleBlockTransferScalarPerVector_NPerBlock, // CShuffleBlockTransferScalarPerVector_NPerBlock
          MaskingSpec, // MaskingSpecialization
          Deterministic>;

  std::vector<typename DeviceOpInstance::ProblemDesc> problem_descs;

  for (std::size_t i = 0; i < param.num_batches; i++) {
    int M = param.host_seqstart_q[i + 1] - param.host_seqstart_q[i]; // seqlen Q
    int N = param.host_seqstart_k.empty()
        ? param.host_seqstart_k[i + 1] - param.host_seqstart_k[i]
        : param.host_seqlen_k[i];
    int K = param.K;
    int Kv = param.Kv;
    int G1 = param.num_heads;

    std::vector<ck::index_t> q_gs_ms_ks_lengths{1, G1, M, K};
    std::vector<ck::index_t> q_gs_ms_ks_strides{
        0, param.q_strides[0], param.q_strides[1], param.q_strides[2]};

    std::vector<ck::index_t> k_gs_ns_ks_lengths{1, G1, N, K};
    std::vector<ck::index_t> k_gs_ns_ks_strides{
        0, param.k_strides[0], param.k_strides[1], param.k_strides[2]};

    // to be changed to v_gs_ns_os_lengths
    std::vector<ck::index_t> v_gs_os_ns_lengths{1, G1, Kv, N};
    std::vector<ck::index_t> v_gs_os_ns_strides{
        0, param.v_strides[0], param.v_strides[2], param.v_strides[1]};

    std::vector<ck::index_t> y_gs_ms_os_lengths{1, G1, M, Kv};
    std::vector<ck::index_t> y_gs_ms_os_strides{
        0, param.out_strides[0], param.out_strides[1], param.out_strides[2]};

    std::vector<ck::index_t> z_gs_ms_ns_lengths{1, G1, M, N};
    std::vector<ck::index_t> z_gs_ms_ns_strides{
        0,
        param.randvals_strides[0],
        param.randvals_strides[1],
        param.randvals_strides[2]};

    std::vector<ck::index_t> lse_gs_ms_lengths{1, G1, M};
    std::vector<ck::index_t> lse_gs_ms_strides{0, param.M, 1};

    problem_descs.push_back({
        q_gs_ms_ks_lengths,
        q_gs_ms_ks_strides,
        k_gs_ns_ks_lengths,
        k_gs_ns_ks_strides,
        z_gs_ms_ns_lengths,
        z_gs_ms_ns_strides,
        v_gs_os_ns_lengths,
        v_gs_os_ns_strides,
        y_gs_ms_os_lengths,
        y_gs_ms_os_strides,
        lse_gs_ms_lengths,
        lse_gs_ms_strides,
        {}, // std::array<std::vector<ck::index_t>,
            // 1>{acc0_biases_gs_ms_ns_lengths},
        {}, // std::array<std::vector<ck::index_t>,
            // 1>{acc0_biases_gs_ms_ns_strides},
        {}, // std::array<std::vector<ck::index_t>,
            // 1>{acc1_biases_gs_ms_os_lengths},
        {}, // std::array<std::vector<ck::index_t>,
            // 1>{acc1_biases_gs_ms_os_strides},
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
      {}, // std::array<void*, 1> p_acc0_biases;
      {}, // std::array<void*, 1> p_acc1_biases;
      problem_descs,
      QKVElementOp{},
      QKVElementOp{},
      Scale{alpha},
      QKVElementOp{},
      YElementOp{},
      param.dropout_prob,
      std::tuple<unsigned long long, unsigned long long>(
          param.rng_seed, param.rng_offset));

  SimpleDeviceMem workspace(op.GetWorkSpaceSize(arg_ptr.get()));

  op.SetWorkSpacePointer(arg_ptr.get(), workspace.GetDeviceBuffer());

  if (!op.IsSupportedArgument(arg_ptr.get())) {
    std::ostringstream ostr;
    ostr << op.GetTypeString() << " does not support this problem";

    throw std::runtime_error(ostr.str());
  }

  (void)invoker.Run(arg_ptr.get(), StreamConfig{stream, false});
};
