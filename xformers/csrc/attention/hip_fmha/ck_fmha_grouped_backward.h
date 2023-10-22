#pragma once

#include <sstream>
#include <stdexcept>

#include <ck/ck.hpp>
#include <ck/tensor_operation/gpu/device/gemm_specialization.hpp>
#include <ck/tensor_operation/gpu/device/tensor_specialization.hpp>
#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>
#include <ck/utility/sequence.hpp>
#include "ck/tensor_operation/gpu/device/impl/device_grouped_mha_bwd_xdl_cshuffle_qloop_v1.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_mha_bwd_xdl_cshuffle_qloop_v2.hpp"

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
  static constexpr bool Deterministic = true;

  static constexpr ck::index_t kABBlockTransferSrcScalarPerVector = 1;
  static constexpr ck::index_t kB1BlockTransferSrcScalarPerVector = 1;
  static constexpr ck::index_t kCShuffleBlockTransferScalarPerVector = 1;
  static constexpr ck::index_t kAcc0BiasTransferSrcScalarPerVector = 1;

  template <
      ck::index_t kGemm1NPerBlock,
      ck::index_t kGemm1NXdlPerWave,
      ck::index_t kCShuffleNXdlPerWavePerShuffle,
      typename kCShuffleBlockTransferClusterLengths>
  using DeviceOpInstanceTemp = ck::tensor_operation::device::
      DeviceGroupedMultiheadAttentionBackward_Qloop_Xdl_CShuffle_V1<
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
          kGemm1NPerBlock, // KPerBlock = kGemm1NerBlock
          kGemm1NPerBlock,
          32, // Gemm1KPerBlock
          32, // Gemm2KPerBlock
          8, // AK1
          8, // BK1
          2, // B1K1
          32, // MPerXDL
          32, // NPerXDL
          4, // MXdlPerWave
          1, // NXdlPerWave
          kGemm1NXdlPerWave,
          1, // Gemm2NXdlPerWave
          S<4, 64, 1>, // ABlockTransfer
          S<1, 0, 2>,
          S<1, 0, 2>,
          2,
          kABBlockTransferSrcScalarPerVector, // TUNABLE
          8,
          true,
          S<4, 64, 1>, // BBlockTransfer
          S<1, 0, 2>,
          S<1, 0, 2>,
          2,
          kABBlockTransferSrcScalarPerVector, // TUNABLE
          8,
          true,
          kAcc0BiasTransferSrcScalarPerVector, // TUNABLE
          1, // CShuffleMXdlPerWavePerShuffle
          kCShuffleNXdlPerWavePerShuffle,
          kCShuffleBlockTransferClusterLengths,
          kCShuffleBlockTransferScalarPerVector, // TUNABLE
          MaskingSpec,
          Deterministic>;

  static void Run(GroupedBackwardParams& param, hipStream_t stream) {
    if (param.K <= 32 && param.Kv <= 32) {
      constexpr ck::index_t kGemm1NPerBlock = 32;
      constexpr ck::index_t kGemm1NXdlPerWave = 1;
      constexpr ck::index_t kCShuffleNXdlPerWavePerShuffle = 1;
      using kCShuffleBlockTransferClusterLengths = S<1, 64, 1, 4>;

      using DeviceOpInstance = DeviceOpInstanceTemp<
          kGemm1NPerBlock,
          kGemm1NXdlPerWave,
          kCShuffleNXdlPerWavePerShuffle,
          kCShuffleBlockTransferClusterLengths>;

      RunWithDeviceOp<DeviceOpInstance>(param, stream);
    } else if (param.K <= 64 && param.Kv <= 64) {
      constexpr ck::index_t kGemm1NPerBlock = 64;
      constexpr ck::index_t kGemm1NXdlPerWave = 2;
      constexpr ck::index_t kCShuffleNXdlPerWavePerShuffle = 2;
      using kCShuffleBlockTransferClusterLengths = S<1, 32, 1, 8>;

      using DeviceOpInstance = DeviceOpInstanceTemp<
          kGemm1NPerBlock,
          kGemm1NXdlPerWave,
          kCShuffleNXdlPerWavePerShuffle,
          kCShuffleBlockTransferClusterLengths>;

      RunWithDeviceOp<DeviceOpInstance>(param, stream);
    } else {
      using DeviceOpInstance = ck::tensor_operation::device::
          DeviceGroupedMultiheadAttentionBackward_Qloop_Xdl_CShuffle_V2<
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
              64, // MPerBlock
              128, // NPerBlock
              128, // KPerBlock
              128, // Gemm1NPerBlock
              32, // Gemm1KPerBlock
              64, // Gemm2KPerBlock
              8, // AK1
              8, // BK1
              2, // B1K1
              32, // MPerXDL
              32, // NPerXDL
              2, // MXdlPerWave
              1, // NXdlPerWave
              4, // Gemm1NXdlPerWave
              1, // Gemm2NXdlPerWave
              S<4, 64, 1>, // ABlockTransfer
              S<1, 0, 2>,
              S<1, 0, 2>,
              2,
              kABBlockTransferSrcScalarPerVector, // TUNABLE
              8,
              true,
              S<4, 64, 1>, // B0BlockTransfer
              S<1, 0, 2>,
              S<1, 0, 2>,
              2,
              kABBlockTransferSrcScalarPerVector, // TUNABLE
              8,
              true,
              kAcc0BiasTransferSrcScalarPerVector, // TUNABLE
              S<8, 32, 1>, // B1BlockTransfer
              S<0, 2, 1>,
              S<0, 2, 1>,
              1,
              kB1BlockTransferSrcScalarPerVector, // TUNABLE
              2,
              false,
              1, // CShuffleMXdlPerWavePerShuffle
              4, // CShuffleNXdlPerWavePerShuffle
              S<1, 32, 1, 8>,
              kCShuffleBlockTransferScalarPerVector, // TUNABLE
              MaskingSpec,
              Deterministic>;

      RunWithDeviceOp<DeviceOpInstance>(param, stream);
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

      // to be changed to v_gs_ns_os_lengths
      std::vector<ck::index_t> v_gs_os_ns_lengths{1, G1, Kv, N};
      std::vector<ck::index_t> v_gs_os_ns_strides{
          0, param.v_strides[1], param.v_strides[2], param.v_strides[0]};

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
