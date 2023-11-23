#pragma once

#include <optional>
#include <sstream>
#include <stdexcept>

#include <ck/host_utility/device_prop.hpp>
#include <ck/host_utility/kernel_launch.hpp>
#include <ck/tensor/tensor_view.hpp>
#include <ck/tensor_description/cluster_descriptor.hpp>
#include <ck/tensor_description/tensor_descriptor_helper.hpp>
#include <ck/utility/common_header.hpp>

#include <ck/tile_program/block_tile_pipeline/block_fmha_pipeline_problem.hpp>
#include <ck/tile_program/block_tile_pipeline/block_fmha_pipeline_qkvs.hpp>
#include <ck/tile_program/block_tile_pipeline/block_fmha_pipeline_qkvs_default_policy.hpp>
#include <ck/tile_program/block_tile_pipeline/block_fmha_pipeline_qr_ks_vs.hpp>
#include <ck/tile_program/block_tile_pipeline/block_fmha_pipeline_qr_ks_vs_default_policy.hpp>
#include <ck/tile_program/tile/tile_fmha_shape.hpp>

#include "ck_tiled_fmha_params.h"
#include "ck_tiled_fmha_forward_kernel.h"
#include "ck_tiled_fmha_fwd_epilogue.h"
#include "ck_tiled_fmha_fwd_tile_partitioner.h"

template <typename scalar_t, int32_t custom_mask_type, bool has_attn_bias>
struct batched_infer_masktype_attnbias_dispatched {
  using QDataType = scalar_t;
  using KDataType = scalar_t;
  using VDataType = scalar_t;
  using BiasDataType = scalar_t;
  using SaccDataType = float; // data type for first gemm accumulation
  using SMPLComputeDataType = float; // data type for reduction, softmax
  using PDataType = scalar_t; // data type for A matrix of second gemm
  using OaccDataType = float; // data type for second gemm accumulation
  using ODataType = scalar_t;

  using VLayout = ck::tensor_layout::gemm::RowMajor;

  using FmhaBlockTileHdim64 = ck::Sequence<128, 64, 32, 64, 32, 64>;
  using FmhaBlockTileHdim128 = ck::Sequence<128, 128, 32, 128, 32, 128>;
  using FmhaBlockWarps = ck::Sequence<4, 1, 1>;
  using FmhaWarpTile = ck::Sequence<32, 32, 16>;
  using FmhaShapeHDim64 = ck::tile_program::TileFmhaShape<
      FmhaBlockTileHdim64,
      FmhaBlockWarps,
      FmhaWarpTile,
      FmhaBlockWarps,
      FmhaWarpTile,
      VLayout>;
  using FmhaShapeHDim128 = ck::tile_program::TileFmhaShape<
      FmhaBlockTileHdim128,
      FmhaBlockWarps,
      FmhaWarpTile,
      FmhaBlockWarps,
      FmhaWarpTile,
      VLayout>;

  using FmhaTilePartitionerHDim64 = FmhaFwdTilePartitioner<FmhaShapeHDim64>;
  using FmhaTilePartitionerHDim128 = FmhaFwdTilePartitioner<FmhaShapeHDim128>;
  using FmhaPipelineProblemHDim64 =
      ck::tile_program::block::BlockFmhaPipelineProblem<
          QDataType,
          KDataType,
          VDataType,
          SaccDataType,
          SMPLComputeDataType,
          BiasDataType,
          PDataType,
          OaccDataType,
          ODataType,
          256, // BlockSize
          FmhaShapeHDim64>;
  using FmhaPipelineProblemHDim128 =
      ck::tile_program::block::BlockFmhaPipelineProblem<
          QDataType,
          KDataType,
          VDataType,
          SaccDataType,
          SMPLComputeDataType,
          BiasDataType,
          PDataType,
          OaccDataType,
          ODataType,
          256, // BlockSize
          FmhaShapeHDim128>;

  using FmhaPipelineHDim64 = ck::tile_program::block::BlockFmhaPipelineQRKSVS<
      FmhaPipelineProblemHDim64>;
  using FmhaPipelineHDim128 = ck::tile_program::block::BlockFmhaPipelineQRKSVS<
      FmhaPipelineProblemHDim128>;

  using FmhaEpilogue =
      FmhaFwdEpilogue<FmhaFwdEpilogueProblem<OaccDataType, ODataType>>;
  using FmhaKernelHDim64 = FmhaFwdKernel<
      FmhaTilePartitionerHDim64,
      FmhaPipelineHDim64,
      FmhaEpilogue>;
  using FmhaKernelHDim128 = FmhaFwdKernel<
      FmhaTilePartitionerHDim128,
      FmhaPipelineHDim128,
      FmhaEpilogue>;

#ifndef BATCHED_INFER_HEADDIM_SWITCH
#define BATCHED_INFER_HEADDIM_SWITCH(HEAD_DIM1, HEAD_DIM2, ...)  \
  [&] {                                                          \
    if (HEAD_DIM1 == HEAD_DIM2 && HEAD_DIM2 == 64) {             \
      using FmhaKernel = FmhaKernelHDim64;                       \
      __VA_ARGS__();                                             \
    } else if (HEAD_DIM1 == HEAD_DIM2 && HEAD_DIM2 == 128) {     \
      using FmhaKernel = FmhaKernelHDim128;                      \
      __VA_ARGS__();                                             \
    } else {                                                     \
      throw std::runtime_error("Head-dim sizes not supported!"); \
    }                                                            \
  }()
#endif

  static void Run(BatchedForwardParams& param, hipStream_t stream) {
    BATCHED_INFER_HEADDIM_SWITCH(
        param.K, param.Kv, [&] { RunWithKernel<FmhaKernel>(param, stream); });
  };

  template <typename FmhaKernel>
  static void RunWithKernel(BatchedForwardParams& param, hipStream_t stream) {
    dim3 kGridSize = FmhaKernel::GridSize(param.B, param.Hq, param.M, param.Kv);
    constexpr dim3 kBlockSize = FmhaKernel::BlockSize();

    constexpr ck::index_t kWarpPerCu = 8; // 2 warps per SIMD
    constexpr ck::index_t kWarpPerBlock = kBlockSize.x / warpSize;
    constexpr ck::index_t kBlockPerCu = kWarpPerCu / kWarpPerBlock;

    std::optional<
        std::tuple<const void*, ck::index_t, ck::index_t, ck::index_t>>
        bias;

    if (param.has_attn_bias)
      bias = std::make_tuple(
          param.attn_bias_ptr,
          param.attn_bias_strides[2],
          param.attn_bias_strides[1],
          param.attn_bias_strides[0]);

    auto kargs = FmhaKernel::MakeKargs(
        param.q_ptr,
        param.k_ptr,
        param.v_ptr,
        param.out_ptr,
        param.M, // seqlen_q
        param.N, // seqlen_k
        param.K, // hdim_q
        param.Kv, // hdim_v
        param.scale,
        param.q_strides[1], // q, k, v, out tensor seq-dim stride
        param.k_strides[1],
        param.v_strides[1],
        param.out_strides[1],
        param.q_strides[2], // q, k, v, out tensor head-dim stride
        param.k_strides[2],
        param.v_strides[2],
        param.out_strides[2],
        param.q_strides[0], // q, k, v, out tensor batch-dim stride
        param.k_strides[0],
        param.v_strides[0],
        param.out_strides[0],
        bias);

    (void)launch_kernel<kBlockSize.x, kBlockPerCu>(
        StreamConfig{stream, false},
        FmhaKernel{},
        kGridSize,
        kBlockSize,
        0,
        kargs);
  };
};

template <typename scalar_t, int32_t custom_mask_type, bool has_attn_bias>
void run_batched_infer_masktype_attnbias_dispatched(
    BatchedForwardParams& param,
    hipStream_t stream) {
  batched_infer_masktype_attnbias_dispatched<
      scalar_t,
      custom_mask_type,
      has_attn_bias>::Run(param, stream);
};
