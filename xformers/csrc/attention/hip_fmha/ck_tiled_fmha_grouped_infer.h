#pragma once

#include <optional>
#include <sstream>
#include <stdexcept>

#include <ck/utility/common_header.hpp>
#include <ck/host_utility/device_prop.hpp>
#include <ck/host_utility/kernel_launch.hpp>
#include <ck/tensor_description/cluster_descriptor.hpp>
#include <ck/tensor_description/tensor_descriptor_helper.hpp>
#include <ck/tensor/tensor_view.hpp>

#include <ck/tile_program/block_tile_pipeline/block_fmha_pipeline_problem.hpp>
#include <ck/tile_program/block_tile_pipeline/block_fmha_pipeline_qkvs.hpp>
#include <ck/tile_program/block_tile_pipeline/block_fmha_pipeline_qkvs_default_policy.hpp>
#include <ck/tile_program/block_tile_pipeline/block_fmha_pipeline_qr_ks_vs.hpp>
#include <ck/tile_program/block_tile_pipeline/block_fmha_pipeline_qr_ks_vs_default_policy.hpp>
#include <ck/tile_program/tile/tile_fmha_shape.hpp>

#include "ck_fmha_op_helper.h"
#include "ck_fmha_util.h"
#include "ck_tiled_fmha_forward_kernel.h"
#include "ck_tiled_fmha_fwd_epilogue.h"
#include "ck_tiled_fmha_fwd_tile_partitioner.h"
#include "ck_tiled_fmha_params.h"

template <typename scalar_t, int32_t custom_mask_type>
struct grouped_infer_masktype_attnbias_dispatched
{
    using QDataType           = scalar_t;
    using KDataType           = scalar_t;
    using VDataType           = scalar_t;
    using BiasDataType        = scalar_t;
    using SaccDataType        = float;    // data type for first gemm accumulation
    using SMPLComputeDataType = float;    // data type for reduction, softmax
    using PDataType           = scalar_t; // data type for A matrix of second gemm
    using OaccDataType        = float;    // data type for second gemm accumulation
    using ODataType           = scalar_t;

    using VLayout = ck::tensor_layout::gemm::RowMajor;

    using FmhaBlockTileHdim64  = ck::Sequence<128, 64, 32, 64, 32, 64>;
    using FmhaBlockTileHdim128 = ck::Sequence<128, 128, 32, 128, 32, 128>;
    using FmhaBlockWarps       = ck::Sequence<4, 1, 1>;
    using FmhaWarpTile         = ck::Sequence<32, 32, 16>;
    using FmhaShapeHDim64      = ck::tile_program::TileFmhaShape<FmhaBlockTileHdim64,
                                                            FmhaBlockWarps,
                                                            FmhaWarpTile,
                                                            FmhaBlockWarps,
                                                            FmhaWarpTile,
                                                            VLayout>;
    using FmhaShapeHDim128     = ck::tile_program::TileFmhaShape<FmhaBlockTileHdim128,
                                                             FmhaBlockWarps,
                                                             FmhaWarpTile,
                                                             FmhaBlockWarps,
                                                             FmhaWarpTile,
                                                             VLayout>;

    using FmhaEpilogue = FmhaFwdEpilogue<FmhaFwdEpilogueProblem<OaccDataType, ODataType>>;

    // This is the default setting, the effective setting should be done according to M/N size of
    // each batch
    static constexpr bool MNeedPadding = true;
    static constexpr bool NNeedPadding = true;

#ifndef GROUPED_INFER_HEADDIM_SWITCH
#define GROUPED_INFER_HEADDIM_SWITCH(HEAD_DIM1, HEAD_DIM2, ...)        \
    [&] {                                                              \
        if(HEAD_DIM1 == HEAD_DIM2 && HEAD_DIM2 == 64)                  \
        {                                                              \
            using FmhaShape = FmhaShapeHDim64;                         \
            __VA_ARGS__();                                             \
        }                                                              \
        else if(HEAD_DIM1 == HEAD_DIM2 && HEAD_DIM2 == 128)            \
        {                                                              \
            using FmhaShape = FmhaShapeHDim128;                        \
            __VA_ARGS__();                                             \
        }                                                              \
        else                                                           \
        {                                                              \
            throw std::runtime_error("Head-dim sizes not supported!"); \
        }                                                              \
    }()
#endif

    static void Run(GroupedForwardParams& param, hipStream_t stream)
    {
        GROUPED_INFER_HEADDIM_SWITCH(param.K, param.Kv, [&] {
            using FmhaTilePartitioner = FmhaFwdTilePartitioner<FmhaShape>;
            using FmhaPipelineProblem =
                ck::tile_program::block::BlockFmhaPipelineProblem<QDataType,
                                                                  KDataType,
                                                                  VDataType,
                                                                  SaccDataType,
                                                                  SMPLComputeDataType,
                                                                  BiasDataType,
                                                                  PDataType,
                                                                  OaccDataType,
                                                                  ODataType,
                                                                  256, // BlockSize
                                                                  FmhaShape>;

            using FmhaPipeline =
                ck::tile_program::block::BlockFmhaPipelineQRKSVS<FmhaPipelineProblem>;

            using FmhaKernel = FmhaFwdKernel<FmhaTilePartitioner,
                                             FmhaPipeline,
                                             FmhaEpilogue,
                                             MNeedPadding,
                                             NNeedPadding>;

            RunWithKernel<FmhaKernel>(param, stream);
        });
    };

    template <typename FmhaKernel>
    static void RunWithKernel(GroupedForwardParams& param, hipStream_t stream)
    {
        dim3 kGridSize = FmhaKernel::GridSize(param.num_batches, param.Hq, param.M, param.Kv);
        constexpr dim3 kBlockSize = FmhaKernel::BlockSize();

        constexpr ck::index_t kWarpPerCu    = 8; // 2 warps per SIMD
        constexpr ck::index_t kWarpPerBlock = kBlockSize.x / warpSize;
        constexpr ck::index_t kBlockPerCu   = kWarpPerCu / kWarpPerBlock;

        std::optional<std::tuple<const void*, ck::index_t, ck::index_t>> bias;

        if(param.has_attn_bias)
        {
            bias = std::make_tuple(
                param.attn_bias_ptr, param.attn_bias_strides[2], param.attn_bias_strides[1]);
        };

        auto kargs =
            FmhaKernel::MakeKargs(param.q_ptr,
                                  param.k_ptr,
                                  param.v_ptr,
                                  param.out_ptr,
                                  param.seqstart_q_dev_ptr,
                                  param.seqstart_k_dev_ptr,
                                  param.seqlen_k_dev_ptr,
                                  param.K,  // hdim_q
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
                                  bias);

        (void)launch_kernel<kBlockSize.x, kBlockPerCu>(
            StreamConfig{stream, false}, FmhaKernel{}, kGridSize, kBlockSize, 0, kargs);
    };
};

template <typename scalar_t, int32_t custom_mask_type>
void run_grouped_infer_masktype_attnbias_dispatched(GroupedForwardParams& param, hipStream_t stream)
{
    grouped_infer_masktype_attnbias_dispatched<scalar_t, custom_mask_type>::Run(param, stream);
};
