/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ck/host_utility/device_prop.hpp>
#include <ck/host_utility/kernel_launch.hpp>
#include <ck/tensor/tensor_view.hpp>
#include <ck/tensor_description/cluster_descriptor.hpp>
#include <ck/tensor_description/tensor_descriptor_helper.hpp>
#include <ck/utility/common_header.hpp>

#include <ck/tile_program/block_tile/block_masking.hpp>
#include <ck/tile_program/block_tile_pipeline/block_fmha_bwd_dot_do_o.hpp>
#include <ck/tile_program/block_tile_pipeline/block_fmha_bwd_pipeline_dispatcher.hpp>
#include <ck/tile_program/block_tile_pipeline/block_fmha_bwd_pipeline_problem.hpp>
#include <ck/tile_program/tile/tile_fmha_shape.hpp>
#include <ck/tile_program/tile/tile_fmha_traits.hpp>

#include "ck_tiled_bool_switch.h"
#include "ck_tiled_fmha_bwd_setting.h"
#include "ck_tiled_fmha_params.h"

#include "ck_tiled_fmha_backward_kernel.hpp"
#include "ck_tiled_fmha_bwd_epilogue.hpp"
#include "ck_tiled_fmha_bwd_tile_partitioner.hpp"
#include "ck_tiled_fmha_definitions.hpp"

template <
    typename scalar_t,
    bool has_causal_mask,
    bool has_attn_bias,
    ck::index_t MaxK>
struct grouped_backward_causalmask_attnbias_dispatched {
  using FmhaBwdEpilogue_ = FmhaBwdEpilogue<FmhaBwdEpilogueProblem<
      typename FmhaBwdTypeConfig<scalar_t>::AccDataType,
      typename FmhaBwdTypeConfig<scalar_t>::KGradDataType,
      typename FmhaBwdTypeConfig<scalar_t>::VGradDataType>>;

  using FmhaBwdLoadStrategy_ = typename FmhaBwdLoadStrategy<MaxK>::type;

  template <typename FmhaTraits, typename FmhaMask>
  using FmhaBwdPipelineProblemTemp =
      ck::tile_program::block::BlockFmhaBwdPipelineProblem<
          typename FmhaBwdTypeConfig<scalar_t>::QDataType,
          typename FmhaBwdTypeConfig<scalar_t>::KDataType,
          typename FmhaBwdTypeConfig<scalar_t>::VDataType,
          typename FmhaBwdTypeConfig<scalar_t>::GemmDataType,
          typename FmhaBwdTypeConfig<scalar_t>::LSEDataType,
          typename FmhaBwdTypeConfig<scalar_t>::AccDataType,
          typename FmhaBwdTypeConfig<scalar_t>::DDataType,
          typename FmhaBwdTypeConfig<scalar_t>::BiasDataType,
          typename FmhaBwdTypeConfig<scalar_t>::RandValOutputDataType,
          typename FmhaBwdTypeConfig<scalar_t>::ODataType,
          typename FmhaBwdTypeConfig<scalar_t>::OGradDataType,
          typename FmhaBwdTypeConfig<scalar_t>::QGradDataType,
          typename FmhaBwdTypeConfig<scalar_t>::KGradDataType,
          typename FmhaBwdTypeConfig<scalar_t>::VGradDataType,
          typename FmhaBwdTypeConfig<scalar_t>::BiasGradDataType,
          FmhaBwdShape<MaxK>,
          true, // kIsGroupMode
          FmhaMask,
          FmhaTraits>;

  static void Run(GroupedBackwardParams& param, hipStream_t stream) {
    {
      constexpr ck::index_t kBlockSize = 256;
      bool pad_seqlen_q = !(param.M % kBlockSize == 0);
      bool pad_headdim_v = !(param.Kv % FmhaBwdShape<MaxK>::kVHeaddim == 0);

      BOOL_SWITCH_2(
          pad_seqlen_q, kPadSeqLenQ, pad_headdim_v, kPadHeadDimV, [&] {
            constexpr ck::index_t occupancy = 2;

            using FmhaOGradDotOTraits_ =
                ck::tile_program::TileFmhaBwdOGradDotOTraits<
                    kPadSeqLenQ,
                    kPadHeadDimV,
                    occupancy>;

            using FmhaBwdOGradDotOPipelineProblem =
                ck::tile_program::block::BlockFmhaBwdOGradDotOPipelineProblem<
                    typename FmhaBwdTypeConfig<scalar_t>::ODataType,
                    typename FmhaBwdTypeConfig<scalar_t>::OGradDataType,
                    typename FmhaBwdTypeConfig<scalar_t>::DDataType,
                    kBlockSize,
                    FmhaBwdShape<MaxK>::kVHeaddim,
                    true, // kIsGroupMode
                    FmhaOGradDotOTraits_>;

            using FmhaBwdOGradDotOPipeline_ =
                typename ck::tile_program::block::BlockFmhaBwdOGradDotO<
                    FmhaBwdOGradDotOPipelineProblem>;

            using FmhaBwdOGradDotOKernel_ = FmhaBwdOGradDotOKernel<
                FmhaBwdOGradDotOTilePartitioner<kBlockSize>,
                FmhaBwdOGradDotOPipeline_>;

            RunWithBwdOGradDotOKernel<FmhaBwdOGradDotOKernel_>(param, stream);
          });
    };

    {
      const bool has_local_attention = (param.window_size > 0) ? true : false;

      BOOL_SWITCH(has_local_attention, USE_LOCAL_ATTENTION, [&] {
        constexpr ck::index_t occupancy = 1;
        constexpr bool has_masking = has_causal_mask || USE_LOCAL_ATTENTION;
        const bool has_dropout = (param.dropout_prob > 0.0f);

        using FmhaMask = ck::tile_program::block::
            GenericAttentionMask<has_masking, USE_LOCAL_ATTENTION>;

        using FmhaBwdShape_ = FmhaBwdShape<MaxK>;
        using FmhaBwdTilePartitioner_ = FmhaBwdTilePartitioner<FmhaBwdShape_>;

        constexpr bool kPadSeqLenQ = true;
        constexpr bool kPadSeqLenK = true;

        const bool pad_headdim_q = !(param.K % FmhaBwdShape_::kQKHeaddim == 0);
        const bool pad_headdim_v = !(param.Kv % FmhaBwdShape_::kVHeaddim == 0);

        // usually headdim_q and headdim_v are same, consider them together
        // to determine whether to do padding saving some compiling time
        const bool pad_headdim = (pad_headdim_q || pad_headdim_v);

        BOOL_SWITCH_2(has_dropout, kHasDropout, pad_headdim, kPadHeadDim, [&] {
          using FmhaBwdTraits_ = ck::tile_program::TileFmhaTraits<
              kPadSeqLenQ,
              kPadSeqLenK,
              kPadHeadDim, // kPadHeadDimQ,
              kPadHeadDim, // kPadHeadDimV,
              has_attn_bias,
              false, // kStoreLSE
              kHasDropout,
              occupancy>;

          using FmhaBwdPipelineProblem =
              FmhaBwdPipelineProblemTemp<FmhaBwdTraits_, FmhaMask>;

          using FmhaBwdPipeline_ =
              typename ck::tile_program::block::BlockFmhaBwdPipelineDispatcher<
                  FmhaBwdLoadStrategy_,
                  FmhaBwdPipelineProblem>::BlockPipeline;

          using FmhaBwdKernel_ = FmhaBwdKernel<
              FmhaBwdTilePartitioner_,
              FmhaBwdPipeline_,
              FmhaBwdEpilogue_>;

          RunWithBwdKernel<FmhaBwdKernel_>(param, stream);
        });
      });
    };
  }

  template <typename FmhaBwdOGradDotOKernel>
  static void RunWithBwdOGradDotOKernel(
      GroupedBackwardParams& param,
      hipStream_t stream) {
    const auto kargs = [&] {
      return FmhaBwdOGradDotOKernel::MakeKargs(
          param.out_ptr,
          param.grad_out_ptr,
          param.dot_out_ptr,
          1.0f - param.dropout_prob,
          param.seqstart_q_dev_ptr,
          param.Kv,
          param.grad_out_strides[0], // stride_do
          param.out_strides[0], // stride_o
          param.grad_out_strides[1], // nhead_stride_do
          param.out_strides[1], // nhead_stride_o
          param.lsed_strides[1],
          param.lsed_strides[0], // batch_stride_d
          param.grad_out_strides[2]); // hdim_stride_do
    }();

    dim3 kGridSize = FmhaBwdOGradDotOKernel::GridSize(
        param.num_batches, param.Hq, param.max_seqlen_q);
    constexpr dim3 kBlockSize = FmhaBwdOGradDotOKernel::BlockSize();
    constexpr ck::index_t kBlockPerCu = FmhaBwdOGradDotOKernel::kBlockPerCu;

    (void)launch_kernel<kBlockSize.x, kBlockPerCu>(
        StreamConfig{stream, false},
        FmhaBwdOGradDotOKernel{},
        kGridSize,
        kBlockSize,
        0,
        kargs);
  }

  template <typename FmhaBwdKernel>
  static void RunWithBwdKernel(
      GroupedBackwardParams& param,
      hipStream_t stream) {
    const auto kargs = [&] {
      return FmhaBwdKernel::MakeKargs(
          param.q_ptr,
          param.k_ptr,
          param.v_ptr,
          param.attn_bias_ptr,
          param.logsumexp_ptr,
          param.grad_out_ptr,
          param.dot_out_ptr,
          nullptr, // randval_ptr
          param.grad_q_ptr,
          param.grad_k_ptr,
          param.grad_v_ptr,
          param.grad_bias_ptr,
          param.seqstart_q_dev_ptr,
          param.seqstart_k_dev_ptr,
          param.seqlen_k_dev_ptr,
          param.K,
          param.Kv,
          param.Hq,
          param.Hq / param.Hkv,
          param.scale,
          param.q_strides[0], // q, k, v, bias, do, dk, dv, dbias seq-dim
                              // stride
          param.k_strides[0],
          param.v_strides[0],
          param.attn_bias_strides[1],
          0, // stride_randval
          param.grad_out_strides[0],
          param.grad_k_strides[0],
          param.grad_v_strides[0],
          param.attn_bias_strides[1], // assume grad_bias has same strides as
                                      // bias
          param.q_strides[1], // q, k, v, bias, do, lse/dot, dbias
                              // nhead-dim strides
          param.k_strides[1],
          param.v_strides[1],
          param.attn_bias_strides[0],
          0, // nhead_stride_randval
          param.grad_out_strides[1],
          param.lsed_strides[1], // assume lse/dot is in BHM contiguous layout
          param.attn_bias_strides[0], // assume grad_bias has same strides as
                                      // bias
          param.lsed_strides[0], // batch_stride_lse
          param.grad_out_strides[2], // hdim_stride_do
          static_cast<CausalMaskType>(param.custom_mask_type),
          param.window_size,
          param.dropout_prob, // dropout ratio
          false, // is_store_randval
          {param.philox_seed, param.philox_offset});
    }();

    dim3 kGridSize = FmhaBwdKernel::GridSize(
        param.num_batches, param.Hq, param.max_seqlen_k);
    constexpr dim3 kBlockSize = FmhaBwdKernel::BlockSize();
    constexpr ck::index_t kBlockPerCu = FmhaBwdKernel::kBlockPerCu;

    (void)launch_kernel<kBlockSize.x, kBlockPerCu>(
        StreamConfig{stream, false},
        FmhaBwdKernel{},
        kGridSize,
        kBlockSize,
        0,
        kargs);
  }
};

template <
    typename scalar_t,
    bool has_causal_mask,
    bool has_attn_bias,
    ck::index_t MaxK>
void run_grouped_backward_causalmask_attnbias_dispatched(
    GroupedBackwardParams& param,
    hipStream_t stream) {
  grouped_backward_causalmask_attnbias_dispatched<
      scalar_t,
      has_causal_mask,
      has_attn_bias,
      MaxK>::Run(param, stream);
};
