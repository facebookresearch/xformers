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
#include <ck/tile_program/block_tile_pipeline/block_fmha_bwd_pipeline_problem.hpp>
#include <ck/tile_program/tile/tile_fmha_shape.hpp>
#include <ck/tile_program/tile/tile_fmha_traits.hpp>

#include "ck_tiled_bool_switch.h"
#include "ck_tiled_fmha_bwd_setting.h"
#include "ck_tiled_fmha_params.h"

#include "fmha_bwd_epilogue.hpp"
#include "fmha_bwd_kernel.hpp"
#include "fmha_bwd_tile_partitioner.hpp"

template <
    typename ScalarType,
    bool kHasCausalMask,
    bool kHasBias,
    bool kHasBiasGrad,
    bool kHasDropout,
    ck::index_t MaxK>
struct batched_backward_causalmask_bias_dropout_dispatch {
  using FmhaBwdEpilogue_ = FmhaBwdEpilogue<FmhaBwdEpilogueProblem<
      typename FmhaBwdTypeConfig<ScalarType>::AccDataType,
      typename FmhaBwdTypeConfig<ScalarType>::KGradDataType,
      typename FmhaBwdTypeConfig<ScalarType>::VGradDataType>>;

  template <typename FmhaTraits, typename FmhaMask>
  using FmhaBwdPipelineProblemTemp =
      ck::tile_program::block::BlockFmhaBwdPipelineProblem<
          typename FmhaBwdTypeConfig<ScalarType>::QDataType,
          typename FmhaBwdTypeConfig<ScalarType>::KDataType,
          typename FmhaBwdTypeConfig<ScalarType>::VDataType,
          typename FmhaBwdTypeConfig<ScalarType>::GemmDataType,
          typename FmhaBwdTypeConfig<ScalarType>::LSEDataType,
          typename FmhaBwdTypeConfig<ScalarType>::AccDataType,
          typename FmhaBwdTypeConfig<ScalarType>::DDataType,
          typename FmhaBwdTypeConfig<ScalarType>::BiasDataType,
          typename FmhaBwdTypeConfig<ScalarType>::RandValOutputDataType,
          typename FmhaBwdTypeConfig<ScalarType>::ODataType,
          typename FmhaBwdTypeConfig<ScalarType>::OGradDataType,
          typename FmhaBwdTypeConfig<ScalarType>::QGradDataType,
          typename FmhaBwdTypeConfig<ScalarType>::KGradDataType,
          typename FmhaBwdTypeConfig<ScalarType>::VGradDataType,
          typename FmhaBwdTypeConfig<ScalarType>::BiasGradDataType,
          FmhaBwdShape<MaxK>,
          false, // kIsGroupMode
          FmhaMask,
          FmhaTraits>;

  static void Run(BatchedBackwardParams& param, hipStream_t stream) {
    {
      constexpr ck::index_t kBlockSize = 256;

      const bool pad_seqlen_q = !(param.M % kBlockSize == 0);
      const bool pad_headdim_v =
          !(param.Kv % FmhaBwdShape<MaxK>::kVHeaddim == 0);

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
                    typename FmhaBwdTypeConfig<ScalarType>::ODataType,
                    typename FmhaBwdTypeConfig<ScalarType>::OGradDataType,
                    typename FmhaBwdTypeConfig<ScalarType>::DDataType,
                    kBlockSize,
                    FmhaBwdShape<MaxK>::kVHeaddim,
                    false, // kIsGroupMode
                    FmhaOGradDotOTraits_>;

            using FmhaBwdOGradDotOPipeline =
                typename ck::tile_program::block::BlockFmhaBwdOGradDotO<
                    FmhaBwdOGradDotOPipelineProblem>;

            using FmhaBwdOGradDotOKernel_ = FmhaBwdOGradDotOKernel<
                FmhaBwdOGradDotOTilePartitioner<kBlockSize>,
                FmhaBwdOGradDotOPipeline>;

            RunWithBwdOGradDotOKernel<FmhaBwdOGradDotOKernel_>(param, stream);
          });
    }

    {
      const bool has_local_attention = (param.window_size > 0) ? true : false;

      BOOL_SWITCH(has_local_attention, USE_LOCAL_ATTENTION, [&] {
        constexpr ck::index_t occupancy = 1;
        constexpr bool has_masking = kHasCausalMask || USE_LOCAL_ATTENTION;

        using FmhaMask =
            ck::tile_program::block::SimplifiedGenericAttentionMask<
                has_masking>;

        using FmhaBwdShape_ = FmhaBwdShape<MaxK>;
        using FmhaBwdTilePartitioner_ = FmhaBwdTilePartitioner<FmhaBwdShape_>;

        constexpr bool kPadSeqLenQ = true;
        constexpr bool kPadSeqLenK = true;

        const bool pad_headdim_q = !(param.K % FmhaBwdShape_::kQKHeaddim == 0);
        const bool pad_headdim_v = !(param.Kv % FmhaBwdShape_::kVHeaddim == 0);

        // usually headdim_q and headdim_v are same, consider them together
        // to determine whether to do padding saving some compiling time
        const bool pad_headdim = (pad_headdim_q || pad_headdim_v);

        BOOL_SWITCH(pad_headdim, kPadHeadDim, [&] {
          using FmhaBwdTraits_ = ck::tile_program::TileFmhaTraits<
              kPadSeqLenQ,
              kPadSeqLenK,
              kPadHeadDim, // kPadHeadDimQ,
              kPadHeadDim, // kPadHeadDimV,
              kHasBias,
              kHasBiasGrad,
              false, // kStoreLSE
              kHasDropout,
              occupancy>;

          using FmhaBwdPipelineProblem =
              FmhaBwdPipelineProblemTemp<FmhaBwdTraits_, FmhaMask>;

          constexpr auto FmhaBwdPipelineEnum_ =
              FmhaBwdPipelineEnumSelector<MaxK>::value;

          using FmhaBwdPipeline_ = typename FmhaBwdPipelineMaker<
              FmhaBwdPipelineEnum_,
              FmhaBwdPipelineProblem>::pipeline;

          using FmhaBwdDQDKDVKernel_ = FmhaBwdDQDKDVKernel<
              FmhaBwdTilePartitioner_,
              FmhaBwdPipeline_,
              FmhaBwdEpilogue_>;

          RunWithBwdDQDKDVKernel<FmhaBwdDQDKDVKernel_>(param, stream);
        });
      });
    };
  }

  template <typename FmhaBwdOGradDotOKernel>
  static void RunWithBwdOGradDotOKernel(
      BatchedBackwardParams& param,
      hipStream_t stream) {
    const auto kargs = [&] {
      return FmhaBwdOGradDotOKernel::MakeKargs(
          param.out_ptr,
          param.grad_out_ptr,
          param.dot_out_ptr,
          1.0f - param.dropout_prob,
          param.M,
          param.Kv,
          param.grad_out_strides[1], // stride_do
          param.out_strides[1], // stride_o
          param.grad_out_strides[2], // nhead_stride_do
          param.out_strides[2], // nhead_stride_o
          param.lsed_strides[1], // nhead_stride_d
          param.grad_out_strides[0], // batch_stride_do
          param.out_strides[0], // batch_stride_o
          param.lsed_strides[0]); // batch_stride_d
    }();

    dim3 kGridSize =
        FmhaBwdOGradDotOKernel::GridSize(param.B, param.Hq, param.M);
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

  template <typename FmhaBwdDQDKDVKernel>
  static void RunWithBwdDQDKDVKernel(
      BatchedBackwardParams& param,
      hipStream_t stream) {
    const auto kargs = [&] {
      return FmhaBwdDQDKDVKernel::MakeKargs(
          param.q_ptr,
          param.k_ptr,
          param.v_ptr,
          param.attn_bias_ptr,
          param.logsumexp_ptr,
          param.grad_out_ptr,
          param.dot_out_ptr,
          nullptr, // rand_val_ptr
          param.grad_q_ptr,
          param.grad_k_ptr,
          param.grad_v_ptr,
          param.grad_bias_ptr,
          param.M, // seqlen_q
          param.N, // seqlen_k
          param.K,
          param.Kv,
          param.Hq,
          param.Hq / param.Hkv,
          param.scale,
          param.q_strides[1], // q, k, v, bias, do, dk, dv, dbias seq-dim
                              // stride
          param.k_strides[1],
          param.v_strides[1],
          param.attn_bias_strides[2],
          0, // stride_randval
          param.grad_out_strides[1],
          param.grad_k_strides[1],
          param.grad_v_strides[1],
          param.attn_bias_strides[2], // assume grad_bias has same strides as
                                      // bias
          param.q_strides[2], // q, k, v, bias, do, lse/dot, dbias
                              // nhead-dim strides
          param.k_strides[2],
          param.v_strides[2],
          param.attn_bias_strides[1],
          0, // nhead_stride_randval
          param.grad_out_strides[2],
          param.lsed_strides[1],
          param.attn_bias_strides[1], // assume grad_bias has same strides as
                                      // bias
          param.q_strides[0], // q, k, v, bias, do, lse/dot, dk, dv, dbias,
                              // batch-dim strides
          param.k_strides[0],
          param.v_strides[0],
          param.attn_bias_strides[0],
          0, // batch_stride_randval
          param.grad_out_strides[0],
          param.lsed_strides[0], // lse/dot is in BHM contiguous layout
          param.grad_k_strides[0],
          param.grad_v_strides[0],
          param.attn_bias_strides[0], // assume grad_bias has same strides as
                                      // bias
          (param.window_size > 0) ? param.window_size - 1
                                  : -1, // window_left_size
          (param.custom_mask_type == 0) ? -1 : 0, // window_right_size
          param.custom_mask_type,
          param.dropout_prob, // dropout ratio
          false, // is_store_randval
          {param.philox_seed, param.philox_offset});
    }();

    dim3 kGridSize = FmhaBwdDQDKDVKernel::GridSize(param.B, param.Hq, param.N);
    constexpr dim3 kBlockSize = FmhaBwdDQDKDVKernel::BlockSize();
    constexpr ck::index_t kBlockPerCu = FmhaBwdDQDKDVKernel::kBlockPerCu;

    (void)launch_kernel<kBlockSize.x, kBlockPerCu>(
        StreamConfig{stream, false},
        FmhaBwdDQDKDVKernel{},
        kGridSize,
        kBlockSize,
        0,
        kargs);
  }
};

template <
    typename ScalarType,
    bool kHasCausalMask,
    bool kHasBias,
    bool kHasBiasGrad,
    bool kHasDropout,
    ck::index_t MaxK>
void run_batched_backward_causalmask_bias_dropout_dispatch(
    BatchedBackwardParams& param,
    hipStream_t stream) {
  batched_backward_causalmask_bias_dropout_dispatch<
      ScalarType,
      kHasCausalMask,
      kHasBias,
      kHasBiasGrad,
      kHasDropout,
      MaxK>::Run(param, stream);
};
