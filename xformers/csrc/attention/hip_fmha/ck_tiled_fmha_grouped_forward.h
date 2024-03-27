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
#include <ck/tile_program/block_tile_pipeline/block_fmha_pipeline_problem.hpp>
#include <ck/tile_program/block_tile_pipeline/block_fmha_pipeline_qr_ks_vs.hpp>
#include <ck/tile_program/tile/tile_fmha_shape.hpp>
#include <ck/tile_program/tile/tile_fmha_traits.hpp>

#include "ck_tiled_bool_switch.h"
#include "ck_tiled_fmha_fwd_setting.h"
#include "ck_tiled_fmha_params.h"

#include "ck_tiled_fmha_definitions.hpp"
#include "ck_tiled_fmha_forward_kernel.hpp"
#include "ck_tiled_fmha_fwd_epilogue.hpp"
#include "ck_tiled_fmha_fwd_tile_partitioner.hpp"

template <
    typename scalar_t,
    bool has_causal_mask,
    bool has_attn_bias,
    ck::index_t MaxK>
struct grouped_forward_causalmask_attnbias_dispatched {
  using FmhaFwdEpilogue_ = FmhaFwdEpilogue<FmhaFwdEpilogueProblem<
      typename FmhaFwdTypeConfig<scalar_t>::OaccDataType,
      typename FmhaFwdTypeConfig<scalar_t>::ODataType>>;

  template <typename FmhaTraits, typename FmhaMask>
  using FmhaPipelineProblemTemp =
      ck::tile_program::block::BlockFmhaPipelineProblem<
          typename FmhaFwdTypeConfig<scalar_t>::QDataType,
          typename FmhaFwdTypeConfig<scalar_t>::KDataType,
          typename FmhaFwdTypeConfig<scalar_t>::VDataType,
          typename FmhaFwdTypeConfig<scalar_t>::SaccDataType,
          typename FmhaFwdTypeConfig<scalar_t>::SMPLComputeDataType,
          typename FmhaFwdTypeConfig<scalar_t>::BiasDataType,
          typename FmhaFwdTypeConfig<scalar_t>::RandValOutputDataType,
          typename FmhaFwdTypeConfig<scalar_t>::LSEDataType,
          typename FmhaFwdTypeConfig<scalar_t>::PDataType,
          typename FmhaFwdTypeConfig<scalar_t>::OaccDataType,
          typename FmhaFwdTypeConfig<scalar_t>::ODataType,
          FmhaFwdShape<MaxK>,
          true, // kIsGroupMode
          FmhaMask,
          FmhaTraits>;

  static void Run(GroupedForwardParams& param, hipStream_t stream) {
    const bool has_local_attention = (param.window_size > 0) ? true : false;

    BOOL_SWITCH(has_local_attention, USE_LOCAL_ATTENTION, [&] {
      constexpr bool has_masking = has_causal_mask || USE_LOCAL_ATTENTION;
      const bool has_dropout = (param.dropout_prob > 0.0f);

      using FmhaMask = ck::tile_program::block::
          GenericAttentionMask<has_masking, USE_LOCAL_ATTENTION>;

      using FmhaFwdShape_ = FmhaFwdShape<MaxK>;
      using FmhaFwdTilePartitioner_ = FmhaFwdTilePartitioner<FmhaFwdShape_>;
      constexpr ck::index_t occupancy =
          (MaxK == 64) ? 3 : (MaxK == 256) ? 1 : 2;

      constexpr bool kPadSeqLenQ = true;
      constexpr bool kPadSeqLenK = true;

      const bool pad_headdim_q =
          !(param.K % FmhaFwdShape_::kK0BlockLength == 0);
      const bool pad_headdim_v = !(param.Kv % FmhaFwdShape_::kN1 == 0);

      if constexpr (MaxK == 256) {
        BOOL_SWITCH_3(
            has_dropout,
            kHasDropout,
            pad_headdim_q,
            kPadHeadDimQ,
            pad_headdim_v,
            kPadHeadDimV,
            [&] {
              using FmhaFwdTraits_ = ck::tile_program::TileFmhaTraits<
                  kPadSeqLenQ,
                  kPadSeqLenK,
                  kPadHeadDimQ,
                  kPadHeadDimV,
                  has_attn_bias,
                  true, // kStoreLSE
                  kHasDropout,
                  occupancy>;

              using FmhaPipelineProblem =
                  FmhaPipelineProblemTemp<FmhaFwdTraits_, FmhaMask>;

              using FmhaFwdPipeline_ =
                  ck::tile_program::block::BlockFmhaPipelineQRKSVS<
                      FmhaPipelineProblem>;

              using FmhaFwdKernel_ = FmhaFwdKernel<
                  FmhaFwdTilePartitioner_,
                  FmhaFwdPipeline_,
                  FmhaFwdEpilogue_>;

              RunWithKernel<FmhaFwdKernel_>(param, stream);
            });
      } else {
        BOOL_SWITCH_3(
            has_dropout,
            kHasDropout,
            pad_headdim_q,
            kPadHeadDimQ,
            pad_headdim_v,
            kPadHeadDimV,
            [&] {
              using FmhaFwdTraits_ = ck::tile_program::TileFmhaTraits<
                  kPadSeqLenQ,
                  kPadSeqLenK,
                  kPadHeadDimQ,
                  kPadHeadDimV,
                  has_attn_bias,
                  true, // kStoreLSE
                  kHasDropout,
                  occupancy>;

              using FmhaPipelineProblem =
                  FmhaPipelineProblemTemp<FmhaFwdTraits_, FmhaMask>;

              using FmhaFwdPipeline_ =
                  ck::tile_program::block::BlockFmhaPipelineQRKSVS<
                      FmhaPipelineProblem>;
              using FmhaFwdKernel_ = FmhaFwdKernel<
                  FmhaFwdTilePartitioner_,
                  FmhaFwdPipeline_,
                  FmhaFwdEpilogue_>;

              RunWithKernel<FmhaFwdKernel_>(param, stream);
            });
      };
    });
  };

  template <typename FmhaFwdKernel>
  static void RunWithKernel(GroupedForwardParams& param, hipStream_t stream) {
    const auto kargs = [&] {
      return FmhaFwdKernel::MakeKargs(
          param.q_ptr,
          param.k_ptr,
          param.v_ptr,
          param.attn_bias_ptr,
          nullptr, // rand_val_ptr
          param.logsumexp_ptr,
          param.out_ptr,
          param.seqstart_q_dev_ptr,
          param.seqstart_k_dev_ptr,
          param.seqlen_k_dev_ptr,
          param.K, // hdim_q
          param.Kv, // hdim_v
          param.Hq, // nhead_q
          param.Hq / param.Hkv, // nhead_ratio_qk
          param.scale,
          param.q_strides[0], // q, k, v, bias, randval, out tensor seq-dim
                              // stride
          param.k_strides[0],
          param.v_strides[0],
          param.attn_bias_strides[2],
          0, // stride_randval
          param.out_strides[0],
          param.q_strides[1], // q, k, v, bias, randval, lse, out tensor
                              // head-dim stride
          param.k_strides[1],
          param.v_strides[1],
          param.attn_bias_strides[1],
          0, // nhead_stride_randval
          param.lse_strides[1],
          param.out_strides[1],
          static_cast<CausalMaskType>(param.custom_mask_type),
          param.window_size,
          1.0f, // descale_qk, not used
          1.0f, // descale_sv, not used
          param.use_dropout ? param.dropout_prob : 0.0f, // dropout ratio
          false, // is_store_randval
          {param.philox_seed, param.philox_offset});
    }();

    dim3 kGridSize = FmhaFwdKernel::GridSize(
        param.num_batches, param.Hq, param.max_seqlen_q, param.Kv);
    constexpr dim3 kBlockSize = FmhaFwdKernel::BlockSize();
    constexpr ck::index_t kBlockPerCu = FmhaFwdKernel::kBlockPerCu;

    (void)launch_kernel<kBlockSize.x, kBlockPerCu>(
        StreamConfig{stream, false},
        FmhaFwdKernel{},
        kGridSize,
        kBlockSize,
        0,
        kargs);
  };
};

template <
    typename scalar_t,
    bool has_causal_mask,
    bool has_attn_bias,
    ck::index_t MaxK>
void run_grouped_forward_causalmask_attnbias_dispatched(
    GroupedForwardParams& param,
    hipStream_t stream) {
  grouped_forward_causalmask_attnbias_dispatched<
      scalar_t,
      has_causal_mask,
      has_attn_bias,
      MaxK>::Run(param, stream);
};
