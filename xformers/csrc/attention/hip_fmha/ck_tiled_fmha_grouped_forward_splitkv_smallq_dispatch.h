/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ck_tile/core/numeric/integer.hpp>
#include <ck_tile/host/kernel_launch.hpp>
#include <ck_tile/host/stream_config.hpp>
#include <ck_tile/ops/epilogue.hpp>
#include <ck_tile/ops/fmha.hpp>

#include "ck_tiled_bool_switch.h"
#include "ck_tiled_fmha_fwd_splitkv_smallq_setting.h"
#include "ck_tiled_fmha_num_kv_split_switch.h"
#include "ck_tiled_fmha_params.h"

template <
    typename ScalarType,
    bool kHasMask,
    bool kHasBias,
    ck_tile::index_t MaxK>
struct grouped_forward_splitkv_smallq_mask_bias_dropout_dispatch {
  template <typename FmhaTraits>
  using AttentionVariant = ck_tile::ComposedAttention<
      FmhaTraits::kHasLogitsSoftCap * ck_tile::LOGITS_SOFT_CAP,
      CK_TILE_FMHA_FWD_FAST_EXP2>;
  template <
      typename FmhaFwdSplitKVTraits,
      typename FmhaMask,
      typename ODataType>
  using FmhaFwdSplitKVPipelineProblemTemp =
      ck_tile::BlockFmhaFwdSplitKVPipelineProblem<
          typename FmhaFwdTypeConfig<ScalarType>::QDataType,
          typename FmhaFwdTypeConfig<ScalarType>::KDataType,
          typename FmhaFwdTypeConfig<ScalarType>::VDataType,
          typename FmhaFwdTypeConfig<ScalarType>::SaccDataType,
          typename FmhaFwdTypeConfig<ScalarType>::SMPLComputeDataType,
          typename FmhaFwdTypeConfig<ScalarType>::BiasDataType,
          typename FmhaFwdTypeConfig<ScalarType>::LSEDataType,
          typename FmhaFwdTypeConfig<ScalarType>::PDataType,
          typename FmhaFwdTypeConfig<ScalarType>::OaccDataType,
          ODataType,
          typename FmhaFwdSplitKVSmallQShape<MaxK>::Type,
          true, // kIsGroupMode
          AttentionVariant<FmhaFwdSplitKVTraits>,
          FmhaMask,
          FmhaFwdSplitKVTraits>;

  template <ck_tile::index_t kN1, typename FmhaSplitKVCombineTraits>
  using FmhaSplitKVCombinePipelineProblemTemp =
      ck_tile::BlockFmhaSplitKVCombinePipelineProblem<
          typename FmhaFwdTypeConfig<ScalarType>::LSEDataType,
          typename FmhaFwdTypeConfig<ScalarType>::OaccDataType,
          typename FmhaFwdTypeConfig<ScalarType>::ODataType,
          MaxK, // headdim_v
          true, // kIsGroupMode
          kN1,
          FmhaSplitKVCombineTraits>;

  static void Run(GroupedForwardParams& param, hipStream_t stream) {
    {
      using FmhaMask = ck_tile::SimplifiedGenericAttentionMask<kHasMask>;

      using FmhaTileShape = typename FmhaFwdSplitKVSmallQShape<MaxK>::Type;

      constexpr ck_tile::index_t occupancy = -1;

      constexpr auto kBiasEnum = kHasBias
          ? ck_tile::BlockAttentionBiasEnum::ELEMENTWISE_BIAS
          : ck_tile::BlockAttentionBiasEnum::NO_BIAS;

      constexpr bool kPadSeqLenQ = true;
      constexpr bool kPadSeqLenK = true;

      const bool pad_headdim_q = !(param.K % FmhaTileShape::kSubQKHeaddim == 0);
      const bool pad_headdim_v = !(param.Kv % FmhaTileShape::kN1 == 0);

      BOOL_SWITCH_2(
          pad_headdim_q, kPadHeadDimQ, pad_headdim_v, kPadHeadDimV, [&] {
            using FmhaTraits = ck_tile::TileFmhaFwdSplitKVTraits<
                kPadSeqLenQ,
                kPadSeqLenK,
                kPadHeadDimQ,
                kPadHeadDimV,
                false, // kHasLogitsSoftCap
                kBiasEnum,
                false, // kHasBiasGrad place-holder
                true, // kStoreLSE
                false, // kDoFp8StaticQuant place-holder
                false, // kIsPagedKV
                true, // kHasUnevenSplits
                false, // kMergeNumHeadGroupsSeqLenQ
                occupancy>;

            if (param.num_kv_splits > 1) {
              using ODataType =
                  typename FmhaFwdTypeConfig<ScalarType>::OaccDataType;
              using FmhaPipelineProblem = FmhaFwdSplitKVPipelineProblemTemp<
                  FmhaTraits,
                  FmhaMask,
                  ODataType>;

              using FmhaFwdPipeline_ =
                  ck_tile::BlockFmhaFwdSplitKVPipelineNWarpSShuffleQRKSVS<
                      FmhaPipelineProblem>;

              using FmhaFwdEpilogue_ =
                  ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
                      typename FmhaFwdTypeConfig<ScalarType>::OaccDataType,
                      ODataType,
                      false,
                      false>>;

              using FmhaFwdKernel_ = ck_tile::
                  FmhaFwdSplitKVKernel<FmhaFwdPipeline_, FmhaFwdEpilogue_>;

              RunWithFwdSplitKVKernel<FmhaFwdKernel_>(param, stream);
            } else {
              using ODataType =
                  typename FmhaFwdTypeConfig<ScalarType>::ODataType;
              using FmhaPipelineProblem = FmhaFwdSplitKVPipelineProblemTemp<
                  FmhaTraits,
                  FmhaMask,
                  ODataType>;

              using FmhaFwdPipeline_ =
                  ck_tile::BlockFmhaFwdSplitKVPipelineNWarpSShuffleQRKSVS<
                      FmhaPipelineProblem>;

              using FmhaFwdEpilogue_ =
                  ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
                      typename FmhaFwdTypeConfig<ScalarType>::OaccDataType,
                      ODataType,
                      false,
                      false>>;

              using FmhaFwdKernel_ = ck_tile::
                  FmhaFwdSplitKVKernel<FmhaFwdPipeline_, FmhaFwdEpilogue_>;

              RunWithFwdSplitKVKernel<FmhaFwdKernel_>(param, stream);
            }
          });
    };

    if (param.num_kv_splits > 1) {
      using FmhaTileShape = typename FmhaFwdSplitKVSmallQShape<MaxK>::Type;

      constexpr ck_tile::index_t kN1 = 32;
      constexpr ck_tile::index_t kM0 =
          ck_tile::BlockFmhaSplitKVCombinePipelineTileSizes<
              typename FmhaFwdTypeConfig<ScalarType>::OaccDataType,
              kN1>::kM0;

      constexpr ck_tile::index_t occupancy = -1;

      constexpr bool kPadSeqLenQ = true;

      const bool pad_headdim_v = !(param.Kv % kN1 == 0);

      BOOL_SWITCH(pad_headdim_v, kPadHeadDimV, [&] {
        FMHA_FWD_NUM_KV_SPLITS_SWITCH(param.num_kv_splits, kLogMaxSplits, [&] {
          using FmhaTraits = ck_tile::TileFmhaFwdSplitKVCombineTraits<
              kPadSeqLenQ,
              kPadHeadDimV,
              true, // kStoreLSE
              false, // kDoFp8StaticQuant place-holder
              kLogMaxSplits,
              -1>;

          using FmhaPipelineProblem =
              FmhaSplitKVCombinePipelineProblemTemp<kN1, FmhaTraits>;

          using FmhaPipeline =
              ck_tile::BlockFmhaFwdSplitKVCombinePipeline<FmhaPipelineProblem>;

          using FmhaEpilogue =
              ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
                  typename FmhaFwdTypeConfig<ScalarType>::OaccDataType,
                  typename FmhaFwdTypeConfig<ScalarType>::ODataType,
                  kPadSeqLenQ,
                  kPadHeadDimV>>;

          using FmhaKernel =
              ck_tile::FmhaFwdSplitKVCombineKernel<FmhaPipeline, FmhaEpilogue>;

          RunWithSplitKVCombineKernel<FmhaKernel>(param, stream);
        });
      });
    };
  };

  template <typename FmhaFwdSplitKVKernel>
  static void RunWithFwdSplitKVKernel(
      GroupedForwardParams& param,
      hipStream_t stream) {
    const auto kargs = [&] {
      if (param.num_kv_splits > 1)
        return FmhaFwdSplitKVKernel::MakeKargs(
            param.q_ptr,
            param.k_ptr,
            param.v_ptr,
            param.attn_bias_ptr,
            param.logsumexp_acc_ptr,
            param.out_acc_ptr,
            param.num_batches,
            param.seqstart_q_dev_ptr,
            param.seqstart_k_dev_ptr,
            param.seqlen_k_dev_ptr,
            param.K, // hdim_q
            param.Kv, // hdim_v
            param.Hq, // nhead_q
            param.Hq / param.Hkv, // nhead_ratio_qk
            param.num_kv_splits, // num_splits
            nullptr, // block_table_ptr
            0, // batch_stride_block_table
            0, // page_block_size
            false, // is_gappy
            param.scale,
            1.0f, // scale_pz
            0.f, // logits_soft_cap
            param.q_strides[0], // q, k, v, bias, out_acc tensor seq-dim
                                // stride
            param.k_strides[0],
            param.v_strides[0],
            param.attn_bias_strides[2],
            param.out_acc_strides[1],
            param.q_strides[1], // q, k, v, bias, lse_acc, out_acc tensor
                                // head-dim stride
            param.k_strides[1],
            param.v_strides[1],
            param.attn_bias_strides[1],
            param.lse_acc_strides[1],
            param.out_acc_strides[2],
            0, // batch_stride_k, not used, only used for paged-kvcache
            0, // batch_stride_v, not used, only used for paged-kvcache
            param.lse_acc_strides[0], // split_stride_lse_acc
            param.out_acc_strides[0], // split_stride_out_acc
            (param.window_size > 0) ? param.window_size - 1
                                    : -1, // window_left_size
            (param.custom_mask_type == 0) ? -1 : 0, // window_right_size
            param.custom_mask_type);
      else
        return FmhaFwdSplitKVKernel::MakeKargs(
            param.q_ptr,
            param.k_ptr,
            param.v_ptr,
            param.attn_bias_ptr,
            param.logsumexp_ptr,
            param.out_ptr,
            param.num_batches,
            param.seqstart_q_dev_ptr,
            param.seqstart_k_dev_ptr,
            param.seqlen_k_dev_ptr,
            param.K, // hdim_q
            param.Kv, // hdim_v
            param.Hq, // nhead_q
            param.Hq / param.Hkv, // nhead_ratio_qk
            param.num_kv_splits, // num_splits
            nullptr, // block_table_ptr
            0, // batch_stride_block_table
            0, // page_block_size
            false, // is_gappy
            param.scale,
            1.0f, // scale_p
            0.0f, // logits_soft_cap
            param.q_strides[0], // q, k, v, bias, out tensor seq-dim stride
            param.k_strides[0],
            param.v_strides[0],
            param.attn_bias_strides[2],
            param.out_strides[0],
            param.q_strides[1], // q, k, v, bias, lse, out tensor head-dim
                                // stride
            param.k_strides[1],
            param.v_strides[1],
            param.attn_bias_strides[1],
            param.lse_strides[0],
            param.out_strides[1],
            0, // batch_stride_k, not used, only used for paged-kvcache
            0, // batch_stride_v, not used, only used for paged-kvcache
            0, // split_stride_lse_acc
            0, // split_stride_out_acc
            (param.window_size > 0) ? param.window_size - 1
                                    : -1, // window_left_size
            (param.custom_mask_type == 0) ? -1 : 0, // window_right_size
            param.custom_mask_type);
    }();

    dim3 kGridSize = FmhaFwdSplitKVKernel::GridSize(
        param.num_batches,
        param.Hq,
        param.Hkv,
        param.max_seqlen_q,
        param.Kv,
        param.num_kv_splits);
    constexpr dim3 kBlockSize = FmhaFwdSplitKVKernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = FmhaFwdSplitKVKernel::kBlockPerCu;

    (void)ck_tile::launch_kernel(
        ck_tile::stream_config{stream, false},
        ck_tile::make_kernel<kBlockSize.x, kBlockPerCu>(
            FmhaFwdSplitKVKernel{}, kGridSize, kBlockSize, 0, kargs));
  };

  template <typename FmhaSplitKVCombineKernel>
  static void RunWithSplitKVCombineKernel(
      GroupedForwardParams& param,
      hipStream_t stream) {
    const auto kargs = [&] {
      return FmhaSplitKVCombineKernel::MakeKargs(
          param.logsumexp_acc_ptr,
          param.out_acc_ptr,
          param.logsumexp_ptr,
          param.out_ptr,
          param.num_batches,
          param.seqstart_q_dev_ptr,
          param.Kv,
          param.num_kv_splits,
          1.0f,
          param.out_acc_strides[1], // row_stride_o_acc,
          param.out_strides[0], // row_stride_o,
          param.lse_acc_strides[1], // nhead_stride_lse_acc
          param.out_acc_strides[2], // nhead_stride_o_acc,
          param.lse_strides[0], // nhead_stride_lse,
          param.out_strides[1], // nhead_stride_o,
          param.lse_acc_strides[0], // split_stride_lse_acc,
          param.out_acc_strides[0]); // split_stride_o_acc
    }();

    dim3 kGridSize = FmhaSplitKVCombineKernel::GridSize(
        param.num_batches, param.Hq, param.max_seqlen_q, param.Kv);
    constexpr dim3 kBlockSize = FmhaSplitKVCombineKernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu =
        FmhaSplitKVCombineKernel::kBlockPerCu;

    (void)ck_tile::launch_kernel(
        ck_tile::stream_config{stream, false},
        ck_tile::make_kernel<kBlockSize.x, kBlockPerCu>(
            FmhaSplitKVCombineKernel{}, kGridSize, kBlockSize, 0, kargs));
  };
};
