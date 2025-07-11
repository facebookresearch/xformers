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
#include "ck_tiled_fmha_fwd_setting.h"
#include "ck_tiled_fmha_params.h"

template <
    typename ScalarType,
    bool kHasMask,
    bool kHasBias,
    ck_tile::index_t MaxK,
    ck_tile::index_t MTile>
struct grouped_infer_pagedkv_mask_bias_dropout_dispatch {
  using fmha_variant = ck_tile::ComposedAttention<
      false * ck_tile::LOGITS_SOFT_CAP,
      CK_TILE_FMHA_FWD_FAST_EXP2>;

  using FmhaTileShape = typename FmhaFwdShape<MaxK, MTile>::Type;

  template <
      typename FmhaFwdPagedKVTraits,
      typename FmhaMask,
      typename ODataType>
  using FmhaFwdPagedKVPipelineProblemTemp =
      ck_tile::BlockFmhaFwdPagedKVPipelineProblem<
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
          FmhaTileShape,
          true, // kIsGroupMode
          fmha_variant,
          FmhaMask,
          FmhaFwdPagedKVTraits>;

  static void Run(GroupedForwardParams& param, hipStream_t stream) {
    {
      using FmhaMask = ck_tile::SimplifiedGenericAttentionMask<kHasMask>;

      constexpr ck_tile::index_t occupancy = -1;

      constexpr auto kBiasEnum = kHasBias
          ? ck_tile::BlockAttentionBiasEnum::ELEMENTWISE_BIAS
          : ck_tile::BlockAttentionBiasEnum::NO_BIAS;

      constexpr bool kPadSeqLenQ = true;
      constexpr bool kPadSeqLenK = true;

      bool pad_headdim_q = !(param.K % FmhaTileShape::kSubQKHeaddim == 0);
      bool pad_headdim_v = !(param.Kv % FmhaTileShape::kN1 == 0);

      BOOL_SWITCH_2(
          pad_headdim_q, kPadHeadDimQ, pad_headdim_v, kPadHeadDimV, [&] {
            using FmhaTraits = ck_tile::TileFmhaFwdPagedKVTraits<
                kPadSeqLenQ,
                kPadSeqLenK,
                kPadHeadDimQ,
                kPadHeadDimV,
                false, // kHasLogitsSoftCap_
                kBiasEnum,
                false, // kHasBiasGrad place-holder
                false, // kStoreLSE
                true, // kIsPagedKV
                false, // kDoFp8StaticQuant place-holder
                occupancy>;

            using ODataType = typename FmhaFwdTypeConfig<ScalarType>::ODataType;
            using FmhaPipelineProblem = FmhaFwdPagedKVPipelineProblemTemp<
                FmhaTraits,
                FmhaMask,
                ODataType>;

            using FmhaPipeline =
                ck_tile::BlockFmhaFwdPagedKVPipelineQRKSVS<FmhaPipelineProblem>;

            using FmhaEpilogue =
                ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
                    typename FmhaFwdTypeConfig<ScalarType>::OaccDataType,
                    ODataType,
                    false,
                    false>>;

            using FmhaKernel =
                ck_tile::FmhaFwdPagedKVKernel<FmhaPipeline, FmhaEpilogue>;

            RunWithFwdPagedKVKernel<FmhaKernel>(param, stream);
          });
    };
  };

  template <typename FmhaKernel>
  static void RunWithFwdPagedKVKernel(
      GroupedForwardParams& param,
      hipStream_t stream) {
    const auto kargs = [&] {
      return FmhaKernel::MakeKargs(
          param.q_ptr,
          param.k_ptr,
          param.v_ptr,
          param.attn_bias_ptr,
          nullptr, // lse_ptr,
          param.out_ptr, // o_ptr
          param.seqstart_q_dev_ptr,
          param.seqstart_k_dev_ptr,
          param.seqlen_k_dev_ptr,
          param.K, // hdim_q
          param.Kv, // hdim_v
          param.Hq, // nhead_q
          param.Hq / param.Hkv, // nhead_ratio_qk
          param.use_paged_kvcache ? param.block_table_ptr : nullptr,
          param.use_paged_kvcache ? param.batch_stride_block_table : 0,
          param.use_paged_kvcache ? param.page_block_size : 0,
          param.use_paged_kvcache ? param.is_gappy : false,
          param.scale,
          1.0f, // scale_p
          1.0f, // scale_o
          0, // logits_soft_cap
          param.q_strides[0], // q, k, v, bias, out tensor seq-dim
                              // stride
          param.k_strides[0],
          param.v_strides[0],
          param.attn_bias_strides[2],
          param.out_strides[0],
          param.q_strides[1], // q, k, v, bias, lse, out tensor
                              // head-dim stride
          param.k_strides[1],
          param.v_strides[1],
          param.attn_bias_strides[1],
          0, // nhead_stride_lse
          param.out_strides[1],
          param.use_paged_kvcache ? param.k_strides[0] * param.page_block_size
                                  : 0, // batch_stride_k
          param.use_paged_kvcache ? param.v_strides[0] * param.page_block_size
                                  : 0, // batch_stride_v
          (param.window_size > 0) ? param.window_size - 1
                                  : -1, // window_left_size
          (param.custom_mask_type == 0) ? -1 : 0, // window_right_size
          param.custom_mask_type,
          0); // min_seqlen_q
    }();

    dim3 kGridSize = FmhaKernel::GridSize(
        param.num_batches,
        param.Hq,
        param.max_seqlen_q,
        param.Kv,
        kargs.seqlen_k_ptr != nullptr);
    constexpr dim3 kBlockSize = FmhaKernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = FmhaKernel::kBlockPerCu;

    (void)ck_tile::launch_kernel(
        ck_tile::stream_config{stream, false},
        ck_tile::make_kernel<kBlockSize.x, kBlockPerCu>(
            FmhaKernel{}, kGridSize, kBlockSize, 0, kargs));
  };
};
