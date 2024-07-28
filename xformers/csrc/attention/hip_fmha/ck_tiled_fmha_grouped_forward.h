/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ck_tile/core/numeric/integer.hpp>
#include <ck_tile/host.hpp>
#include <ck_tile/ops/epilogue.hpp>
#include <ck_tile/ops/fmha.hpp>

#include "ck_tiled_bool_switch.h"
#include "ck_tiled_fmha_fwd_setting.h"
#include "ck_tiled_fmha_params.h"

template <
    typename ScalarType,
    bool kHasCausalMask,
    bool kHasBias,
    bool kHasDropout,
    ck_tile::index_t MaxK>
struct grouped_forward_causalmask_bias_dropout_dispatch {
  template <typename FmhaTraits, typename FmhaMask>
  using FmhaPipelineProblemTemp = ck_tile::BlockFmhaPipelineProblem<
      typename FmhaFwdTypeConfig<ScalarType>::QDataType,
      typename FmhaFwdTypeConfig<ScalarType>::KDataType,
      typename FmhaFwdTypeConfig<ScalarType>::VDataType,
      typename FmhaFwdTypeConfig<ScalarType>::SaccDataType,
      typename FmhaFwdTypeConfig<ScalarType>::SMPLComputeDataType,
      typename FmhaFwdTypeConfig<ScalarType>::BiasDataType,
      typename FmhaFwdTypeConfig<ScalarType>::RandValOutputDataType,
      typename FmhaFwdTypeConfig<ScalarType>::LSEDataType,
      typename FmhaFwdTypeConfig<ScalarType>::PDataType,
      typename FmhaFwdTypeConfig<ScalarType>::OaccDataType,
      typename FmhaFwdTypeConfig<ScalarType>::ODataType,
      FmhaFwdShape<MaxK>,
      true, // kIsGroupMode
      FmhaMask,
      FmhaTraits>;

  static void Run(GroupedForwardParams& param, hipStream_t stream) {
    const bool has_local_attention = (param.window_size > 0) ? true : false;

    BOOL_SWITCH(has_local_attention, USE_LOCAL_ATTENTION, [&] {
      constexpr bool has_masking = kHasCausalMask || USE_LOCAL_ATTENTION;

      using FmhaMask = ck_tile::SimplifiedGenericAttentionMask<has_masking>;

      using FmhaFwdShape_ = FmhaFwdShape<MaxK>;

      constexpr ck_tile::index_t occupancy = (MaxK == 64) ? 3
          : (MaxK == 256)                                 ? 1
                                                          : 2;

      constexpr auto kBiasEnum = kHasBias
          ? ck_tile::BlockAttentionBiasEnum::ELEMENTWISE_BIAS
          : ck_tile::BlockAttentionBiasEnum::NO_BIAS;

      constexpr bool kPadSeqLenQ = true;
      constexpr bool kPadSeqLenK = true;

      const bool pad_headdim_q =
          !(param.K % FmhaFwdShape_::kK0BlockLength == 0);
      const bool pad_headdim_v = !(param.Kv % FmhaFwdShape_::kN1 == 0);

      BOOL_SWITCH_2(
          pad_headdim_q, kPadHeadDimQ, pad_headdim_v, kPadHeadDimV, [&] {
            using FmhaFwdTraits_ = ck_tile::TileFmhaTraits<
                kPadSeqLenQ,
                kPadSeqLenK,
                kPadHeadDimQ,
                kPadHeadDimV,
                kBiasEnum,
                false, // kHasBiasGrad place-holder
                true, // kStoreLSE
                kHasDropout,
                false, // kDoFp8StaticQuant place-holder
                occupancy>;

            using FmhaPipelineProblem =
                FmhaPipelineProblemTemp<FmhaFwdTraits_, FmhaMask>;

            using FmhaFwdPipeline_ =
                ck_tile::BlockFmhaPipelineQRKSVS<FmhaPipelineProblem>;

            using FmhaFwdEpilogue_ =
                ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
                    typename FmhaFwdTypeConfig<ScalarType>::OaccDataType,
                    typename FmhaFwdTypeConfig<ScalarType>::ODataType,
                    kPadSeqLenQ,
                    kPadHeadDimV>>;

            if (param.seqlen_k_dev_ptr !=
                nullptr) { // seqlen_k of batches are padded
              using FmhaTilePartitioner =
                  ck_tile::FmhaFwdTilePartitioner_HBS<FmhaFwdShape_>;
              using FmhaFwdKernel_ = ck_tile::FmhaFwdKernel<
                  FmhaTilePartitioner,
                  FmhaFwdPipeline_,
                  FmhaFwdEpilogue_>;

              RunWithKernel<FmhaFwdKernel_>(param, stream);
            } else {
              using FmhaTilePartitioner =
                  ck_tile::FmhaFwdTilePartitioner_SHB<FmhaFwdShape_>;
              using FmhaFwdKernel_ = ck_tile::FmhaFwdKernel<
                  FmhaTilePartitioner,
                  FmhaFwdPipeline_,
                  FmhaFwdEpilogue_>;

              RunWithKernel<FmhaFwdKernel_>(param, stream);
            }
          });
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
          1.0f, // scale_p
          1.0f, // scale_o
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
          param.lse_strides[0], // batch_stride_lse
          (param.window_size > 0) ? param.window_size - 1
                                  : -1, // window_left_size
          (param.custom_mask_type == 0) ? -1 : 0, // window_right_size
          param.custom_mask_type,
          param.dropout_prob,
          false, // is_store_randval
          {param.philox_seed, param.philox_offset});
    }();

    dim3 kGridSize = FmhaFwdKernel::GridSize(
        param.num_batches, param.Hq, param.max_seqlen_q, param.Kv);
    constexpr dim3 kBlockSize = FmhaFwdKernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = FmhaFwdKernel::kBlockPerCu;

    (void)ck_tile::launch_kernel(
        ck_tile::stream_config{stream, false},
        ck_tile::make_kernel<kBlockSize.x, kBlockPerCu>(
            FmhaFwdKernel{}, kGridSize, kBlockSize, 0, kargs));
  };
};

template <
    typename ScalarType,
    bool kHasCausalMask,
    bool kHasBias,
    bool kHasDropout,
    ck_tile::index_t MaxK>
void run_grouped_forward_causalmask_bias_dropout_dispatch(
    GroupedForwardParams& param,
    hipStream_t stream) {
  grouped_forward_causalmask_bias_dropout_dispatch<
      ScalarType,
      kHasCausalMask,
      kHasBias,
      kHasDropout,
      MaxK>::Run(param, stream);
};
