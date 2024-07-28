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
struct batched_forward_causalmask_bias_dropout_dispatch {
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
      false, // kIsGroupMode
      FmhaMask,
      FmhaTraits>;

  static void Run(BatchedForwardParams& param, hipStream_t stream) {
    const bool has_local_attention = (param.window_size > 0) ? true : false;

    BOOL_SWITCH(has_local_attention, USE_LOCAL_ATTENTION, [&] {
      constexpr bool has_masking = kHasCausalMask || USE_LOCAL_ATTENTION;

      using FmhaMask = ck_tile::SimplifiedGenericAttentionMask<has_masking>;

      using FmhaFwdShape_ = FmhaFwdShape<MaxK>;
      using FmhaFwdTilePartitioner_ =
          ck_tile::FmhaFwdTilePartitioner<FmhaFwdShape_>;
      constexpr ck_tile::index_t occupancy =
          (MaxK == 64) ? 3 : ((MaxK == 256) ? 1 : 2);

      constexpr auto kBiasEnum = kHasBias
          ? ck_tile::BlockAttentionBiasEnum::ELEMENTWISE_BIAS
          : ck_tile::BlockAttentionBiasEnum::NO_BIAS;

      const bool pad_seqlen_q = !(param.M % FmhaFwdShape_::kM0 == 0);
      const bool pad_seqlen_k =
          (param.N == 0) || !(param.N % FmhaFwdShape_::kN0 == 0);
      const bool pad_headdim_q =
          !(param.K % FmhaFwdShape_::kK0BlockLength == 0);
      const bool pad_headdim_v = !(param.Kv % FmhaFwdShape_::kN1 == 0);

      // usually headdim_q and headdim_v are same, consider them together to
      // determine whether to do padding saving some compiling time
      const bool pad_headdim = (pad_headdim_q || pad_headdim_v);

      const bool use_async_pipeline =
          ((param.K % 8 == 0) && (param.Kv % 8 == 0) && (MaxK <= 128));

      BOOL_SWITCH_3(
          pad_seqlen_q,
          kPadSeqLenQ,
          pad_seqlen_k,
          kPadSeqLenK,
          pad_headdim,
          kPadHeadDim,
          [&] {
            using FmhaFwdTraits_ = ck_tile::TileFmhaTraits<
                kPadSeqLenQ,
                kPadSeqLenK,
                kPadHeadDim, // kPadHeadDimQ
                kPadHeadDim, // kPadHeadDimV
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
                    kPadHeadDim>>;

            using FmhaFwdKernel_ = ck_tile::FmhaFwdKernel<
                FmhaFwdTilePartitioner_,
                FmhaFwdPipeline_,
                FmhaFwdEpilogue_>;

            RunWithKernel<FmhaFwdKernel_>(param, stream);
          });
    });
  };

  template <typename FmhaFwdKernel>
  static void RunWithKernel(BatchedForwardParams& param, hipStream_t stream) {
    const auto kargs = [&] {
      return FmhaFwdKernel::MakeKargs(
          param.q_ptr,
          param.k_ptr,
          param.v_ptr,
          param.attn_bias_ptr,
          nullptr, // rand_val_ptr
          param.logsumexp_ptr,
          param.out_ptr,
          param.M, // seqlen_q
          param.N, // seqlen_k
          param.K, // hdim_q
          param.Kv, // hdim_v
          param.Hq, // nhead_q
          param.Hq / param.Hkv, // nhead_ratio_qk
          param.scale,
          1.0f, // scale_p
          1.0f, // scale_o
          param.q_strides[1], // q, k, v, bias, randval, out tensor seq-dim
                              // stride
          param.k_strides[1],
          param.v_strides[1],
          param.attn_bias_strides[2],
          0, // stride_randval
          param.out_strides[1],
          param.q_strides[2], // q, k, v, bias, randval, lse, out tensor
                              // head-dim stride
          param.k_strides[2],
          param.v_strides[2],
          param.attn_bias_strides[1],
          0, // nhead_randva
          param.lse_strides[1], // nhead_stride_lse
          param.out_strides[2],
          param.q_strides[0], // q, k, v, bias, randval, lse, out tensor
                              // batch-dim stride
          param.k_strides[0],
          param.v_strides[0],
          param.attn_bias_strides[0],
          0, // batch_stride_randval
          param.lse_strides[0], // batch_stride_lse
          param.out_strides[0],
          (param.window_size > 0) ? param.window_size - 1
                                  : -1, // window_left_size
          (param.custom_mask_type == 0) ? -1 : 0, // window_right_size
          param.custom_mask_type,
          param.dropout_prob, // dropout ratio
          false, // is_store_randval
          {param.philox_seed, param.philox_offset});
    }();

    dim3 kGridSize =
        FmhaFwdKernel::GridSize(param.B, param.Hq, param.M, param.Kv);
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
void run_batched_forward_causalmask_bias_dropout_dispatch(
    BatchedForwardParams& param,
    hipStream_t stream) {
  batched_forward_causalmask_bias_dropout_dispatch<
      ScalarType,
      kHasCausalMask,
      kHasBias,
      kHasDropout,
      MaxK>::Run(param, stream);
};
