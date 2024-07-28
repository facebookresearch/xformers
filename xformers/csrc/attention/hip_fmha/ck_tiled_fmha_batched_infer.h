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
#include "ck_tiled_headdim_switch.h"

template <
    typename ScalarType,
    bool kHasCausalMask,
    bool kHasBias,
    bool kHasDropout,
    ck_tile::index_t MaxK>
struct batched_infer_causalmask_bias_dropout_dispatch {
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

      using FmhaShape = FmhaFwdShape<MaxK>;
      using FmhaTilePartitioner = ck_tile::FmhaFwdTilePartitioner<FmhaShape>;
      constexpr ck_tile::index_t occupancy =
          (MaxK == 64) ? 3 : ((MaxK == 256) ? 1 : 2);

      constexpr auto kBiasEnum = kHasBias
          ? ck_tile::BlockAttentionBiasEnum::ELEMENTWISE_BIAS
          : ck_tile::BlockAttentionBiasEnum::NO_BIAS;

      const bool pad_seqlen_q = !(param.M % FmhaShape::kM0 == 0);
      const bool pad_seqlen_k =
          (param.N == 0) || !(param.N % FmhaShape::kN0 == 0);
      const bool pad_headdim_v = !(param.Kv % FmhaShape::kN1 == 0);
      const bool pad_headdim_q = !(param.K % FmhaShape::kK0BlockLength == 0);

      // usually headdim_q and headdim_v are same, consider them together to
      // determine whether to do padding saving some compiling time
      const bool pad_headdim = (pad_headdim_q || pad_headdim_v);

      const bool use_async_pipeline =
          ((param.K % 8 == 0) && (param.Kv % 8 == 0) && (MaxK <= 128));

      if (!use_async_pipeline) {
        BOOL_SWITCH_3(
            pad_seqlen_q,
            kPadSeqLenQ,
            pad_seqlen_k,
            kPadSeqLenK,
            pad_headdim,
            kPadHeadDim,
            [&] {
              using FmhaTraits = ck_tile::TileFmhaTraits<
                  kPadSeqLenQ,
                  kPadSeqLenK,
                  kPadHeadDim, // kPadHeadDimQ,
                  kPadHeadDim, // kPadHeadDimV,
                  kBiasEnum,
                  false, // kHasBiasGrad place-holder
                  false, // kStoreLSE
                  kHasDropout,
                  false, // kDoFp8StaticQuant place-holder
                  occupancy>;

              using FmhaPipelineProblem =
                  FmhaPipelineProblemTemp<FmhaTraits, FmhaMask>;

              using FmhaPipeline =
                  ck_tile::BlockFmhaPipelineQRKSVS<FmhaPipelineProblem>;

              using FmhaEpilogue =
                  ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
                      typename FmhaFwdTypeConfig<ScalarType>::OaccDataType,
                      typename FmhaFwdTypeConfig<ScalarType>::ODataType,
                      kPadSeqLenQ,
                      kPadHeadDim>>;

              using FmhaKernel = ck_tile::FmhaFwdKernel<
                  FmhaTilePartitioner,
                  FmhaPipeline,
                  FmhaEpilogue>;

              RunWithKernel<FmhaKernel>(param, stream);
            });
      } else {
        BOOL_SWITCH(pad_seqlen_k, kPadSeqLenK, [&] {
          using FmhaTraits = ck_tile::TileFmhaTraits<
              true, // kPadSeqLenQ,
              kPadSeqLenK,
              true, // kPadHeadDimQ,
              true, // kPadHeadDimV,
              kBiasEnum,
              false, // kHasBiasGrad place-holder
              false, // kStoreLSE
              kHasDropout,
              false, // kDoFp8StaticQuant place-holder
              occupancy>;

          using FmhaPipelineProblem =
              FmhaPipelineProblemTemp<FmhaTraits, FmhaMask>;

          using FmhaPipeline =
              ck_tile::BlockFmhaPipelineQRKSVSAsync<FmhaPipelineProblem>;

          using FmhaEpilogue =
              ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
                  typename FmhaFwdTypeConfig<ScalarType>::OaccDataType,
                  typename FmhaFwdTypeConfig<ScalarType>::ODataType,
                  true,
                  true>>;

          using FmhaKernel = ck_tile::
              FmhaFwdKernel<FmhaTilePartitioner, FmhaPipeline, FmhaEpilogue>;

          RunWithKernel<FmhaKernel>(param, stream);
        });
      };
    });
  };

  template <typename FmhaKernel>
  static void RunWithKernel(BatchedForwardParams& param, hipStream_t stream) {
    const auto kargs = [&] {
      return FmhaKernel::MakeKargs(
          param.q_ptr,
          param.k_ptr,
          param.v_ptr,
          param.attn_bias_ptr,
          nullptr, // rand_val_ptr
          nullptr, // lse_ptr
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
          0, // nhead_stride_randval
          0, // nhead_stride_lse
          param.out_strides[2],
          param.q_strides[0], // q, k, v, bias, randval, lse, out tensor
                              // batch-dim stride
          param.k_strides[0],
          param.v_strides[0],
          param.attn_bias_strides[0],
          0, // batch_stride_randval
          0, // batch_stride_lse
          param.out_strides[0],
          (param.window_size > 0) ? param.window_size - 1
                                  : -1, // window_left_size
          (param.custom_mask_type == 0) ? -1 : 0, // window_right_size
          param.custom_mask_type,
          param.dropout_prob, // dropout ratio
          false, // is_store_randval
          {param.philox_seed, param.philox_offset});
    }();

    dim3 kGridSize = FmhaKernel::GridSize(param.B, param.Hq, param.M, param.Kv);
    constexpr dim3 kBlockSize = FmhaKernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = FmhaKernel::kBlockPerCu;

    (void)ck_tile::launch_kernel(
        ck_tile::stream_config{stream, false},
        ck_tile::make_kernel<kBlockSize.x, kBlockPerCu>(
            FmhaKernel{}, kGridSize, kBlockSize, 0, kargs));
  };
};

template <
    typename ScalarType,
    bool kHasCausalMask,
    bool kHasBias,
    bool kHasDropout,
    ck_tile::index_t MaxK>
void run_batched_infer_causalmask_bias_dropout_dispatch(
    BatchedForwardParams& param,
    hipStream_t stream) {
  batched_infer_causalmask_bias_dropout_dispatch<
      ScalarType,
      kHasCausalMask,
      kHasBias,
      kHasDropout,
      MaxK>::Run(param, stream);
};
