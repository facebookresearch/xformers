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
#include "ck_tiled_fmha_bwd_setting.h"
#include "ck_tiled_fmha_params.h"

template <
    typename ScalarType,
    bool kHasCausalMask,
    bool kHasBias,
    bool kHasBiasGrad,
    bool kHasDropout,
    ck_tile::index_t MaxK>
struct batched_backward_causalmask_bias_dropout_dispatch {
  template <typename FmhaTraits, typename FmhaMask>
  using FmhaBwdPipelineProblemTemp = ck_tile::BlockFmhaBwdPipelineProblem<
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
      constexpr ck_tile::index_t kBlockSize = 256;

      const bool pad_seqlen_q = !(param.M % kBlockSize == 0);
      const bool pad_headdim_v =
          !(param.Kv % FmhaBwdShape<MaxK>::kVHeaddim == 0);

      BOOL_SWITCH_2(
          pad_seqlen_q, kPadSeqLenQ, pad_headdim_v, kPadHeadDimV, [&] {
            constexpr ck_tile::index_t occupancy = 2;

            using FmhaOGradDotOTraits_ = ck_tile::TileFmhaBwdOGradDotOTraits<
                kPadSeqLenQ,
                kPadHeadDimV,
                occupancy>;

            using FmhaBwdOGradDotOPipelineProblem =
                ck_tile::BlockFmhaBwdOGradDotOPipelineProblem<
                    typename FmhaBwdTypeConfig<ScalarType>::ODataType,
                    typename FmhaBwdTypeConfig<ScalarType>::OGradDataType,
                    typename FmhaBwdTypeConfig<ScalarType>::DDataType,
                    kBlockSize,
                    FmhaBwdShape<MaxK>::kVHeaddim,
                    false, // kIsGroupMode
                    FmhaOGradDotOTraits_>;

            using FmhaBwdOGradDotOPipeline =
                typename ck_tile::BlockFmhaBwdOGradDotO<
                    FmhaBwdOGradDotOPipelineProblem>;

            using FmhaBwdOGradDotOKernel_ = ck_tile::FmhaBwdOGradDotOKernel<
                ck_tile::FmhaBwdOGradDotOTilePartitioner<kBlockSize>,
                FmhaBwdOGradDotOPipeline>;

            RunWithBwdOGradDotOKernel<FmhaBwdOGradDotOKernel_>(param, stream);
          });
    }

    {
      const bool has_local_attention = (param.window_size > 0) ? true : false;

      BOOL_SWITCH(has_local_attention, USE_LOCAL_ATTENTION, [&] {
        constexpr ck_tile::index_t occupancy = 1;
        constexpr bool has_masking = kHasCausalMask || USE_LOCAL_ATTENTION;

        using FmhaMask = ck_tile::SimplifiedGenericAttentionMask<has_masking>;

        using FmhaBwdShape_ = FmhaBwdShape<MaxK>;
        using FmhaBwdTilePartitioner_ =
            ck_tile::FmhaBwdTilePartitioner<FmhaBwdShape_>;

        constexpr auto kBiasEnum = kHasBias
            ? ck_tile::BlockAttentionBiasEnum::ELEMENTWISE_BIAS
            : ck_tile::BlockAttentionBiasEnum::NO_BIAS;

        constexpr bool kPadSeqLenQ = true;
        constexpr bool kPadSeqLenK = true;

        const bool pad_headdim_q = !(param.K % FmhaBwdShape_::kQKHeaddim == 0);
        const bool pad_headdim_v = !(param.Kv % FmhaBwdShape_::kVHeaddim == 0);

        // usually headdim_q and headdim_v are same, consider them together
        // to determine whether to do padding saving some compiling time
        const bool pad_headdim = (pad_headdim_q || pad_headdim_v);

        BOOL_SWITCH(pad_headdim, kPadHeadDim, [&] {
          using FmhaBwdTraits_ = ck_tile::TileFmhaTraits<
              kPadSeqLenQ,
              kPadSeqLenK,
              kPadHeadDim, // kPadHeadDimQ,
              kPadHeadDim, // kPadHeadDimV,
              kBiasEnum,
              kHasBiasGrad,
              false, // kStoreLSE
              kHasDropout,
              false, // kDoFp8StaticQuant place-holder
              occupancy>;

          using FmhaBwdPipelineProblem =
              FmhaBwdPipelineProblemTemp<FmhaBwdTraits_, FmhaMask>;

          constexpr auto FmhaBwdPipelineEnum_ =
              FmhaBwdPipelineEnumSelector<MaxK>::value;

          using FmhaBwdPipeline_ = typename FmhaBwdPipelineMaker<
              FmhaBwdPipelineEnum_,
              FmhaBwdPipelineProblem>::pipeline;

          using FmhaBwdKGradEpilogue_ =
              ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
                  typename FmhaBwdTypeConfig<ScalarType>::AccDataType,
                  typename FmhaBwdTypeConfig<ScalarType>::KGradDataType,
                  kPadSeqLenK,
                  kPadHeadDim>>;

          using FmhaBwdVGradEpilogue_ =
              ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
                  typename FmhaBwdTypeConfig<ScalarType>::AccDataType,
                  typename FmhaBwdTypeConfig<ScalarType>::VGradDataType,
                  kPadSeqLenK,
                  kPadHeadDim>>;

          using FmhaBwdDQDKDVKernel_ = ck_tile::FmhaBwdDQDKDVKernel<
              FmhaBwdTilePartitioner_,
              FmhaBwdPipeline_,
              FmhaBwdKGradEpilogue_,
              FmhaBwdVGradEpilogue_>;

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
    constexpr ck_tile::index_t kBlockPerCu =
        FmhaBwdOGradDotOKernel::kBlockPerCu;

    (void)ck_tile::launch_kernel(
        ck_tile::stream_config{stream, false},
        ck_tile::make_kernel<kBlockSize.x, kBlockPerCu>(
            FmhaBwdOGradDotOKernel{}, kGridSize, kBlockSize, 0, kargs));
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
    constexpr ck_tile::index_t kBlockPerCu = FmhaBwdDQDKDVKernel::kBlockPerCu;

    (void)ck_tile::launch_kernel(
        ck_tile::stream_config{stream, false},
        ck_tile::make_kernel<kBlockSize.x, kBlockPerCu>(
            FmhaBwdDQDKDVKernel{}, kGridSize, kBlockSize, 0, kargs));
  }
};

template <
    typename ScalarType,
    bool kHasCausalMask,
    bool kHasBias,
    bool kHasBiasGrad,
    bool kHasDropout,
    ck_tile::index_t MaxK>
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
