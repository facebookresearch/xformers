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
#include "ck_tiled_fmha_bwd_setting.h"
#include "ck_tiled_fmha_params.h"

template <
    typename ScalarType,
    bool kHasMask,
    bool kHasBias,
    bool kHasBiasGrad,
    bool kHasDropout,
    ck_tile::index_t MaxK>
struct grouped_backward_mask_bias_dropout_dispatch {
  using FmhaBlockDropout =
      typename FmhaBwdBlockDropoutMaker<kHasDropout, MaxK>::dropout;

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
      true, // kIsGroupMode
      false, // non-deterministic
      FmhaMask,
      FmhaBlockDropout,
      FmhaTraits>;

  static constexpr bool NeedConvertGradQ = !std::is_same<
      typename FmhaBwdTypeConfig<ScalarType>::AccDataType,
      typename FmhaBwdTypeConfig<ScalarType>::QGradDataType>::value;

  static void Run(GroupedBackwardParams& param, hipStream_t stream) {
    {
      constexpr ck_tile::index_t kBlockSize = 64;
      bool pad_headdim_v = !(param.Kv % MaxK == 0);

      constexpr bool kPadSeqLenQ = true;

      BOOL_SWITCH(pad_headdim_v, kPadHeadDimV, [&] {
        constexpr ck_tile::index_t occupancy = 2;

        using FmhaOGradDotOTraits_ = ck_tile::
            TileFmhaBwdOGradDotOTraits<kPadSeqLenQ, kPadHeadDimV, occupancy>;

        using FmhaBwdOGradDotOPipelineProblem =
            ck_tile::BlockFmhaBwdOGradDotOPipelineProblem<
                typename FmhaBwdTypeConfig<ScalarType>::ODataType,
                typename FmhaBwdTypeConfig<ScalarType>::OGradDataType,
                typename FmhaBwdTypeConfig<ScalarType>::DDataType,
                kBlockSize,
                MaxK, // kVHeaddim
                true, // kIsGroupMode
                FmhaOGradDotOTraits_>;

        using FmhaBwdOGradDotOPipeline_ =
            typename ck_tile::BlockFmhaBwdOGradDotO<
                FmhaBwdOGradDotOPipelineProblem>;

        using FmhaBwdOGradDotOKernel_ =
            ck_tile::FmhaBwdOGradDotOKernel<FmhaBwdOGradDotOPipeline_>;

        RunWithBwdOGradDotOKernel<FmhaBwdOGradDotOKernel_>(param, stream);
      });
    };

    {
      constexpr ck_tile::index_t occupancy = 1;
      const bool has_dropout = (param.dropout_prob > 0.0f);

      using FmhaMask = ck_tile::SimplifiedGenericAttentionMask<kHasMask>;

      constexpr auto kBiasEnum = kHasBias
          ? ck_tile::BlockAttentionBiasEnum::ELEMENTWISE_BIAS
          : ck_tile::BlockAttentionBiasEnum::NO_BIAS;

      constexpr bool kPadSeqLenQ = true;
      constexpr bool kPadSeqLenK = true;

      const bool pad_headdim_q =
          !(param.K % FmhaBwdShape<MaxK>::kQKHeaddim == 0);
      const bool pad_headdim_v =
          !(param.Kv % FmhaBwdShape<MaxK>::kVHeaddim == 0);

      BOOL_SWITCH_2(
          pad_headdim_q, kPadHeadDimQ, pad_headdim_v, kPadHeadDimV, [&] {
            using FmhaBwdTraits_ = ck_tile::TileFmhaTraits<
                kPadSeqLenQ,
                kPadSeqLenK,
                kPadHeadDimQ,
                kPadHeadDimV,
                false, // kHasLogitsSoftCap
                kBiasEnum,
                kHasBiasGrad,
                false, // kStoreLSE
                false, // place-holder for kHasDropout, not used actually
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
                    kPadHeadDimQ>>;

            using FmhaBwdVGradEpilogue_ =
                ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
                    typename FmhaBwdTypeConfig<ScalarType>::AccDataType,
                    typename FmhaBwdTypeConfig<ScalarType>::VGradDataType,
                    kPadSeqLenK,
                    kPadHeadDimV>>;

            using FmhaBwdDQDKDVKernel_ = ck_tile::FmhaBwdDQDKDVKernel<
                FmhaBwdPipeline_,
                FmhaBwdKGradEpilogue_,
                FmhaBwdVGradEpilogue_>;

            RunWithBwdDQDKDVKernel<FmhaBwdDQDKDVKernel_>(param, stream);
          });
    };

    if constexpr (NeedConvertGradQ) {
      constexpr ck_tile::index_t kBlockSize = 128;

      const bool pad_seqlen_q = true;
      const bool pad_headdim_q = !(param.K % MaxK == 0);

      BOOL_SWITCH_2(
          pad_seqlen_q, kPadSeqLenQ, pad_headdim_q, kPadHeadDimQ, [&] {
            constexpr ck_tile::index_t occupancy = 2;

            using FmhaBwdConvertQGradTraits_ =
                ck_tile::TileFmhaBwdConvertQGradTraits<
                    kPadSeqLenQ,
                    kPadHeadDimQ,
                    occupancy>;

            using FmhaBwdConvertQGradPipelineProblem =
                ck_tile::BlockFmhaBwdConvertQGradPipelineProblem<
                    typename FmhaBwdTypeConfig<ScalarType>::AccDataType,
                    typename FmhaBwdTypeConfig<ScalarType>::QGradDataType,
                    kBlockSize,
                    64, // kM0
                    1, // kN0, no use
                    MaxK, // kQKHeaddim
                    true, // kIsGroupMode
                    false, // kIsDeterministic
                    FmhaBwdConvertQGradTraits_>;

            using FmhaBwdConvertQGradPipeline =
                typename ck_tile::BlockFmhaBwdConvertQGrad<
                    FmhaBwdConvertQGradPipelineProblem>;

            using FmhaBwdConvertQGradKernel_ =
                ck_tile::FmhaBwdConvertQGradKernel<FmhaBwdConvertQGradPipeline>;

            RunWithBwdConvertQGradKernel<FmhaBwdConvertQGradKernel_>(
                param, stream);
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
          param.lsed_strides[0]); // nhead_stride_d
    }();

    dim3 kGridSize = FmhaBwdOGradDotOKernel::GridSize(
        param.num_batches, param.Hq, param.max_seqlen_q);
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
      GroupedBackwardParams& param,
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
          nullptr, // randval_ptr
          param.grad_k_ptr,
          param.grad_v_ptr,
          param.grad_bias_ptr,
          NeedConvertGradQ ? param.grad_q_f32_ptr : param.grad_q_ptr,
          param.seqstart_q_dev_ptr,
          param.seqstart_k_dev_ptr,
          param.seqlen_k_dev_ptr,
          param.K,
          param.Kv,
          param.Hq,
          param.Hq / param.Hkv,
          param.scale,
          param.q_strides[0], // q, k, v, bias, do, dq_f32, dk, dv, dbias
                              // seq-dim stride
          param.k_strides[0],
          param.v_strides[0],
          param.attn_bias_strides[1],
          0, // stride_randval
          param.grad_out_strides[0],
          NeedConvertGradQ ? param.grad_q_f32_strides[0] : param.q_strides[0],
          param.grad_k_strides[0],
          param.grad_v_strides[0],
          param.attn_bias_strides[1], // assume grad_bias has same strides as
                                      // bias.
          param.q_strides[1], // q, k, v, bias, do, lse/dot, dq_f32, dk, dv,
                              // dbias nhead-dim strides
          param.k_strides[1],
          param.v_strides[1],
          param.attn_bias_strides[0],
          0, // nhead_stride_randval
          param.grad_out_strides[1],
          param.lsed_strides[0], // assume lse/dot is in HM contiguous layout
          NeedConvertGradQ ? param.grad_q_f32_strides[1] : param.q_strides[1],
          param.grad_k_strides[1],
          param.grad_v_strides[1],
          param.attn_bias_strides[0], // assume grad_bias has same strides as
                                      // bias
          0, // split_stride_dq_acc
          (param.window_size > 0) ? param.window_size - 1
                                  : -1, // window_left_size
          (param.custom_mask_type == 0) ? -1 : 0, // window_right_size
          param.custom_mask_type,
          param.dropout_prob, // dropout ratio
          std::make_pair(param.philox_seed, param.philox_offset));
    }();

    dim3 kGridSize = FmhaBwdDQDKDVKernel::GridSize(
        param.num_batches, param.Hq, param.max_seqlen_k);
    constexpr dim3 kBlockSize = FmhaBwdDQDKDVKernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = FmhaBwdDQDKDVKernel::kBlockPerCu;

    (void)ck_tile::launch_kernel(
        ck_tile::stream_config{stream, false},
        ck_tile::make_kernel<kBlockSize.x, kBlockPerCu>(
            FmhaBwdDQDKDVKernel{}, kGridSize, kBlockSize, 0, kargs));
  }

  template <typename FmhaBwdConvertQGradKernel>
  static void RunWithBwdConvertQGradKernel(
      GroupedBackwardParams& param,
      hipStream_t stream) {
    const auto kargs = [&] {
      return FmhaBwdConvertQGradKernel::MakeKargs(
          param.grad_q_f32_ptr,
          param.grad_q_ptr,
          param.seqstart_q_dev_ptr,
          param.seqstart_k_dev_ptr,
          param.K, // headdim of q/k
          param.q_strides[0],
          param.grad_q_f32_strides[0],
          param.q_strides[1],
          param.grad_q_f32_strides[1],
          0);
    }();

    dim3 kGridSize = FmhaBwdConvertQGradKernel::GridSize(
        param.num_batches, param.Hq, param.max_seqlen_q);
    constexpr dim3 kBlockSize = FmhaBwdConvertQGradKernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu =
        FmhaBwdConvertQGradKernel::kBlockPerCu;

    (void)ck_tile::launch_kernel(
        ck_tile::stream_config{stream, false},
        ck_tile::make_kernel<kBlockSize.x, kBlockPerCu>(
            FmhaBwdConvertQGradKernel{}, kGridSize, kBlockSize, 0, kargs));
  }
};

template <
    typename ScalarType,
    bool kHasMask,
    bool kHasBias,
    bool kHasBiasGrad,
    bool kHasDropout,
    ck_tile::index_t MaxK>
void run_grouped_backward_mask_bias_dropout_dispatch(
    GroupedBackwardParams& param,
    hipStream_t stream) {
  grouped_backward_mask_bias_dropout_dispatch<
      ScalarType,
      kHasMask,
      kHasBias,
      kHasBiasGrad,
      kHasDropout,
      MaxK>::Run(param, stream);
};
