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
#include "ck_tiled_fmha_num_kv_split_switch.h"
#include "ck_tiled_fmha_params.h"

template <
    typename ScalarType,
    bool kHasCausalMask,
    bool kHasBias,
    ck_tile::index_t MaxK>
struct batched_infer_splitkv_causalmask_bias_dropout_dispatch {
  template <typename FmhaFwdSplitKVTraits, typename FmhaMask>
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
          typename FmhaFwdTypeConfig<ScalarType>::OaccDataType,
          FmhaFwdShape<MaxK>,
          false, // kIsGroupMode
          FmhaMask,
          FmhaFwdSplitKVTraits>;

  template <
      ck_tile::index_t kM0,
      ck_tile::index_t kN1,
      typename FmhaSplitKVCombineTraits>
  using FmhaSplitKVCombinePipelineProblemTemp =
      ck_tile::BlockFmhaSplitKVCombinePipelineProblem<
          typename FmhaFwdTypeConfig<ScalarType>::LSEDataType,
          typename FmhaFwdTypeConfig<ScalarType>::OaccDataType,
          typename FmhaFwdTypeConfig<ScalarType>::ODataType,
          MaxK, // headdim_v
          kM0,
          kN1,
          false, // kIsGroupMode
          FmhaSplitKVCombineTraits>;

  static void Run(BatchedForwardParams& param, hipStream_t stream) {
    {
      const bool has_local_attention = (param.window_size > 0) ? true : false;

      BOOL_SWITCH(has_local_attention, USE_LOCAL_ATTENTION, [&] {
        constexpr bool has_masking = kHasCausalMask || USE_LOCAL_ATTENTION;

        using FmhaMask = ck_tile::SimplifiedGenericAttentionMask<has_masking>;

        using FmhaShape = FmhaFwdShape<MaxK>;
        using FmhaTilePartitioner =
            ck_tile::FmhaFwdSplitKVTilePartitioner<FmhaShape>;
        constexpr ck_tile::index_t occupancy = -1;

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

        BOOL_SWITCH_3(
            pad_seqlen_q,
            kPadSeqLenQ,
            pad_seqlen_k,
            kPadSeqLenK,
            pad_headdim,
            kPadHeadDim,
            [&] {
              using FmhaTraits = ck_tile::TileFmhaFwdSplitKVTraits<
                  kPadSeqLenQ,
                  kPadSeqLenK,
                  kPadHeadDim, // kPadHeadDimQ,
                  kPadHeadDim, // kPadHeadDimV,
                  kBiasEnum,
                  false, // kHasBiasGrad place-holder
                  false, // kStoreLSE
                  false, // kDoFp8StaticQuant place-holder
                  false, // kIsPagedKV
                  true, // kHasUnevenSplits
                  occupancy>;

              using FmhaPipelineProblem =
                  FmhaFwdSplitKVPipelineProblemTemp<FmhaTraits, FmhaMask>;

              using FmhaPipeline = ck_tile::BlockFmhaFwdSplitKVPipelineQRKSVS<
                  FmhaPipelineProblem>;

              using FmhaEpilogue =
                  ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
                      typename FmhaFwdTypeConfig<ScalarType>::OaccDataType,
                      typename FmhaFwdTypeConfig<ScalarType>::OaccDataType,
                      kPadSeqLenQ,
                      kPadHeadDim>>;

              using FmhaKernel = ck_tile::FmhaFwdSplitKVKernel<
                  FmhaTilePartitioner,
                  FmhaPipeline,
                  FmhaEpilogue>;

              RunWithFwdSplitKVKernel<FmhaKernel>(param, stream);
            });
      });
    };

    {
      constexpr ck_tile::index_t kM0 = FmhaFwdShape<MaxK>::kM0 / 2;
      constexpr ck_tile::index_t kN1 = FmhaFwdShape<MaxK>::kN1 / 2;

      using FmhaTilePartitioner =
          ck_tile::FmhaFwdSplitKVCombineTilePartitioner<kM0, kN1>;
      constexpr ck_tile::index_t occupancy = -1;

      const bool pad_seqlen_q = !(param.M % kM0 == 0);
      const bool pad_headdim_v = !(param.Kv % kN1 == 0);

      BOOL_SWITCH_2(
          pad_seqlen_q, kPadSeqLenQ, pad_headdim_v, kPadHeadDimV, [&] {
            FMHA_FWD_NUM_KV_SPLITS_SWITCH(
                param.num_kv_splits, kLogMaxSplits, [&] {
                  using FmhaTraits = ck_tile::TileFmhaFwdSplitKVCombineTraits<
                      kPadSeqLenQ,
                      kPadHeadDimV,
                      false, // kStoreLSE
                      false, // kDoFp8StaticQuant place-holder
                      kLogMaxSplits,
                      -1>;

                  using FmhaPipelineProblem =
                      FmhaSplitKVCombinePipelineProblemTemp<
                          kM0,
                          kN1,
                          FmhaTraits>;

                  using FmhaPipeline =
                      ck_tile::BlockFmhaFwdSplitKVCombinePipeline<
                          FmhaPipelineProblem>;

                  using FmhaEpilogue = ck_tile::Default2DEpilogue<
                      ck_tile::Default2DEpilogueProblem<
                          typename FmhaFwdTypeConfig<ScalarType>::OaccDataType,
                          typename FmhaFwdTypeConfig<ScalarType>::ODataType,
                          kPadSeqLenQ,
                          kPadHeadDimV>>;

                  using FmhaKernel = ck_tile::FmhaFwdSplitKVCombineKernel<
                      FmhaTilePartitioner,
                      FmhaPipeline,
                      FmhaEpilogue>;

                  RunWithSplitKVCombineKernel<FmhaKernel>(param, stream);
                });
          });
    };
  };

  template <typename FmhaFwdSplitKVKernel>
  static void RunWithFwdSplitKVKernel(
      BatchedForwardParams& param,
      hipStream_t stream) {
    const auto kargs = [&] {
      return FmhaFwdSplitKVKernel::MakeKargs(
          param.q_ptr,
          param.k_ptr,
          param.v_ptr,
          param.attn_bias_ptr,
          param.logsumexp_acc_ptr,
          param.out_acc_ptr,
          param.B, // batch
          param.M, // seqlen_q
          param.N, // seqlen_k
          nullptr, // seqlen_k_ptr, not used
          param.K, // hdim_q
          param.Kv, // hdim_v
          param.Hq, // nhead_q
          param.Hq / param.Hkv, // nhead_ratio_qk
          param.num_kv_splits, // num_splits
          nullptr, // block_table_ptr, not used
          0, // batch_stride_block_table, not used
          0, // page_table_size, not used
          nullptr, // cache_batch_idx, not used
          param.scale,
          1.0f, // scale_p
          param.q_strides[1], // q, k, v, bias, out_acc tensor seq-dim
                              // stride
          param.k_strides[1],
          param.v_strides[1],
          param.attn_bias_strides[2],
          param.out_acc_strides[2],
          param.q_strides[2], // q, k, v, bias, lse_acc, out_acc tensor
                              // head-dim stride
          param.k_strides[2],
          param.v_strides[2],
          param.attn_bias_strides[1],
          param.lse_acc_strides[2],
          param.out_acc_strides[3],
          param.q_strides[0], // q, k, v, bias, lse_acc, out_acc tensor
                              // batch-dim stride
          param.k_strides[0],
          param.v_strides[0],
          param.attn_bias_strides[0],
          param.lse_acc_strides[1],
          param.out_acc_strides[1],
          param.lse_acc_strides[0], // split_stride_lse_acc
          param.out_acc_strides[0], // split_stride_out_acc
          (param.window_size > 0) ? param.window_size - 1
                                  : -1, // window_left_size
          (param.custom_mask_type == 0) ? -1 : 0, // window_right_size
          param.custom_mask_type);
    }();

    dim3 kGridSize = FmhaFwdSplitKVKernel::GridSize(
        param.B, param.Hq, param.M, param.Kv, param.num_kv_splits);
    constexpr dim3 kBlockSize = FmhaFwdSplitKVKernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = FmhaFwdSplitKVKernel::kBlockPerCu;

    (void)ck_tile::launch_kernel(
        ck_tile::stream_config{stream, false},
        ck_tile::make_kernel<kBlockSize.x, kBlockPerCu>(
            FmhaFwdSplitKVKernel{}, kGridSize, kBlockSize, 0, kargs));
  };

  template <typename FmhaSplitKVCombineKernel>
  static void RunWithSplitKVCombineKernel(
      BatchedForwardParams& param,
      hipStream_t stream) {
    const auto kargs = [&] {
      return FmhaSplitKVCombineKernel::MakeKargs(
          param.logsumexp_acc_ptr,
          param.out_acc_ptr,
          nullptr, // lse_ptr, not used
          param.out_ptr,
          param.B, // batches
          param.M, // seqlen_q
          param.Kv,
          param.num_kv_splits,
          1.0f,
          param.out_acc_strides[2], // row_stride_o_acc
          param.out_strides[1], // row_stride_o
          param.lse_acc_strides[2], // head_stride_lse_acc
          param.out_acc_strides[3], // head_stride_o_acc
          0, // head_stride_lse, // not used
          param.out_strides[2], // head_stride_o
          param.lse_acc_strides[1], // batch_stride_lse_acc
          param.out_acc_strides[1], // batch_stride_o_acc
          0, // batch_stride_lse, not used
          param.out_strides[0], // batch_stride_o
          param.lse_acc_strides[0], // split_stride_lse_acc
          param.out_acc_strides[0]); // split_stride_out_acc
    }();

    dim3 kGridSize = FmhaSplitKVCombineKernel::GridSize(
        param.B, param.Hq, param.M, param.Kv);
    constexpr dim3 kBlockSize = FmhaSplitKVCombineKernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu =
        FmhaSplitKVCombineKernel::kBlockPerCu;

    (void)ck_tile::launch_kernel(
        ck_tile::stream_config{stream, false},
        ck_tile::make_kernel<kBlockSize.x, kBlockPerCu>(
            FmhaSplitKVCombineKernel{}, kGridSize, kBlockSize, 0, kargs));
  };
};
