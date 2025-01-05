/*
 * Copyright (c) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ck_tile/core.hpp>
#include <ck_tile/ops/fmha.hpp>
#include <ck_tile/ops/fmha/block/block_dropout.hpp>

template <typename DataType>
struct FmhaBwdTypeConfig;

template <>
struct FmhaBwdTypeConfig<ck_tile::fp16_t> {
  using QDataType = ck_tile::fp16_t;
  using KDataType = ck_tile::fp16_t;
  using VDataType = ck_tile::fp16_t;
  using GemmDataType = ck_tile::fp16_t;
  using BiasDataType = ck_tile::fp16_t;
  using RandValOutputDataType = unsigned short;
  using LSEDataType = float;
  using AccDataType = float; // data type for gemm accumulation
  using DDataType = float;
  using ODataType = ck_tile::fp16_t;
  using OGradDataType = ck_tile::fp16_t;
  using QGradDataType = ck_tile::fp16_t;
  using KGradDataType = ck_tile::fp16_t;
  using VGradDataType = ck_tile::fp16_t;
  using BiasGradDataType = ck_tile::fp16_t;
};

template <>
struct FmhaBwdTypeConfig<ck_tile::bf16_t> {
  using QDataType = ck_tile::bf16_t;
  using KDataType = ck_tile::bf16_t;
  using VDataType = ck_tile::bf16_t;
  using GemmDataType = ck_tile::bf16_t;
  using BiasDataType = ck_tile::bf16_t;
  using RandValOutputDataType = unsigned short;
  using LSEDataType = float;
  using AccDataType = float; // data type for gemm accumulation
  using DDataType = float;
  using ODataType = ck_tile::bf16_t;
  using OGradDataType = ck_tile::bf16_t;
  using QGradDataType = ck_tile::bf16_t;
  using KGradDataType = ck_tile::bf16_t;
  using VGradDataType = ck_tile::bf16_t;
  using BiasGradDataType = ck_tile::bf16_t;
};

template <ck_tile::index_t MaxK>
struct FmhaBwdBlockTile;

template <>
struct FmhaBwdBlockTile<32> {
  using tile_lengths = ck_tile::sequence<32, 128, 32, 32, 32, 32, 64, 32, 32>;
  using gemm02_warps = ck_tile::sequence<1, 4, 1>; // default for gemm0/gemm2
  using gemm13_warps = ck_tile::sequence<4, 1, 1>; // default for gemm1/gemm3
  using gemm4_warps = ck_tile::sequence<2, 2, 1>; // default for gemm4
};

template <>
struct FmhaBwdBlockTile<64> {
  using tile_lengths = ck_tile::sequence<32, 128, 64, 32, 64, 32, 32, 64, 64>;
  using gemm02_warps = ck_tile::sequence<1, 4, 1>; // default for gemm0/gemm2
  using gemm13_warps = ck_tile::sequence<4, 1, 1>; // default for gemm1/gemm3
  using gemm4_warps = ck_tile::sequence<1, 4, 1>; // default for gemm4
};

template <>
struct FmhaBwdBlockTile<96> {
  using tile_lengths = ck_tile::sequence<16, 128, 96, 16, 96, 16, 32, 128, 128>;
  using gemm02_warps = ck_tile::sequence<1, 4, 1>; // default for gemm0/gemm2
  using gemm13_warps = ck_tile::sequence<4, 1, 1>; // default for gemm1/gemm3
  using gemm4_warps = ck_tile::sequence<1, 4, 1>; // default for gemm4
};

template <>
struct FmhaBwdBlockTile<128> {
  using tile_lengths =
      ck_tile::sequence<16, 128, 128, 16, 128, 16, 32, 128, 128>;
  using gemm02_warps = ck_tile::sequence<1, 4, 1>; // default for gemm0/gemm2
  using gemm13_warps = ck_tile::sequence<4, 1, 1>; // default for gemm1/gemm3
  using gemm4_warps = ck_tile::sequence<1, 4, 1>; // default for gemm4
};

template <>
struct FmhaBwdBlockTile<256> {
  using tile_lengths =
      ck_tile::sequence<16, 64, 256, 16, 256, 16, 32, 256, 256>;
  using gemm02_warps = ck_tile::sequence<1, 4, 1>; // default for gemm0/gemm2
  using gemm13_warps = ck_tile::sequence<4, 1, 1>; // default for gemm1/gemm3
  using gemm4_warps = ck_tile::sequence<1, 4, 1>; // default for gemm4
};

using FmhaBwdWarpTile1 = ck_tile::sequence<32, 32, 16>;
using FmhaBwdWarpTile2 = ck_tile::sequence<16, 16, 32>;
using FmhaBwdWarpTile3 = ck_tile::sequence<16, 16, 16>;

template <ck_tile::index_t MaxK>
struct FmhaBwdShape;

template <>
struct FmhaBwdShape<32> : ck_tile::TileFmhaBwdShape<
                              typename FmhaBwdBlockTile<32>::tile_lengths,
                              typename FmhaBwdBlockTile<32>::gemm02_warps,
                              FmhaBwdWarpTile2,
                              typename FmhaBwdBlockTile<32>::gemm13_warps,
                              FmhaBwdWarpTile3,
                              typename FmhaBwdBlockTile<32>::gemm02_warps,
                              FmhaBwdWarpTile2,
                              typename FmhaBwdBlockTile<32>::gemm13_warps,
                              FmhaBwdWarpTile3,
                              typename FmhaBwdBlockTile<32>::gemm4_warps,
                              FmhaBwdWarpTile2> {};

template <>
struct FmhaBwdShape<64> : ck_tile::TileFmhaBwdShape<
                              typename FmhaBwdBlockTile<64>::tile_lengths,
                              typename FmhaBwdBlockTile<64>::gemm02_warps,
                              FmhaBwdWarpTile2,
                              typename FmhaBwdBlockTile<64>::gemm13_warps,
                              FmhaBwdWarpTile3,
                              typename FmhaBwdBlockTile<64>::gemm02_warps,
                              FmhaBwdWarpTile2,
                              typename FmhaBwdBlockTile<64>::gemm13_warps,
                              FmhaBwdWarpTile3,
                              typename FmhaBwdBlockTile<64>::gemm4_warps,
                              FmhaBwdWarpTile2> {};

template <>
struct FmhaBwdShape<96> : ck_tile::TileFmhaBwdShape<
                              typename FmhaBwdBlockTile<96>::tile_lengths,
                              typename FmhaBwdBlockTile<96>::gemm02_warps,
                              FmhaBwdWarpTile2,
                              typename FmhaBwdBlockTile<96>::gemm13_warps,
                              FmhaBwdWarpTile3,
                              typename FmhaBwdBlockTile<96>::gemm02_warps,
                              FmhaBwdWarpTile2,
                              typename FmhaBwdBlockTile<96>::gemm13_warps,
                              FmhaBwdWarpTile3,
                              typename FmhaBwdBlockTile<96>::gemm4_warps,
                              FmhaBwdWarpTile2> {};

template <>
struct FmhaBwdShape<128> : ck_tile::TileFmhaBwdShape<
                               typename FmhaBwdBlockTile<128>::tile_lengths,
                               typename FmhaBwdBlockTile<128>::gemm02_warps,
                               FmhaBwdWarpTile2,
                               typename FmhaBwdBlockTile<128>::gemm13_warps,
                               FmhaBwdWarpTile3,
                               typename FmhaBwdBlockTile<128>::gemm02_warps,
                               FmhaBwdWarpTile2,
                               typename FmhaBwdBlockTile<128>::gemm13_warps,
                               FmhaBwdWarpTile3,
                               typename FmhaBwdBlockTile<128>::gemm4_warps,
                               FmhaBwdWarpTile2> {};

template <>
struct FmhaBwdShape<256> : ck_tile::TileFmhaBwdShape<
                               typename FmhaBwdBlockTile<256>::tile_lengths,
                               typename FmhaBwdBlockTile<256>::gemm02_warps,
                               FmhaBwdWarpTile2,
                               typename FmhaBwdBlockTile<256>::gemm13_warps,
                               FmhaBwdWarpTile3,
                               typename FmhaBwdBlockTile<256>::gemm02_warps,
                               FmhaBwdWarpTile2,
                               typename FmhaBwdBlockTile<256>::gemm13_warps,
                               FmhaBwdWarpTile3,
                               typename FmhaBwdBlockTile<256>::gemm4_warps,
                               FmhaBwdWarpTile2> {};

template <ck_tile::index_t MaxK>
struct FmhaBwdPipelineEnumSelector {
  static constexpr ck_tile::BlockFmhaBwdPipelineEnum value =
      ck_tile::BlockFmhaBwdPipelineEnum::KRKTRVR_IGLP;
};

template <ck_tile::BlockFmhaBwdPipelineEnum value, typename problem>
struct FmhaBwdPipelineMaker;

template <typename problem>
struct FmhaBwdPipelineMaker<
    ck_tile::BlockFmhaBwdPipelineEnum::KRKTRVR,
    problem> {
  using pipeline = ck_tile::BlockFmhaBwdDQDKDVPipelineKRKTRVR<problem>;
};

template <typename problem>
struct FmhaBwdPipelineMaker<
    ck_tile::BlockFmhaBwdPipelineEnum::KRKTRVR_IGLP,
    problem> {
  using pipeline = ck_tile::BlockFmhaBwdDQDKDVPipelineKRKTRVRIGLP<problem>;
};

template <bool kHasDropout, ck_tile::index_t MaxK>
struct FmhaBwdBlockDropoutMaker;

template <ck_tile::index_t MaxK>
struct FmhaBwdBlockDropoutMaker<false, MaxK> {
  using dropout = ck_tile::BlockDropoutBwd<false, true, false>;
};

template <ck_tile::index_t MaxK>
struct FmhaBwdBlockDropoutMaker<true, MaxK> {
  using FmhaBwdShapeType = FmhaBwdShape<MaxK>;
  static constexpr bool IsWG32 =
      (FmhaBwdShapeType::Gemm0WarpTile::at(ck_tile::number<0>{}) == 32);
  using dropout = ck_tile::BlockDropoutBwd<true, IsWG32, false>;
};
