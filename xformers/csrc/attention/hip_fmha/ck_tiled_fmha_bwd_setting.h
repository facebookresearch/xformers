/*
 * Copyright (c) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ck_tile/core.hpp>
#include <ck_tile/ops/fmha.hpp>

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
  using type = ck_tile::sequence<128, 128, 32, 32, 32, 32, 32, 32, 32>;
  using gemm02_warps = ck_tile::sequence<1, 4, 1>; // default for gemm0/gemm2
  using gemm13_warps = ck_tile::sequence<4, 1, 1>; // default for gemm1/gemm3
  using gemm4_warps = ck_tile::sequence<4, 1, 1>; // default for gemm4
};

template <>
struct FmhaBwdBlockTile<64> {
  using type = ck_tile::sequence<64, 128, 32, 32, 32, 32, 32, 64, 64>;
  using gemm02_warps = ck_tile::sequence<1, 4, 1>; // default for gemm0/gemm2
  using gemm13_warps = ck_tile::sequence<4, 1, 1>; // default for gemm1/gemm3
  using gemm4_warps = ck_tile::sequence<2, 2, 1>; // default for gemm4
};

template <>
struct FmhaBwdBlockTile<128> {
  using type = ck_tile::sequence<64, 128, 32, 32, 32, 32, 32, 128, 128>;
  using gemm02_warps = ck_tile::sequence<1, 4, 1>; // default for gemm0/gemm2
  using gemm13_warps = ck_tile::sequence<4, 1, 1>; // default for gemm1/gemm3
  using gemm4_warps = ck_tile::sequence<2, 2, 1>; // default for gemm4
};

using FmhaBwdWarpTile = ck_tile::sequence<32, 32, 16>;

template <ck_tile::index_t MaxK>
struct FmhaBwdShape;

template <>
struct FmhaBwdShape<32> : ck_tile::TileFmhaBwdShape<
                              typename FmhaBwdBlockTile<32>::type,
                              typename FmhaBwdBlockTile<32>::gemm02_warps,
                              FmhaBwdWarpTile,
                              typename FmhaBwdBlockTile<32>::gemm13_warps,
                              FmhaBwdWarpTile,
                              typename FmhaBwdBlockTile<32>::gemm02_warps,
                              FmhaBwdWarpTile,
                              typename FmhaBwdBlockTile<32>::gemm13_warps,
                              FmhaBwdWarpTile,
                              typename FmhaBwdBlockTile<32>::gemm4_warps,
                              FmhaBwdWarpTile> {};

template <>
struct FmhaBwdShape<64> : ck_tile::TileFmhaBwdShape<
                              typename FmhaBwdBlockTile<64>::type,
                              typename FmhaBwdBlockTile<64>::gemm02_warps,
                              FmhaBwdWarpTile,
                              typename FmhaBwdBlockTile<64>::gemm13_warps,
                              FmhaBwdWarpTile,
                              typename FmhaBwdBlockTile<64>::gemm02_warps,
                              FmhaBwdWarpTile,
                              typename FmhaBwdBlockTile<64>::gemm13_warps,
                              FmhaBwdWarpTile,
                              typename FmhaBwdBlockTile<64>::gemm4_warps,
                              FmhaBwdWarpTile> {};

template <>
struct FmhaBwdShape<128> : ck_tile::TileFmhaBwdShape<
                               typename FmhaBwdBlockTile<128>::type,
                               typename FmhaBwdBlockTile<128>::gemm02_warps,
                               FmhaBwdWarpTile,
                               typename FmhaBwdBlockTile<128>::gemm13_warps,
                               FmhaBwdWarpTile,
                               typename FmhaBwdBlockTile<128>::gemm02_warps,
                               FmhaBwdWarpTile,
                               typename FmhaBwdBlockTile<128>::gemm13_warps,
                               FmhaBwdWarpTile,
                               typename FmhaBwdBlockTile<128>::gemm4_warps,
                               FmhaBwdWarpTile> {};

template <ck_tile::index_t MaxK>
struct FmhaBwdPipelineEnumSelector;

template <>
struct FmhaBwdPipelineEnumSelector<32> {
  static constexpr ck_tile::BlockFmhaBwdPipelineEnum value =
      ck_tile::BlockFmhaBwdPipelineEnum::QSKSVROGradS;
};

template <>
struct FmhaBwdPipelineEnumSelector<64> {
  static constexpr ck_tile::BlockFmhaBwdPipelineEnum value =
      ck_tile::BlockFmhaBwdPipelineEnum::KSKTSVR;
};

template <>
struct FmhaBwdPipelineEnumSelector<128> {
  static constexpr ck_tile::BlockFmhaBwdPipelineEnum value =
      ck_tile::BlockFmhaBwdPipelineEnum::KSVR;
};

template <ck_tile::BlockFmhaBwdPipelineEnum value, typename problem>
struct FmhaBwdPipelineMaker;

template <typename problem>
struct FmhaBwdPipelineMaker<
    ck_tile::BlockFmhaBwdPipelineEnum::QSKSVROGradS,
    problem> {
  using pipeline = ck_tile::BlockFmhaBwdDQDKDVPipelineQSKSVROGradS<problem>;
};

template <typename problem>
struct FmhaBwdPipelineMaker<
    ck_tile::BlockFmhaBwdPipelineEnum::KSKTSVR,
    problem> {
  using pipeline = ck_tile::BlockFmhaBwdDQDKDVPipelineKSKTSVR<problem>;
};

template <typename problem>
struct FmhaBwdPipelineMaker<ck_tile::BlockFmhaBwdPipelineEnum::KSVR, problem> {
  using pipeline = ck_tile::BlockFmhaBwdDQDKDVPipelineKSVR<problem>;
};
