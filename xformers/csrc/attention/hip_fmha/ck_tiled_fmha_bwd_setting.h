/*
 * Copyright (c) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ck/tile_program/tile/tile_fmha_shape.hpp>

template <typename DataType>
struct FmhaBwdTypeConfig;

template <>
struct FmhaBwdTypeConfig<ck::half_t> {
  using QDataType = ck::half_t;
  using KDataType = ck::half_t;
  using VDataType = ck::half_t;
  using GemmDataType = ck::half_t;
  using BiasDataType = ck::half_t;
  using RandValOutputDataType = unsigned short;
  using LSEDataType = float;
  using AccDataType = float; // data type for gemm accumulation
  using DDataType = float;
  using ODataType = ck::half_t;
  using OGradDataType = ck::half_t;
  using QGradDataType = ck::half_t;
  using KGradDataType = ck::half_t;
  using VGradDataType = ck::half_t;
  using BiasGradDataType = ck::half_t;
};

template <>
struct FmhaBwdTypeConfig<ck::bhalf_t> {
  using QDataType = ck::bhalf_t;
  using KDataType = ck::bhalf_t;
  using VDataType = ck::bhalf_t;
  using GemmDataType = ck::bhalf_t;
  using BiasDataType = ck::bhalf_t;
  using RandValOutputDataType = unsigned short;
  using LSEDataType = float;
  using AccDataType = float; // data type for gemm accumulation
  using DDataType = float;
  using ODataType = ck::bhalf_t;
  using OGradDataType = ck::bhalf_t;
  using QGradDataType = ck::bhalf_t;
  using KGradDataType = ck::bhalf_t;
  using VGradDataType = ck::bhalf_t;
  using BiasGradDataType = ck::bhalf_t;
};

template <ck::index_t MaxK>
struct FmhaBwdLoadStrategy;

template <>
struct FmhaBwdLoadStrategy<32> {
  using type = ck::Sequence<true, false, true, false, true, true, false>;
};

template <>
struct FmhaBwdLoadStrategy<64> {
  using type = ck::Sequence<false, false, true, true, true, false, false>;
};

template <>
struct FmhaBwdLoadStrategy<128> {
  using type = ck::Sequence<false, false, true, false, true, false, false>;
};

template <ck::index_t MaxK>
struct FmhaBwdBlockTile;

template <>
struct FmhaBwdBlockTile<32> {
  using type = ck::Sequence<128, 128, 32, 32, 32, 32, 32, 32, 32>;
};

template <>
struct FmhaBwdBlockTile<64> {
  using type = ck::Sequence<64, 128, 32, 32, 32, 32, 32, 64, 64>;
};

template <>
struct FmhaBwdBlockTile<128> {
  using type = ck::Sequence<64, 128, 32, 32, 32, 32, 32, 128, 128>;
};

using FmhaBwdBlockWarps0 = ck::Sequence<1, 4, 1>; // default for gemm0/gemm2
using FmhaBwdBlockWarps1 = ck::Sequence<4, 1, 1>; // default for gemm1/gemm3
using FmhaBwdBlockWarps2 = ck::Sequence<2, 2, 1>; // default for gemm4
using FmhaBwdWarpTile = ck::Sequence<32, 32, 16>;

template <ck::index_t MaxK>
struct FmhaBwdShape;

template <>
struct FmhaBwdShape<32> : ck::tile_program::TileFmhaBwdShape<
                              typename FmhaBwdBlockTile<32>::type,
                              typename FmhaBwdLoadStrategy<32>::type,
                              FmhaBwdBlockWarps0,
                              FmhaBwdWarpTile,
                              FmhaBwdBlockWarps1,
                              FmhaBwdWarpTile,
                              FmhaBwdBlockWarps0,
                              FmhaBwdWarpTile,
                              FmhaBwdBlockWarps1,
                              FmhaBwdWarpTile,
                              ck::Sequence<4, 1, 1>,
                              FmhaBwdWarpTile> {};

template <>
struct FmhaBwdShape<64> : ck::tile_program::TileFmhaBwdShape<
                              typename FmhaBwdBlockTile<64>::type,
                              typename FmhaBwdLoadStrategy<64>::type,
                              FmhaBwdBlockWarps0,
                              FmhaBwdWarpTile,
                              FmhaBwdBlockWarps1,
                              FmhaBwdWarpTile,
                              FmhaBwdBlockWarps0,
                              FmhaBwdWarpTile,
                              FmhaBwdBlockWarps1,
                              FmhaBwdWarpTile,
                              FmhaBwdBlockWarps2,
                              FmhaBwdWarpTile> {};

template <>
struct FmhaBwdShape<128> : ck::tile_program::TileFmhaBwdShape<
                               typename FmhaBwdBlockTile<128>::type,
                               typename FmhaBwdLoadStrategy<128>::type,
                               FmhaBwdBlockWarps0,
                               FmhaBwdWarpTile,
                               FmhaBwdBlockWarps1,
                               FmhaBwdWarpTile,
                               FmhaBwdBlockWarps0,
                               FmhaBwdWarpTile,
                               FmhaBwdBlockWarps1,
                               FmhaBwdWarpTile,
                               FmhaBwdBlockWarps2,
                               FmhaBwdWarpTile> {};
