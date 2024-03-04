/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ck/tile_program/tile/tile_fmha_shape.hpp>

enum struct CausalMaskType {
  MaskDisabled,
  MaskUpperTriangleFromTopLeft,
  MaskUpperTriangleFromBottomRight
};

template <typename DataType>
struct FmhaFwdTypeConfig;

template <>
struct FmhaFwdTypeConfig<ck::half_t> {
  using QDataType = ck::half_t;
  using KDataType = ck::half_t;
  using VDataType = ck::half_t;
  using BiasDataType = ck::half_t;
  using LSEDataType =
      float; // data type for lse(logsumexp L_j = max_j + log(l_j))
  using SaccDataType = float; // data type for first gemm accumulation
  using SMPLComputeDataType = float; // data type for reduction, softmax
  using PDataType = ck::half_t; // data type for A matrix of second gemm
  using OaccDataType = float; // data type for second gemm accumulation
  using ODataType = ck::half_t;
};

template <>
struct FmhaFwdTypeConfig<ck::bhalf_t> {
  using QDataType = ck::bhalf_t;
  using KDataType = ck::bhalf_t;
  using VDataType = ck::bhalf_t;
  using BiasDataType = ck::bhalf_t;
  using LSEDataType =
      float; // data type for lse(logsumexp L_j = max_j + log(l_j))
  using SaccDataType = float; // data type for first gemm accumulation
  using SMPLComputeDataType = float; // data type for reduction, softmax
  using PDataType = ck::bhalf_t; // data type for A matrix of second gemm
  using OaccDataType = float; // data type for second gemm accumulation
  using ODataType = ck::bhalf_t;
};

template <ck::index_t MaxK>
struct FmhaFwdBlockTile;

template <>
struct FmhaFwdBlockTile<32> {
  using type = ck::Sequence<128, 64, 16, 32, 32, 32>;
};

template <>
struct FmhaFwdBlockTile<64> {
  using type = ck::Sequence<128, 64, 32, 64, 32, 64>;
};

template <>
struct FmhaFwdBlockTile<128> {
  using type = ck::Sequence<128, 128, 32, 128, 32, 128>;
};

template <>
struct FmhaFwdBlockTile<256> {
  using type = ck::Sequence<128, 128, 32, 256, 32, 256>;
};

using FmhaFwdBlockWarps = ck::Sequence<4, 1, 1>;
using FmhaFwdWarpTile = ck::Sequence<32, 32, 16>;

static constexpr bool IsVLayoutRowMajor = true;

template <ck::index_t MaxK>
struct FmhaFwdShape;

template <>
struct FmhaFwdShape<32> : ck::tile_program::TileFmhaShape<
                              typename FmhaFwdBlockTile<32>::type,
                              ck::Sequence<2, 1, 1>,
                              FmhaFwdWarpTile,
                              ck::Sequence<2, 1, 1>,
                              FmhaFwdWarpTile,
                              IsVLayoutRowMajor> {};

template <>
struct FmhaFwdShape<64> : ck::tile_program::TileFmhaShape<
                              typename FmhaFwdBlockTile<64>::type,
                              FmhaFwdBlockWarps,
                              FmhaFwdWarpTile,
                              FmhaFwdBlockWarps,
                              FmhaFwdWarpTile,
                              IsVLayoutRowMajor> {};

template <>
struct FmhaFwdShape<128> : ck::tile_program::TileFmhaShape<
                               typename FmhaFwdBlockTile<128>::type,
                               FmhaFwdBlockWarps,
                               FmhaFwdWarpTile,
                               FmhaFwdBlockWarps,
                               FmhaFwdWarpTile,
                               IsVLayoutRowMajor> {};

template <>
struct FmhaFwdShape<256> : ck::tile_program::TileFmhaShape<
                               typename FmhaFwdBlockTile<256>::type,
                               FmhaFwdBlockWarps,
                               FmhaFwdWarpTile,
                               FmhaFwdBlockWarps,
                               FmhaFwdWarpTile,
                               IsVLayoutRowMajor> {};
