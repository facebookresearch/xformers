/*
 * Copyright (c) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ck_tile/core.hpp>
#include <ck_tile/ops/fmha.hpp>
#include "ck_tiled_fmha_fwd_type_config.h"

template <ck_tile::index_t MaxK>
struct FmhaFwdBlockTile;

// Tile-sizes: M N0 K0 N1 K1 MaxK (MaxK % K0 == 0, MaxK % N1 == 0, N0 % K1 == 0)
//
template <>
struct FmhaFwdBlockTile<32> {
  using type = ck_tile::sequence<128, 64, 16, 32, 32, 32>;
  using gemm0_warps = ck_tile::sequence<2, 1, 1>;
  using gemm1_warps = ck_tile::sequence<2, 1, 1>;
};

template <>
struct FmhaFwdBlockTile<64> {
  using type = ck_tile::sequence<128, 64, 32, 64, 32, 64>;
  using gemm0_warps = ck_tile::sequence<4, 1, 1>;
  using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <>
struct FmhaFwdBlockTile<96> {
  using type = ck_tile::sequence<128, 128, 32, 128, 32, 96>;
  using gemm0_warps = ck_tile::sequence<4, 1, 1>;
  using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <>
struct FmhaFwdBlockTile<128> {
  using type = ck_tile::sequence<128, 128, 32, 128, 32, 128>;
  using gemm0_warps = ck_tile::sequence<4, 1, 1>;
  using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <>
struct FmhaFwdBlockTile<256> {
  using type = ck_tile::sequence<128, 128, 32, 256, 32, 256>;
  using gemm0_warps = ck_tile::sequence<4, 1, 1>;
  using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

using FmhaFwdWarpTile = ck_tile::sequence<32, 32, 16>;

template <ck_tile::index_t MaxK>
struct FmhaFwdShape;

template <>
struct FmhaFwdShape<32> : ck_tile::TileFmhaShape<
                              typename FmhaFwdBlockTile<32>::type,
                              typename FmhaFwdBlockTile<32>::gemm0_warps,
                              FmhaFwdWarpTile,
                              typename FmhaFwdBlockTile<32>::gemm1_warps,
                              FmhaFwdWarpTile,
                              IsVLayoutRowMajor> {};

template <>
struct FmhaFwdShape<64> : ck_tile::TileFmhaShape<
                              typename FmhaFwdBlockTile<64>::type,
                              typename FmhaFwdBlockTile<64>::gemm0_warps,
                              FmhaFwdWarpTile,
                              typename FmhaFwdBlockTile<64>::gemm1_warps,
                              FmhaFwdWarpTile,
                              IsVLayoutRowMajor> {};

template <>
struct FmhaFwdShape<96> : ck_tile::TileFmhaShape<
                              typename FmhaFwdBlockTile<96>::type,
                              typename FmhaFwdBlockTile<96>::gemm0_warps,
                              FmhaFwdWarpTile,
                              typename FmhaFwdBlockTile<96>::gemm1_warps,
                              FmhaFwdWarpTile,
                              IsVLayoutRowMajor> {};

template <>
struct FmhaFwdShape<128> : ck_tile::TileFmhaShape<
                               typename FmhaFwdBlockTile<128>::type,
                               typename FmhaFwdBlockTile<128>::gemm0_warps,
                               FmhaFwdWarpTile,
                               typename FmhaFwdBlockTile<128>::gemm1_warps,
                               FmhaFwdWarpTile,
                               IsVLayoutRowMajor> {};

template <>
struct FmhaFwdShape<256> : ck_tile::TileFmhaShape<
                               typename FmhaFwdBlockTile<256>::type,
                               typename FmhaFwdBlockTile<256>::gemm0_warps,
                               FmhaFwdWarpTile,
                               typename FmhaFwdBlockTile<256>::gemm1_warps,
                               FmhaFwdWarpTile,
                               IsVLayoutRowMajor> {};

template <ck_tile::index_t MaxK>
int fwd_get_mtile_size() {
  using FmhaTileShape = FmhaFwdShape<MaxK>;

  return FmhaTileShape::kM0;
};
