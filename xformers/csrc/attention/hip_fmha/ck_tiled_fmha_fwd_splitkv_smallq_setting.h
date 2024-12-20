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
struct FmhaFwdSplitKVSmallQBlockTile;

// Tile-sizes: M N0 K0 N1 K1 MaxK (MaxK % K0 == 0, MaxK % N1 == 0, N0 % K1 == 0)

template <>
struct FmhaFwdSplitKVSmallQBlockTile<32> {
  using type = ck_tile::sequence<16, 64, 16, 32, 32, 32>;
  using gemm0_warps = ck_tile::sequence<1, 2, 1>;
  using gemm1_warps = ck_tile::sequence<1, 2, 1>;
};

template <>
struct FmhaFwdSplitKVSmallQBlockTile<64> {
  using type = ck_tile::sequence<16, 64, 32, 64, 32, 64>;
  using gemm0_warps = ck_tile::sequence<1, 4, 1>;
  using gemm1_warps = ck_tile::sequence<1, 4, 1>;
};

template <>
struct FmhaFwdSplitKVSmallQBlockTile<96> {
  using type = ck_tile::sequence<16, 64, 32, 128, 32, 96>;
  using gemm0_warps = ck_tile::sequence<1, 4, 1>;
  using gemm1_warps = ck_tile::sequence<1, 4, 1>;
};

template <>
struct FmhaFwdSplitKVSmallQBlockTile<128> {
  using type = ck_tile::sequence<16, 64, 64, 128, 64, 128>;
  using gemm0_warps = ck_tile::sequence<1, 4, 1>;
  using gemm1_warps = ck_tile::sequence<1, 4, 1>;
};

template <>
struct FmhaFwdSplitKVSmallQBlockTile<256> {
  using type = ck_tile::sequence<16, 64, 64, 256, 64, 256>;
  using gemm0_warps = ck_tile::sequence<1, 4, 1>;
  using gemm1_warps = ck_tile::sequence<1, 4, 1>;
};

using FmhaFwdSplitKVSmallQWarpTile0 = ck_tile::sequence<16, 16, 16>;
using FmhaFwdSplitKVSmallQWarpTile1 = ck_tile::sequence<16, 16, 16>;

template <ck_tile::index_t MaxK>
struct FmhaFwdSplitKVSmallQShape;

template <>
struct FmhaFwdSplitKVSmallQShape<32> {
  using Type = ck_tile::TileFmhaShape<
      typename FmhaFwdSplitKVSmallQBlockTile<32>::type,
      typename FmhaFwdSplitKVSmallQBlockTile<32>::gemm0_warps,
      FmhaFwdSplitKVSmallQWarpTile0,
      typename FmhaFwdSplitKVSmallQBlockTile<32>::gemm1_warps,
      FmhaFwdSplitKVSmallQWarpTile1,
      IsVLayoutRowMajor>;
};

template <>
struct FmhaFwdSplitKVSmallQShape<64> {
  using Type = ck_tile::TileFmhaShape<
      typename FmhaFwdSplitKVSmallQBlockTile<64>::type,
      typename FmhaFwdSplitKVSmallQBlockTile<64>::gemm0_warps,
      FmhaFwdSplitKVSmallQWarpTile0,
      typename FmhaFwdSplitKVSmallQBlockTile<64>::gemm1_warps,
      FmhaFwdSplitKVSmallQWarpTile1,
      IsVLayoutRowMajor>;
};

template <>
struct FmhaFwdSplitKVSmallQShape<96> {
  using Type = ck_tile::TileFmhaShape<
      typename FmhaFwdSplitKVSmallQBlockTile<96>::type,
      typename FmhaFwdSplitKVSmallQBlockTile<96>::gemm0_warps,
      FmhaFwdSplitKVSmallQWarpTile0,
      typename FmhaFwdSplitKVSmallQBlockTile<96>::gemm1_warps,
      FmhaFwdSplitKVSmallQWarpTile1,
      IsVLayoutRowMajor>;
};

template <>
struct FmhaFwdSplitKVSmallQShape<128> {
  using Type = ck_tile::TileFmhaShape<
      typename FmhaFwdSplitKVSmallQBlockTile<128>::type,
      typename FmhaFwdSplitKVSmallQBlockTile<128>::gemm0_warps,
      FmhaFwdSplitKVSmallQWarpTile0,
      typename FmhaFwdSplitKVSmallQBlockTile<128>::gemm1_warps,
      FmhaFwdSplitKVSmallQWarpTile1,
      IsVLayoutRowMajor>;
};

template <>
struct FmhaFwdSplitKVSmallQShape<256> {
  using Type = ck_tile::TileFmhaShape<
      typename FmhaFwdSplitKVSmallQBlockTile<256>::type,
      typename FmhaFwdSplitKVSmallQBlockTile<256>::gemm0_warps,
      FmhaFwdSplitKVSmallQWarpTile0,
      typename FmhaFwdSplitKVSmallQBlockTile<256>::gemm1_warps,
      FmhaFwdSplitKVSmallQWarpTile1,
      IsVLayoutRowMajor>;
};

template <ck_tile::index_t MaxK>
int fwd_splitkv_smallq_get_mtile_size() {
  using FmhaTileShape = typename FmhaFwdSplitKVSmallQShape<MaxK>::Type;

  return FmhaTileShape::kM0;
};

static int get_mtile_size_for_splitkv_smallq(int max_headdim) {
  int mtile_size_for_splitkv_smallq = 16;

  if (max_headdim <= 32) {
    mtile_size_for_splitkv_smallq = fwd_splitkv_smallq_get_mtile_size<32>();
  } else if (max_headdim <= 64) {
    mtile_size_for_splitkv_smallq = fwd_splitkv_smallq_get_mtile_size<64>();
  } else if (max_headdim <= 96) {
    mtile_size_for_splitkv_smallq = fwd_splitkv_smallq_get_mtile_size<96>();
  } else if (max_headdim <= 128) {
    mtile_size_for_splitkv_smallq = fwd_splitkv_smallq_get_mtile_size<128>();
  } else {
    mtile_size_for_splitkv_smallq = fwd_splitkv_smallq_get_mtile_size<256>();
  };

  return mtile_size_for_splitkv_smallq;
};
