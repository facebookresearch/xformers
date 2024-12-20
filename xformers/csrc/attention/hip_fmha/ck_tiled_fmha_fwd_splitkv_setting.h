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
#include "ck_tiled_fmha_seqlen_q_switch.h"

template <ck_tile::index_t MaxK, ck_tile::index_t MaxSeqLenQ = 0>
struct FmhaFwdSplitKVBlockTile;

// Tile-sizes: M N0 K0 N1 K1 MaxK (MaxK % K0 == 0, MaxK % N1 == 0, N0 % K1 == 0)

template <ck_tile::index_t MaxSeqLenQ>
struct FmhaFwdSplitKVBlockTile<32, MaxSeqLenQ> {
  using type = ck_tile::sequence<32, 64, 16, 32, 32, 32>;
  using gemm0_warps = ck_tile::sequence<2, 1, 1>;
  using gemm1_warps = ck_tile::sequence<2, 1, 1>;
};

template struct FmhaFwdSplitKVBlockTile<32>;

template <ck_tile::index_t MaxSeqLenQ>
struct FmhaFwdSplitKVBlockTile<64, MaxSeqLenQ> {
  using type = ck_tile::sequence<32, 64, 32, 64, 32, 64>;
  using gemm0_warps = ck_tile::sequence<2, 1, 1>;
  using gemm1_warps = ck_tile::sequence<2, 1, 1>;
};

template struct FmhaFwdSplitKVBlockTile<64>;

template <ck_tile::index_t MaxSeqLenQ>
struct FmhaFwdSplitKVBlockTile<96, MaxSeqLenQ> {
  using type = ck_tile::sequence<64, 128, 32, 128, 32, 96>;
  using gemm0_warps = ck_tile::sequence<4, 1, 1>;
  using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template struct FmhaFwdSplitKVBlockTile<96>;

template <>
struct FmhaFwdSplitKVBlockTile<128, 32> {
  using type = ck_tile::sequence<32, 128, 32, 128, 32, 128>;
  using gemm0_warps = ck_tile::sequence<2, 1, 1>;
  using gemm1_warps = ck_tile::sequence<2, 1, 1>;
};

template <>
struct FmhaFwdSplitKVBlockTile<128, 64> {
  using type = ck_tile::sequence<64, 128, 32, 128, 32, 128>;
  using gemm0_warps = ck_tile::sequence<4, 1, 1>;
  using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <ck_tile::index_t MaxSeqLenQ>
struct FmhaFwdSplitKVBlockTile<256, MaxSeqLenQ> {
  using type = ck_tile::sequence<64, 128, 32, 256, 32, 256>;
  using gemm0_warps = ck_tile::sequence<4, 1, 1>;
  using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template struct FmhaFwdSplitKVBlockTile<256>;

using FmhaFwdSplitKVWarpTile = ck_tile::sequence<16, 16, 16>;

template <ck_tile::index_t MaxK, ck_tile::index_t MaxSeqLenQ>
struct FmhaFwdSplitKVShape;

template <ck_tile::index_t MaxSeqLenQ>
struct FmhaFwdSplitKVShape<32, MaxSeqLenQ> {
  using Type = ck_tile::TileFmhaShape<
      typename FmhaFwdSplitKVBlockTile<32>::type,
      typename FmhaFwdSplitKVBlockTile<32>::gemm0_warps,
      FmhaFwdSplitKVWarpTile,
      typename FmhaFwdSplitKVBlockTile<32>::gemm1_warps,
      FmhaFwdSplitKVWarpTile,
      IsVLayoutRowMajor>;
};

template struct FmhaFwdSplitKVShape<32, 32>;
template struct FmhaFwdSplitKVShape<32, 64>;

template <ck_tile::index_t MaxSeqLenQ>
struct FmhaFwdSplitKVShape<64, MaxSeqLenQ> {
  using Type = ck_tile::TileFmhaShape<
      typename FmhaFwdSplitKVBlockTile<64>::type,
      typename FmhaFwdSplitKVBlockTile<64>::gemm0_warps,
      FmhaFwdSplitKVWarpTile,
      typename FmhaFwdSplitKVBlockTile<64, MaxSeqLenQ>::gemm1_warps,
      FmhaFwdSplitKVWarpTile,
      IsVLayoutRowMajor>;
};

template struct FmhaFwdSplitKVShape<64, 32>;
template struct FmhaFwdSplitKVShape<64, 64>;

template <ck_tile::index_t MaxSeqLenQ>
struct FmhaFwdSplitKVShape<96, MaxSeqLenQ> {
  using Type = ck_tile::TileFmhaShape<
      typename FmhaFwdSplitKVBlockTile<96>::type,
      typename FmhaFwdSplitKVBlockTile<96>::gemm0_warps,
      FmhaFwdSplitKVWarpTile,
      typename FmhaFwdSplitKVBlockTile<96, MaxSeqLenQ>::gemm1_warps,
      FmhaFwdSplitKVWarpTile,
      IsVLayoutRowMajor>;
};

template struct FmhaFwdSplitKVShape<96, 32>;
template struct FmhaFwdSplitKVShape<96, 64>;

template <>
struct FmhaFwdSplitKVShape<128, 32> {
  using Type = ck_tile::TileFmhaShape<
      typename FmhaFwdSplitKVBlockTile<128, 32>::type,
      typename FmhaFwdSplitKVBlockTile<128, 32>::gemm0_warps,
      FmhaFwdSplitKVWarpTile,
      typename FmhaFwdSplitKVBlockTile<128, 32>::gemm1_warps,
      FmhaFwdSplitKVWarpTile,
      IsVLayoutRowMajor>;
};

template <>
struct FmhaFwdSplitKVShape<128, 64> {
  using Type = ck_tile::TileFmhaShape<
      typename FmhaFwdSplitKVBlockTile<128, 64>::type,
      typename FmhaFwdSplitKVBlockTile<128, 64>::gemm0_warps,
      FmhaFwdSplitKVWarpTile,
      typename FmhaFwdSplitKVBlockTile<128, 64>::gemm1_warps,
      FmhaFwdSplitKVWarpTile,
      IsVLayoutRowMajor>;
};

template <ck_tile::index_t MaxSeqLenQ>
struct FmhaFwdSplitKVShape<256, MaxSeqLenQ> {
  using Type = ck_tile::TileFmhaShape<
      typename FmhaFwdSplitKVBlockTile<256>::type,
      typename FmhaFwdSplitKVBlockTile<256>::gemm0_warps,
      FmhaFwdSplitKVWarpTile,
      typename FmhaFwdSplitKVBlockTile<256>::gemm1_warps,
      FmhaFwdSplitKVWarpTile,
      IsVLayoutRowMajor>;
};

template struct FmhaFwdSplitKVShape<256, 32>;
template struct FmhaFwdSplitKVShape<256, 64>;

template <ck_tile::index_t MaxK, ck_tile::index_t MaxSeqLenQ>
int fwd_splitkv_get_mtile_size() {
  using FmhaTileShape = typename FmhaFwdSplitKVShape<MaxK, MaxSeqLenQ>::Type;

  return FmhaTileShape::kM0;
};

static int get_mtile_size_for_splitkv(int max_seqlen_q, int max_headdim) {
  int mtile_size_for_splitkv = 64;

  FMHA_FWD_SEQLEN_Q_SWITCH(max_seqlen_q, MaxSeqLenQ, [&] {
    if (max_headdim <= 32) {
      mtile_size_for_splitkv = fwd_splitkv_get_mtile_size<32, MaxSeqLenQ>();
    } else if (max_headdim <= 64) {
      mtile_size_for_splitkv = fwd_splitkv_get_mtile_size<64, MaxSeqLenQ>();
    } else if (max_headdim <= 96) {
      mtile_size_for_splitkv = fwd_splitkv_get_mtile_size<96, MaxSeqLenQ>();
    } else if (max_headdim <= 128) {
      mtile_size_for_splitkv = fwd_splitkv_get_mtile_size<128, MaxSeqLenQ>();
    } else {
      mtile_size_for_splitkv = fwd_splitkv_get_mtile_size<256, MaxSeqLenQ>();
    };
  });

  return mtile_size_for_splitkv;
}
