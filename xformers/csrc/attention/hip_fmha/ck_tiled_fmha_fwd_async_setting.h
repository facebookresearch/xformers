/*
 * Copyright (c) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ck_tile/core.hpp>
#include <ck_tile/ops/fmha.hpp>
#include "ck_fmha_util.h"
#include "ck_tiled_fmha_fwd_type_config.h"

template <ck_tile::index_t MaxK, ck_tile::index_t MTile = 0>
struct FmhaFwdAsyncBlockTile;

// Tile-sizes: M N0 K0 N1 K1 MaxK (MaxK % K0 == 0, MaxK % N1 == 0, N0 % K1 == 0)
//
template <ck_tile::index_t MTile>
struct FmhaFwdAsyncBlockTile<32, MTile> {
  using type = ck_tile::sequence<64, 64, 16, 32, 32, 32>;
  using gemm0_warps = ck_tile::sequence<2, 1, 1>;
  using gemm1_warps = ck_tile::sequence<2, 1, 1>;
};

template struct FmhaFwdAsyncBlockTile<32>;

template <ck_tile::index_t MTile>
struct FmhaFwdAsyncBlockTile<64, MTile> {
  using type = ck_tile::sequence<128, 64, 16, 64, 32, 64>;
  using gemm0_warps = ck_tile::sequence<4, 1, 1>;
  using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template struct FmhaFwdAsyncBlockTile<64>;

template <ck_tile::index_t MTile>
struct FmhaFwdAsyncBlockTile<96, MTile> {
  using type = ck_tile::sequence<128, 64, 32, 128, 32, 96>;
  using gemm0_warps = ck_tile::sequence<4, 1, 1>;
  using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template struct FmhaFwdAsyncBlockTile<96>;

template <>
struct FmhaFwdAsyncBlockTile<128, 64> {
  using type = ck_tile::sequence<64, 128, 32, 128, 32, 128>;
  using gemm0_warps = ck_tile::sequence<4, 1, 1>;
  using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <>
struct FmhaFwdAsyncBlockTile<128, 128> {
  using type = ck_tile::sequence<128, 64, 32, 128, 32, 128>;
  using gemm0_warps = ck_tile::sequence<4, 1, 1>;
  using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <ck_tile::index_t MTile>
struct FmhaFwdAsyncBlockTile<256, MTile> {
  using type = ck_tile::sequence<64, 32, 32, 256, 16, 256>;
  using gemm0_warps = ck_tile::sequence<4, 1, 1>;
  using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template struct FmhaFwdAsyncBlockTile<256>;

using FmhaFwdAsyncWarpTile1 = ck_tile::sequence<32, 32, 16>;
using FmhaFwdAsyncWarpTile2 = ck_tile::sequence<16, 16, 16>;
using FmhaFwdAsyncWarpTile3 = ck_tile::sequence<16, 16, 32>;

template <ck_tile::index_t MaxK, ck_tile::index_t MTile>
struct FmhaFwdAsyncShape;

template <ck_tile::index_t MTile>
struct FmhaFwdAsyncShape<32, MTile> {
  using Type = ck_tile::TileFmhaShape<
      typename FmhaFwdAsyncBlockTile<32>::type,
      typename FmhaFwdAsyncBlockTile<32>::gemm0_warps,
      FmhaFwdAsyncWarpTile1,
      typename FmhaFwdAsyncBlockTile<32>::gemm1_warps,
      FmhaFwdAsyncWarpTile1,
      IsVLayoutRowMajor>;
};

template struct FmhaFwdAsyncShape<32, 64>;
template struct FmhaFwdAsyncShape<32, 128>;

template <ck_tile::index_t MTile>
struct FmhaFwdAsyncShape<64, MTile> {
  using Type = ck_tile::TileFmhaShape<
      typename FmhaFwdAsyncBlockTile<64>::type,
      typename FmhaFwdAsyncBlockTile<64>::gemm0_warps,
      FmhaFwdAsyncWarpTile1,
      typename FmhaFwdAsyncBlockTile<64>::gemm1_warps,
      FmhaFwdAsyncWarpTile1,
      IsVLayoutRowMajor>;
};

template struct FmhaFwdAsyncShape<64, 64>;
template struct FmhaFwdAsyncShape<64, 128>;

template <ck_tile::index_t MTile>
struct FmhaFwdAsyncShape<96, MTile> {
  using Type = ck_tile::TileFmhaShape<
      typename FmhaFwdAsyncBlockTile<96>::type,
      typename FmhaFwdAsyncBlockTile<96>::gemm0_warps,
      FmhaFwdAsyncWarpTile3,
      typename FmhaFwdAsyncBlockTile<96>::gemm1_warps,
      FmhaFwdAsyncWarpTile2,
      IsVLayoutRowMajor>;
};

template struct FmhaFwdAsyncShape<96, 64>;
template struct FmhaFwdAsyncShape<96, 128>;

template <>
struct FmhaFwdAsyncShape<128, 64> {
  using Type = ck_tile::TileFmhaShape<
      typename FmhaFwdAsyncBlockTile<128, 64>::type,
      typename FmhaFwdAsyncBlockTile<128, 64>::gemm0_warps,
      FmhaFwdAsyncWarpTile3,
      typename FmhaFwdAsyncBlockTile<128, 64>::gemm1_warps,
      FmhaFwdAsyncWarpTile2,
      IsVLayoutRowMajor>;
};

template <>
struct FmhaFwdAsyncShape<128, 128> {
  using Type = ck_tile::TileFmhaShape<
      typename FmhaFwdAsyncBlockTile<128, 128>::type,
      typename FmhaFwdAsyncBlockTile<128, 128>::gemm0_warps,
      FmhaFwdAsyncWarpTile3,
      typename FmhaFwdAsyncBlockTile<128, 128>::gemm1_warps,
      FmhaFwdAsyncWarpTile2,
      IsVLayoutRowMajor>;
};

template <ck_tile::index_t MTile>
struct FmhaFwdAsyncShape<256, MTile> {
  using Type = ck_tile::TileFmhaShape<
      typename FmhaFwdAsyncBlockTile<256>::type,
      typename FmhaFwdAsyncBlockTile<256>::gemm0_warps,
      FmhaFwdAsyncWarpTile2,
      typename FmhaFwdAsyncBlockTile<256>::gemm1_warps,
      FmhaFwdAsyncWarpTile2,
      IsVLayoutRowMajor>;
};

template struct FmhaFwdAsyncShape<256, 64>;
template struct FmhaFwdAsyncShape<256, 128>;

static int get_fmha_fwd_async_mtile(
    int num_batches,
    int num_heads,
    int max_seqlen_q) {
  int num_SMs = get_number_of_cu();
  auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };

  int batch_nhead_mblocks =
      num_batches * num_heads * ceildiv(max_seqlen_q, 128);

  if (batch_nhead_mblocks >= 0.8 * num_SMs)
    return 128;

  return 64;
};

static int get_fmha_fwd_async_least_mtile() {
  return 64;
};
