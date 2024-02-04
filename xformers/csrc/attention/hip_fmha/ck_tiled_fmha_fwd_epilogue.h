/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "ck/tile_program/tile/store_tile.hpp"
#include "ck/tile_program/tile/tile_elementwise.hpp"
#include "ck/utility/common_header.hpp"

template <typename OaccDataType_, typename ODataType_>
struct FmhaFwdEpilogueProblem {
  using OaccDataType = ck::remove_cvref_t<OaccDataType_>;
  using ODataType = ck::remove_cvref_t<ODataType_>;
};

template <typename Problem_, typename Policy_ = void>
struct FmhaFwdEpilogue {
  using Problem = ck::remove_cvref_t<Problem_>;
  using OaccDataType = ck::remove_cvref_t<typename Problem::OaccDataType>;
  using ODataType = ck::remove_cvref_t<typename Problem::ODataType>;

  __host__ __device__ static constexpr ck::index_t GetSmemSize() {
    return 0;
  }

  template <typename ODramWindowTmp, typename OAccTile>
  __device__ auto operator()(
      ODramWindowTmp& o_dram_window_tmp,
      const OAccTile& o_acc_tile) {
    using namespace ck;
    using namespace ck::tile_program;

    const auto o =
        tile_elementwise_in(type_convert<ODataType, OaccDataType>, o_acc_tile);
    store_tile(o_dram_window_tmp, o);
  }
};
