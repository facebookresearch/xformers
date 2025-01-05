/*
 * Copyright (c) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ck_tile/core.hpp>
#include <stdexcept>

#define FMHA_FWD_NUM_KV_SPLITS_SWITCH(NUM_SPLITS, CONST_NAME, ...) \
  [&] {                                                            \
    if (NUM_SPLITS <= 8) {                                         \
      constexpr ck_tile::index_t CONST_NAME = 3;                   \
      __VA_ARGS__();                                               \
    } else if (NUM_SPLITS <= 16) {                                 \
      constexpr ck_tile::index_t CONST_NAME = 4;                   \
      __VA_ARGS__();                                               \
    } else {                                                       \
      throw std::runtime_error("num-splits not supported!");       \
    }                                                              \
  }()
