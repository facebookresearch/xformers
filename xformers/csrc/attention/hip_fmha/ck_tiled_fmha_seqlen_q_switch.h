/*
 * Copyright (c) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ck_tile/core.hpp>
#include <stdexcept>

#define FMHA_FWD_SEQLEN_Q_SWITCH(SEQLEN_Q, CONST_NAME, ...) \
  [&] {                                                     \
    if (SEQLEN_Q <= 32) {                                   \
      constexpr ck_tile::index_t CONST_NAME = 32;           \
      __VA_ARGS__();                                        \
    } else {                                                \
      constexpr ck_tile::index_t CONST_NAME = 64;           \
      __VA_ARGS__();                                        \
    }                                                       \
  }()
