/*
 * Copyright (c) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ck_tile/core.hpp>
#include <stdexcept>

#ifndef FMHA_LIMIT_MAX_HEADDIM_TO_256
#define FMHA_LIMIT_MAX_HEADDIM_TO_256 0
#endif

#if FMHA_LIMIT_MAX_HEADDIM_TO_256

#define FMHA_FWD_HEADDIM_SWITCH(HEAD_DIM1, HEAD_DIM2, CONST_NAME, ...) \
  [&] {                                                                \
    if (HEAD_DIM1 <= 32 && HEAD_DIM2 <= 32) {                          \
      constexpr ck_tile::index_t CONST_NAME = 32;                      \
      __VA_ARGS__();                                                   \
    } else if (HEAD_DIM1 <= 64 && HEAD_DIM2 <= 64) {                   \
      constexpr ck_tile::index_t CONST_NAME = 64;                      \
      __VA_ARGS__();                                                   \
    } else if (HEAD_DIM1 <= 96 && HEAD_DIM2 <= 96) {                   \
      constexpr ck_tile::index_t CONST_NAME = 96;                      \
      __VA_ARGS__();                                                   \
    } else if (HEAD_DIM1 <= 128 && HEAD_DIM2 <= 128) {                 \
      constexpr ck_tile::index_t CONST_NAME = 128;                     \
      __VA_ARGS__();                                                   \
    } else if (HEAD_DIM1 <= 256 && HEAD_DIM2 <= 256) {                 \
      constexpr ck_tile::index_t CONST_NAME = 256;                     \
      __VA_ARGS__();                                                   \
    } else {                                                           \
      throw std::runtime_error("Head-dim sizes not supported!");       \
    }                                                                  \
  }()

#else

#define FMHA_FWD_HEADDIM_SWITCH(HEAD_DIM1, HEAD_DIM2, CONST_NAME, ...) \
  [&] {                                                                \
    if (HEAD_DIM1 <= 32 && HEAD_DIM2 <= 32) {                          \
      constexpr ck_tile::index_t CONST_NAME = 32;                      \
      __VA_ARGS__();                                                   \
    } else if (HEAD_DIM1 <= 64 && HEAD_DIM2 <= 64) {                   \
      constexpr ck_tile::index_t CONST_NAME = 64;                      \
      __VA_ARGS__();                                                   \
    } else if (HEAD_DIM1 <= 96 && HEAD_DIM2 <= 96) {                   \
      constexpr ck_tile::index_t CONST_NAME = 96;                      \
      __VA_ARGS__();                                                   \
    } else if (HEAD_DIM1 <= 128 && HEAD_DIM2 <= 128) {                 \
      constexpr ck_tile::index_t CONST_NAME = 128;                     \
      __VA_ARGS__();                                                   \
    } else if (HEAD_DIM1 <= 256 && HEAD_DIM2 <= 256) {                 \
      constexpr ck_tile::index_t CONST_NAME = 256;                     \
      __VA_ARGS__();                                                   \
    } else if (HEAD_DIM1 <= 512 && HEAD_DIM2 <= 512) {                 \
      constexpr ck_tile::index_t CONST_NAME = 512;                     \
      __VA_ARGS__();                                                   \
    } else {                                                           \
      throw std::runtime_error("Head-dim sizes not supported!");       \
    }                                                                  \
  }()

#endif

#define FMHA_BWD_HEADDIM_SWITCH(HEAD_DIM1, HEAD_DIM2, CONST_NAME, ...) \
  [&] {                                                                \
    if (HEAD_DIM1 <= 32 && HEAD_DIM2 <= 32) {                          \
      constexpr ck_tile::index_t CONST_NAME = 32;                      \
      __VA_ARGS__();                                                   \
    } else if (HEAD_DIM1 <= 64 && HEAD_DIM2 <= 64) {                   \
      constexpr ck_tile::index_t CONST_NAME = 64;                      \
      __VA_ARGS__();                                                   \
    } else if (HEAD_DIM1 <= 96 && HEAD_DIM2 <= 96) {                   \
      constexpr ck_tile::index_t CONST_NAME = 96;                      \
      __VA_ARGS__();                                                   \
    } else if (HEAD_DIM1 <= 128 && HEAD_DIM2 <= 128) {                 \
      constexpr ck_tile::index_t CONST_NAME = 128;                     \
      __VA_ARGS__();                                                   \
    } else if (HEAD_DIM1 <= 256 && HEAD_DIM2 <= 256) {                 \
      constexpr ck_tile::index_t CONST_NAME = 256;                     \
      __VA_ARGS__();                                                   \
    } else {                                                           \
      throw std::runtime_error("Head-dim sizes not supported!");       \
    }                                                                  \
  }()
