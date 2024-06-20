/*
 * Copyright (c) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#define BOOL_SWITCH(COND1, CONST_NAME1, ...) \
  [&] {                                      \
    if (COND1) {                             \
      constexpr bool CONST_NAME1 = true;     \
      __VA_ARGS__();                         \
    } else {                                 \
      constexpr bool CONST_NAME1 = false;    \
      __VA_ARGS__();                         \
    }                                        \
  }()

#define BOOL_SWITCH_2(COND1, CONST_NAME1, COND2, CONST_NAME2, ...) \
  [&] {                                                            \
    if (COND1) {                                                   \
      constexpr bool CONST_NAME1 = true;                           \
      BOOL_SWITCH(COND2, CONST_NAME2, ##__VA_ARGS__);              \
    } else {                                                       \
      constexpr bool CONST_NAME1 = false;                          \
      BOOL_SWITCH(COND2, CONST_NAME2, ##__VA_ARGS__);              \
    }                                                              \
  }()

#define BOOL_SWITCH_3(                                                      \
    COND1, CONST_NAME1, COND2, CONST_NAME2, COND3, CONST_NAME3, ...)        \
  [&] {                                                                     \
    if (COND1) {                                                            \
      constexpr bool CONST_NAME1 = true;                                    \
      BOOL_SWITCH_2(COND2, CONST_NAME2, COND3, CONST_NAME3, ##__VA_ARGS__); \
    } else {                                                                \
      constexpr bool CONST_NAME1 = false;                                   \
      BOOL_SWITCH_2(COND2, CONST_NAME2, COND3, CONST_NAME3, ##__VA_ARGS__); \
    }                                                                       \
  }()

#define BOOL_SWITCH_4(                    \
    COND1,                                \
    CONST_NAME1,                          \
    COND2,                                \
    CONST_NAME2,                          \
    COND3,                                \
    CONST_NAME3,                          \
    COND4,                                \
    CONST_NAME4,                          \
    ...)                                  \
  [&] {                                   \
    if (COND1) {                          \
      constexpr bool CONST_NAME1 = true;  \
      BOOL_SWITCH_3(                      \
          COND2,                          \
          CONST_NAME2,                    \
          COND3,                          \
          CONST_NAME3,                    \
          COND4,                          \
          CONST_NAME4,                    \
          ##__VA_ARGS__);                 \
    } else {                              \
      constexpr bool CONST_NAME1 = false; \
      BOOL_SWITCH_3(                      \
          COND2,                          \
          CONST_NAME2,                    \
          COND3,                          \
          CONST_NAME3,                    \
          COND4,                          \
          CONST_NAME4,                    \
          ##__VA_ARGS__);                 \
    }                                     \
  }()
