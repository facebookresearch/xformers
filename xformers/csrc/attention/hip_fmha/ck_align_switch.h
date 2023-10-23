#pragma once

#include <stdexcept>

// assume the maximum alignment is 8 elements
#define ALIGN_SWITCH_1(CONST_ALIGN_MAX1, CONST_ALIGN_NAME1, LENGTH1, ...)   \
  [&] {                                                                     \
    if constexpr (CONST_ALIGN_MAX1 > 0) {                                   \
      if (LENGTH1 % CONST_ALIGN_MAX1 == 0) {                                \
        constexpr ck::index_t CONST_ALIGN_NAME1 = CONST_ALIGN_MAX1;         \
        __VA_ARGS__();                                                      \
      } else {                                                              \
        if constexpr (CONST_ALIGN_MAX1 / 2 > 0) {                           \
          if (LENGTH1 % (CONST_ALIGN_MAX1 / 2) == 0) {                      \
            constexpr ck::index_t CONST_ALIGN_NAME1 = CONST_ALIGN_MAX1 / 2; \
            __VA_ARGS__();                                                  \
          } else {                                                          \
            if constexpr (CONST_ALIGN_MAX1 / 4 > 0) {                       \
              if (LENGTH1 % (CONST_ALIGN_MAX1 / 4) == 0) {                  \
                constexpr ck::index_t CONST_ALIGN_NAME1 =                   \
                    CONST_ALIGN_MAX1 / 4;                                   \
                __VA_ARGS__();                                              \
              } else {                                                      \
                constexpr ck::index_t CONST_ALIGN_NAME1 = 1;                \
                __VA_ARGS__();                                              \
              };                                                            \
            }                                                               \
          };                                                                \
        }                                                                   \
      };                                                                    \
    }                                                                       \
  }()

// assume the maximum alignment is 8 elements
#define ALIGN_SWITCH_2(                                                       \
    CONST_ALIGN_MAX1,                                                         \
    CONST_ALIGN_NAME1,                                                        \
    LENGTH1,                                                                  \
    CONST_ALIGN_MAX2,                                                         \
    CONST_ALIGN_NAME2,                                                        \
    LENGTH2,                                                                  \
    ...)                                                                      \
  [&] {                                                                       \
    if constexpr (CONST_ALIGN_MAX1 > 0) {                                     \
      if (LENGTH1 % CONST_ALIGN_MAX1 == 0) {                                  \
        constexpr ck::index_t CONST_ALIGN_NAME1 = CONST_ALIGN_MAX1;           \
        ALIGN_SWITCH_1(                                                       \
            CONST_ALIGN_MAX2, CONST_ALIGN_NAME2, LENGTH2, ##__VA_ARGS__);     \
      } else {                                                                \
        if constexpr (CONST_ALIGN_MAX1 / 2 > 0) {                             \
          if (LENGTH1 % (CONST_ALIGN_MAX1 / 2) == 0) {                        \
            constexpr ck::index_t CONST_ALIGN_NAME1 = CONST_ALIGN_MAX1 / 2;   \
            ALIGN_SWITCH_1(                                                   \
                CONST_ALIGN_MAX2, CONST_ALIGN_NAME2, LENGTH2, ##__VA_ARGS__); \
          } else {                                                            \
            if constexpr (CONST_ALIGN_MAX1 / 4 > 0) {                         \
              if (LENGTH1 % (CONST_ALIGN_MAX1 / 4) == 0) {                    \
                constexpr ck::index_t CONST_ALIGN_NAME1 =                     \
                    CONST_ALIGN_MAX1 / 4;                                     \
                ALIGN_SWITCH_1(                                               \
                    CONST_ALIGN_MAX2,                                         \
                    CONST_ALIGN_NAME2,                                        \
                    LENGTH2,                                                  \
                    ##__VA_ARGS__);                                           \
              } else {                                                        \
                constexpr ck::index_t CONST_ALIGN_NAME1 = 1;                  \
                ALIGN_SWITCH_1(                                               \
                    CONST_ALIGN_MAX2,                                         \
                    CONST_ALIGN_NAME2,                                        \
                    LENGTH2,                                                  \
                    ##__VA_ARGS__);                                           \
              };                                                              \
            }                                                                 \
          };                                                                  \
        }                                                                     \
      };                                                                      \
    }                                                                         \
  }()

// assume the maximum alignment is 8 elements
#define ALIGN_SWITCH_3(                                                     \
    CONST_ALIGN_MAX1,                                                       \
    CONST_ALIGN_NAME1,                                                      \
    LENGTH1,                                                                \
    CONST_ALIGN_MAX2,                                                       \
    CONST_ALIGN_NAME2,                                                      \
    LENGTH2,                                                                \
    CONST_ALIGN_MAX3,                                                       \
    CONST_ALIGN_NAME3,                                                      \
    LENGTH3,                                                                \
    ...)                                                                    \
  [&] {                                                                     \
    if constexpr (CONST_ALIGN_MAX1 > 0) {                                   \
      if (LENGTH1 % CONST_ALIGN_MAX1 == 0) {                                \
        constexpr ck::index_t CONST_ALIGN_NAME1 = CONST_ALIGN_MAX1;         \
        ALIGN_SWITCH_2(                                                     \
            CONST_ALIGN_MAX2,                                               \
            CONST_ALIGN_NAME2,                                              \
            LENGTH2,                                                        \
            CONST_ALIGN_MAX3,                                               \
            CONST_ALIGN_NAME3,                                              \
            LENGTH3,                                                        \
            ##__VA_ARGS__);                                                 \
      } else {                                                              \
        if constexpr (CONST_ALIGN_MAX1 / 2 > 0) {                           \
          if (LENGTH1 % (CONST_ALIGN_MAX1 / 2) == 0) {                      \
            constexpr ck::index_t CONST_ALIGN_NAME1 = CONST_ALIGN_MAX1 / 2; \
            ALIGN_SWITCH_2(                                                 \
                CONST_ALIGN_MAX2,                                           \
                CONST_ALIGN_NAME2,                                          \
                LENGTH2,                                                    \
                CONST_ALIGN_MAX3,                                           \
                CONST_ALIGN_NAME3,                                          \
                LENGTH3,                                                    \
                ##__VA_ARGS__);                                             \
          } else {                                                          \
            if constexpr (CONST_ALIGN_MAX1 / 4 > 0) {                       \
              if (LENGTH1 % (CONST_ALIGN_MAX1 / 4) == 0) {                  \
                constexpr ck::index_t CONST_ALIGN_NAME1 =                   \
                    CONST_ALIGN_MAX1 / 4;                                   \
                ALIGN_SWITCH_2(                                             \
                    CONST_ALIGN_MAX2,                                       \
                    CONST_ALIGN_NAME2,                                      \
                    LENGTH2,                                                \
                    CONST_ALIGN_MAX3,                                       \
                    CONST_ALIGN_NAME3,                                      \
                    LENGTH3,                                                \
                    ##__VA_ARGS__);                                         \
              } else {                                                      \
                constexpr ck::index_t CONST_ALIGN_NAME1 = 1;                \
                ALIGN_SWITCH_2(                                             \
                    CONST_ALIGN_MAX2,                                       \
                    CONST_ALIGN_NAME2,                                      \
                    LENGTH2,                                                \
                    CONST_ALIGN_MAX3,                                       \
                    CONST_ALIGN_NAME3,                                      \
                    LENGTH3,                                                \
                    ##__VA_ARGS__);                                         \
              };                                                            \
            }                                                               \
          };                                                                \
        }                                                                   \
      };                                                                    \
    }                                                                       \
  }()
