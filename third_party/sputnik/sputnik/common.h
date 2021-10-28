// Copyright 2020 The Sputnik Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_SPUTNIK_COMMON_H_
#define THIRD_PARTY_SPUTNIK_COMMON_H_

namespace sputnik {

/**
 * @brief Helper to round up to the nearest multiple of 'r'.
 */
constexpr __host__ __device__ __forceinline__ int RoundUpTo(int x, int r) {
  return (x + r - 1) / r * r;
}

/**
 * @brief Dividy x by y and round up.
 */
constexpr __host__ __device__ __forceinline__ int DivUp(int x, int y) {
  return (x + y - 1) / y;
}

/**
 * @brief Compute log base 2 statically. Only works when x
 * is a power of 2 and positive.
 *
 * TODO(tgale): GCC doesn't like this function being constexpr. Ensure
 * that this is evaluated statically.
 */
__host__ __device__ __forceinline__ int Log2(int x) {
  if (x >>= 1) return Log2(x) + 1;
  return 0;
}

/**
 * @brief Find the minimum statically.
 */
constexpr __host__ __device__ __forceinline__ int Min(int a, int b) {
  return a < b ? a : b;
}

/**
 * @brief Find the maximum statically.
 */
constexpr __host__ __device__ __forceinline__ int Max(int a, int b) {
  return a > b ? a : b;
}

}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_COMMON_H_
