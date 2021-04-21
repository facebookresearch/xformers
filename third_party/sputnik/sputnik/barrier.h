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

#ifndef THIRD_PARTY_SPUTNIK_BARRIER_H_
#define THIRD_PARTY_SPUTNIK_BARRIER_H_

/**
 * @file @brief Defines a class for managing synchronization between
 * dependent threads.
 */

#include <cstdint>

namespace sputnik {

/**
 * @brief Compute exponents at compile-time.
 */
__device__ constexpr uint32_t StaticPow(uint32_t base, uint32_t exponent) {
  return exponent == 0 ? 1 : base * StaticPow(base, exponent - 1);
}

/**
 * @brief Manages synchronization between dependent threads.
 *
 * Depending on the configuration, our spmm kernels need to synchronize
 * between multiple warps, within a single warp, or between a set of threads
 * that form part of a warp (a "subwarp"). This class abstracts these details
 * from the main kernel implementation.
 */
template <int kBlockItemsY, int kBlockWidth>
struct Barrier {
  //
  /// Static members.
  //

  // The number of threads in a thread block.
  static constexpr int kThreadsPerBlock = kBlockItemsY * kBlockWidth;

  static_assert(kThreadsPerBlock % 32 == 0,
                "The thread-block size must be divisible by the warp size.");
  static_assert(kThreadsPerBlock > 0, "The thread-block size must be nonzero.");

  // The number of threads that collaborates on a single 1-dimensional
  // tile of the output matrix.
  static constexpr int kThreadsPerOutputTile = kBlockWidth;

  static_assert((kThreadsPerOutputTile % 2) == 0 ||
                    (kThreadsPerOutputTile == 1),
                "The number of threads collaborating on a tile must be "
                "a multiple of two or all threads must be independent.");

  // NOTE: We currently don't support packing multiple independent warps
  // into a single thread-block when we're using a subwarp tiling (our
  // mask calculation code could overflow).
  static_assert((kThreadsPerBlock == 32) || (kThreadsPerOutputTile >= 32) ||
                    (kThreadsPerOutputTile == 1),
                "Independent warps must be in separate thread-blocks "
                "when using a subwarp tiling.");

  //
  /// Member variables.
  //

#if __CUDA_ARCH__ >= 700
  // The thread mask for synchronizing at the subwarp granularity. Not
  // needed on pre-Volta architectures.
  uint32_t thread_mask = 0xffffffff;
#endif

  /**
   * @brief Construct the barrier object.
   *
   * @param thread_idx_y The index of this thread in the y-dimension
   * of the 2-dimensional region being computed by its thread-block.
   */
  __device__ __forceinline__ Barrier(int thread_idx_y) {
#if __CUDA_ARCH__ >= 700
    // For subwarp tilings with Volta and beyond, we need to set the
    // appropriate thread mask for synchronization.
    if ((kThreadsPerOutputTile < 32) && (kThreadsPerOutputTile > 1)) {
      // The basic pattern of the thread mask for a given number of threads
      // per output tile. We can compute this at compile-time and then just
      // shift the bits appropriately depending of this subwarps offset within
      // the thread-block.
      constexpr uint32_t kBaseSubwarpMask =
          StaticPow(2, kThreadsPerOutputTile) - 1;
      thread_mask = kBaseSubwarpMask << (thread_idx_y * kThreadsPerOutputTile);
    }
#endif
  }

  /**
   * @brief Synchronize at the appropriate granularity.
   *
   * NOTE: These branches can be evalutate statically and should not
   * manifest at the PTX level or lower.
   */
  __device__ __forceinline__ void Sync() {
#if __CUDA_ARCH__ >= 700
    // For Volta and on, synchronize at the subwarp level to ensure
    // correctness with independent thread scheduling.
    if (kThreadsPerOutputTile > 32) {
      __syncthreads();
    } else if (kThreadsPerOutputTile > 1) {
      __syncwarp(thread_mask);
    }
#else
    // For all architectures prior to Volta, only synchronize if multiple
    // warps are collaborating on a single 1d tile of the output matrix.
    if (kThreadsPerOutputTile > 32) {
      __syncthreads();
    }
#endif
  }

  /**
   * @brief Returns the thread mask being used for synchronization.
   */
  __device__ __forceinline__ uint32_t ThreadMask() const {
#if __CUDA_ARCH__ >= 700
    return thread_mask;
#else
    return 0xffffffff;
#endif
  }
};

}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_BARRIER_H_
