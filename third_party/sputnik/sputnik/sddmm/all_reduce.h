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

#ifndef THIRD_PARTY_SPUTNIK_SDDMM_ALL_REDUCE_H_
#define THIRD_PARTY_SPUTNIK_SDDMM_ALL_REDUCE_H_

/**
 * @file @brief Defines a functor for performing a batch of all reduces
 * to calculate the final results of the SDDMM.
 */

#include "sputnik/common.h"

namespace sputnik {

template <typename LoadType, int kBlockItemsX, int kBlockWidth>
struct AllReduce {
  //
  /// Static members.
  //

  // The number of values that will be loaded per-thread, per-load.
  static constexpr int kValuesPerLoad = sizeof(LoadType) / sizeof(float);

  // The number of outputs each thread is responsible for.
  static constexpr int kThreadItemsX = kBlockItemsX / kBlockWidth;

  //
  /// Member variables.
  //

  // Thread mask used for warp shuffle operations.
  const uint32_t kShflMask;

  // Register file fragment storing the thread local partial results.
  float* inputs;

  // Registe file fragment for storing each threads results.
  float* outputs;

  __device__ __forceinline__ AllReduce(const uint32_t thread_mask,
                                       float* inputs_, float* outputs_)
      : kShflMask(thread_mask), inputs(inputs_), outputs(outputs_) {}
  __device__ __forceinline__ void Swap(int i, int j, float* x) {
    float t = x[i];
    x[i] = x[j];
    x[j] = t;
  }

  __device__ __forceinline__ void ReduceStep(int lane, int i, int j) {
    const int kStep = Log2(lane);
    if ((threadIdx.x >> kStep) & 1) Swap(i, j, inputs);
    inputs[i] += __shfl_xor_sync(kShflMask, inputs[j], lane, kBlockWidth);
  }

  __device__ __forceinline__ void Reduce() {
#pragma unroll
    for (int base_idx = 0; base_idx < kThreadItemsX; ++base_idx) {
#pragma unroll
      for (int k_item_idx = 1; k_item_idx < kBlockWidth; k_item_idx *= 2) {
        const int kBoundX = kBlockWidth / (k_item_idx * 2);
#pragma unroll
        for (int x_item_idx = 0; x_item_idx < kBoundX; ++x_item_idx) {
          const int idx_a = x_item_idx * 2 * kValuesPerLoad * k_item_idx;
          const int idx_b = (x_item_idx * 2 + 1) * kValuesPerLoad * k_item_idx;
          ReduceStep(k_item_idx, base_idx + idx_a, base_idx + idx_b);
        }
      }
    }

    // Move the last four values to the first four of the output. This
    // should get cleaned up during register allocation.
#pragma unroll
    for (int out_idx = 0; out_idx < kThreadItemsX; ++out_idx) {
      outputs[out_idx] = inputs[out_idx];
    }
  }
};

}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_SDDMM_ALL_REDUCE_H_
