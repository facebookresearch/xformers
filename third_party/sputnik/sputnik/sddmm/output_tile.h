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

#ifndef THIRD_PARTY_SPUTNIK_SDDMM_OUTPUT_TILE_H_
#define THIRD_PARTY_SPUTNIK_SDDMM_OUTPUT_TILE_H_

/**
 * @file @brief Defines a functor for efficiently storing results to
 * successive tiles in the output sparse matrix.
 */

namespace sputnik {

template <int kBlockItemsX, int kBlockWidth>
struct OutputTile {
  //
  /// Static members.
  //

  // The number of outputs that each thread owns and is responsible
  // for writing back to global memory.
  static constexpr int kThreadItemsX_ = kBlockItemsX / kBlockWidth;

  //
  /// Member variables.
  //

  // The register file fragment with the results to store.
  const float* output_fragment_;

  // Pointer to the buffer storing the output of the kernel.
  float* output_values_;

  /**
   * @brief Set the initial pointer offsets.
   */
  __device__ __forceinline__ OutputTile(int row_offset, int column_offset,
                                        const float* output_fragment,
                                        float* output_values) {
    output_fragment_ = output_fragment;
    output_values_ = output_values + row_offset + column_offset + threadIdx.x;
  }

  /**
   * @brief Flush the fragment to the output and update the pointer for
   * the next iteration.
   */
  __device__ __forceinline__ void Store(int nonzeros) {
#pragma unroll
    for (int x_item_idx = 0; x_item_idx < kThreadItemsX_; ++x_item_idx) {
      if (nonzeros > 0) {
        sputnik::Store(output_fragment_[x_item_idx], output_values_);
      }
      nonzeros -= kBlockWidth;
      output_values_ += kBlockWidth;
    }
  }
};

}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_SDDMM_OUTPUT_TILE_H_
