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

#ifndef THIRD_PARTY_SPUTNIK_SDDMM_DENSE_TO_SHARED_H_
#define THIRD_PARTY_SPUTNIK_SDDMM_DENSE_TO_SHARED_H_

namespace sputnik {

template <typename LoadType, int kBlockItemsK, int kBlockWidth>
struct DenseToShared {
  static_assert(kBlockItemsK >= kBlockWidth,
                "Sparse tile K-items must be >= thread block width.");
  static_assert(kBlockItemsK % kBlockWidth == 0,
                "Sparse tile K-items must be divisible by block width.");
  static_assert((sizeof(LoadType) / sizeof(float)) <=
                    (kBlockItemsK / kBlockWidth),
                "The number of values per load must be <= values per thread.");

  //
  /// Static members.
  //

  // The number of values that will be loaded per-thread, per-load.
  static constexpr int kValuesPerLoad_ = sizeof(LoadType) / sizeof(float);

  // The number of data items in the k-dimension that each thread owns.
  static constexpr int kThreadItemsK_ =
      kBlockItemsK / kBlockWidth / kValuesPerLoad_;

  //
  /// Member variables.
  //

  // Pointer to the dense matrix in global memory.
  const LoadType *matrix_;

  // Register file framgnet for the loaded values.
  LoadType *matrix_fragment_;

  /**
   * @brief Set the initial pointer offsets.
   */
  __device__ __forceinline__ DenseToShared(int k, int m_index,
                                           const float *matrix,
                                           float *matrix_fragment) {
    matrix_ =
        reinterpret_cast<const LoadType *>(matrix + m_index * k) + threadIdx.x;
    matrix_fragment_ = reinterpret_cast<LoadType *>(matrix_fragment);
  }

  /**
   * @brief Strip-mine a 1-dimensional tile from the matrix.
   */
  __device__ __forceinline__ void Load() {
#pragma unroll
    for (int k_item_idx = 0; k_item_idx < kThreadItemsK_; ++k_item_idx) {
      // Load the values into smem.
      Store(sputnik::Load(matrix_), matrix_fragment_ + k_item_idx);

      // Increment our pointers for the next iteration.
      matrix_ += kBlockWidth;
    }
  }

  /**
   * @brief Loads any residual elements from the matrix.
   */
  __device__ __forceinline__ void Residue(int residue) {
    const LoadType *matrix = matrix_;
#pragma unroll
    for (int k_item_idx = 0; k_item_idx < kThreadItemsK_; ++k_item_idx) {
      if (residue > 0) {
        // Load the values into smem.
        Store(sputnik::Load(matrix), matrix_fragment_ + k_item_idx);
      }
      // Increment our pointer & value index for the next iteration.
      matrix += kBlockWidth;
      residue -= kBlockWidth * kValuesPerLoad_;
    }
  }
};

}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_SDDMM_DENSE_TO_SHARED_H_
