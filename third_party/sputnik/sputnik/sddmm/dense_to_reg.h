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

#ifndef THIRD_PARTY_SPUTNIK_SDDMM_DENSE_TO_REG_H_
#define THIRD_PARTY_SPUTNIK_SDDMM_DENSE_TO_REG_H_

#include "sputnik/load_store.h"
#include "sputnik/tiling_utils.h"
#include "sputnik/vector_utils.h"

namespace sputnik {

template <typename LoadType, int kBlockItemsK, int kBlockItemsX,
          int kBlockWidth>
struct DenseToReg {
  static_assert(kBlockItemsK * kBlockItemsX >= kBlockWidth,
                "Dense tile size must be >= thread block width.");
  static_assert(kBlockItemsK * kBlockItemsX % kBlockWidth == 0,
                "Dense tile size must be divisible by thread block width.");
  static_assert((sizeof(LoadType) / sizeof(float)) <=
                    (kBlockItemsK / kBlockWidth),
                "The number of values per load must be <= values per thread.");

  //
  /// Static members.
  //

  // The number of values that will be loaded per-thread, per-load.
  static constexpr int kValuesPerLoad_ = sizeof(LoadType) / sizeof(float);

  // The number of data items in the x-dimension that each thread will
  // help load data for.
  static constexpr int kThreadItemsX_ = kBlockItemsX;

  // The number of data items in the x-dimension that each thread will
  // load when collaboratively loading column indices.
  static constexpr int kColabItemsX_ =
      kBlockItemsX / kBlockWidth / kValuesPerLoad_;

  // The number of data items in the k-dimension that each thread will
  // help load data for.
  static constexpr int kThreadItemsK_ =
      kBlockItemsK / kBlockWidth / kValuesPerLoad_;

  // Complementary index type to our load type.
  typedef typename Value2Index<LoadType>::Index IndexType;

  //
  /// Member variables.
  //

  // The length of the rows that we'll be loading.
  const int kDimK_;

  // Pointer to the output sparse matrix column indices in global memory.
  //
  // NOTE: The matrix is stored in row-major order, but we'd like to compute
  // the matrix product with its transpose. These indices are column indices
  // for the *transposed* matrix, and row indices for the standard matrix.
  const int *column_indices_;

  // Base pointer to the dense matrix in global memory.
  const LoadType *matrix_base_;

  // Base pointer for the smem block used to store column indices.
  int *column_idxs_tile_base_;

  // Register fragment for the dense matrix values.
  LoadType *matrix_fragment_;

  /**
   * @brief Set the initial pointer offsets.
   */
  __device__ __forceinline__ DenseToReg(
      int k, int row_offset, int column_offset, const int *column_indices,
      const float *matrix, int *column_indices_tile, float *matrix_fragment)
      : kDimK_(k * sizeof(float)) {
    column_indices_ = column_indices + row_offset + column_offset + threadIdx.x;
    column_idxs_tile_base_ = column_indices_tile;

    matrix_base_ = reinterpret_cast<const LoadType *>(matrix) + threadIdx.x;
    matrix_fragment_ = reinterpret_cast<LoadType *>(matrix_fragment);
  }

  /**
   * @brief Load a new set of column indices and calculates our offset matrix
   * pointers.
   */
  __device__ __forceinline__ void LoadColumnIndices(int nonzeros) {
    IndexType *column_idxs_tile =
        reinterpret_cast<IndexType *>(column_idxs_tile_base_) + threadIdx.x;
#pragma unroll
    for (int x_item_idx = 0; x_item_idx < kColabItemsX_; ++x_item_idx) {
      // Staging registers for the indices s.t. we can issue vector loads to
      // shared memory. Zero s.t. it's safe to operate on out-of-bounds idxs.
      int index_staging[kValuesPerLoad_] = {};
      for (int inner_idx = 0; inner_idx < kValuesPerLoad_; ++inner_idx) {
        // Load a column index, scale it by the row length and store in
        // a staging register.
        if (nonzeros > 0) {
          Store(kDimK_ * sputnik::Load(column_indices_),
                index_staging + inner_idx);
        }
        nonzeros -= kBlockWidth;
        column_indices_ += kBlockWidth;
      }
      // Store the vector to shared memory.
      //
      // NOTE: The column indices have now been interleaved. This could
      // theoretically effect cache hit rate. If we observe detrimental
      // effects, we could stride the global memory accesses.
      Store(*reinterpret_cast<IndexType *>(index_staging), column_idxs_tile);
      column_idxs_tile += kBlockWidth;
    }
  }

  /**
   * @brief Strip-mine a 2-dimensional tile from the matrix.
   */
  __device__ __forceinline__ void Load() {
    // Stop memory optimization around the index loading.
    asm volatile("" ::: "memory");

    int *column_idxs_tile = reinterpret_cast<int *>(column_idxs_tile_base_);
#pragma unroll
    for (int x_item_idx = 0; x_item_idx < kThreadItemsX_; ++x_item_idx) {
      // Load the column index and offset the matrix pointer.
      const LoadType *matrix =
          OffsetCast<const LoadType>(matrix_base_, *column_idxs_tile);
#pragma unroll
      for (int k_item_idx = 0; k_item_idx < kThreadItemsK_; ++k_item_idx) {
        int fragment_offset = x_item_idx * kThreadItemsK_ + k_item_idx;
        Store(sputnik::Load(matrix), matrix_fragment_ + fragment_offset);
        matrix += kBlockWidth;
      }
      // Increment our column indices pointer for the next iteration.
      ++column_idxs_tile;
    }
    // Increment our base pointer for the next call.
    matrix_base_ =
        OffsetCast<const LoadType>(matrix_base_, kBlockItemsK * sizeof(float));
  }

  /**
   * @brief Loads any residual elements and computes partial results.
   */
  __device__ __forceinline__ void ResidueAndCompute(int residue,
                                                    const float *scalar_lhs,
                                                    float *output_fragment) {
    int *column_idxs_tile = reinterpret_cast<int *>(column_idxs_tile_base_);
    const LoadType *lhs_fragment =
        reinterpret_cast<const LoadType *>(scalar_lhs);

    // If we're only going to perform a single iteration of the inner loop,
    // pull the predicate check out of the loop.
    if ((kThreadItemsK_ == 1) && residue <= 0) return;

#pragma unroll
    for (int x_item_idx = 0; x_item_idx < kThreadItemsX_; ++x_item_idx) {
      // Load the column index and offset the matrix pointer.
      const LoadType *matrix =
          OffsetCast<const LoadType>(matrix_base_, *column_idxs_tile);

      int inner_residue = residue;
#pragma unroll
      for (int k_item_idx = 0; k_item_idx < kThreadItemsK_; ++k_item_idx) {
        // NOTE: We special-case kThreadItemsK_ == 1 to generate cleaner
        // branching code in this unrolled loop.
        if ((kThreadItemsK_ == 1) || residue > 0) {
          int fragment_offset = x_item_idx * kThreadItemsK_ + k_item_idx;
          Store(sputnik::Load(matrix), matrix_fragment_ + fragment_offset);

          // Compute the FMAs.
          VectorCompute<LoadType>::Dot(matrix_fragment_[fragment_offset],
                                       lhs_fragment[k_item_idx],
                                       output_fragment + x_item_idx);
        }
        matrix += kBlockWidth;
        inner_residue -= kBlockWidth * kValuesPerLoad_;
      }
      // Increment our column indices pointer for the next iteration.
      ++column_idxs_tile;
    }
  }
};

}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_SDDMM_DENSE_TO_REG_H_
