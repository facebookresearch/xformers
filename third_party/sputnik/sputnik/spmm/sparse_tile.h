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

#ifndef THIRD_PARTY_SPUTNIK_SPMM_SPARSE_TILE_H_
#define THIRD_PARTY_SPUTNIK_SPMM_SPARSE_TILE_H_

/**
 * @file @brief Defines functor for efficiently loading tiles from a
 * sparse matrix.
 */

#include "sputnik/cuda_utils.h"
#include "sputnik/load_store.h"
#include "sputnik/tiling_utils.h"
#include "sputnik/type_utils.h"
#include "sputnik/vector_utils.h"

namespace sputnik {

/**
 * @brief Functor for iteratively loading tiles of a sparse matrix.
 */
template <typename Value, int kBlockItemsK, int kBlockWidth>
struct SparseTile {
  // The type of a single element of a Value.
  typedef typename TypeUtils<Value>::ScalarValue ScalarValue;

  static_assert(kBlockItemsK >= kBlockWidth,
                "Sparse tile K-items must be >= thread block width.");
  static_assert(kBlockItemsK % kBlockWidth == 0,
                "Sparse tile K-items must be divisible by block width.");
  static_assert((sizeof(Value) / sizeof(ScalarValue)) <=
                    (kBlockItemsK / kBlockWidth),
                "The number of values per load must be <= values per thread.");

  // Complementary index type to our load type.
  typedef typename Value2Index<ScalarValue>::Index ScalarIndex;
  typedef typename Value2Index<Value>::Index Index;

  //
  /// Static members.
  //

  // The number of values that will be loaded per-thread, per-load.
  static constexpr int kValuesPerLoad_ = sizeof(Value) / sizeof(ScalarValue);

  // The number of data items in the k-dimension that each thread owns.
  static constexpr int kThreadItemsK_ =
      kBlockItemsK / kBlockWidth / kValuesPerLoad_;

  // The number of elements to compute on per-scalar rhs element.
  static constexpr int kElementsPerScalar_ =
      TypeUtils<Value>::kElementsPerScalar;

  //
  /// Member variables.
  //

  // The number of columns in the rhs matrix in bytes.
  const int rhs_columns_;

  // The sparse matrix values array.
  const Value *values_;

  // The sparse matrix column indices for each value. Scaled in ctor
  // by sizeof(ScalarValue) s.t. offset is in bytes, and we can directly
  // add to rhs matrix pointer after loading from smem.
  const Index *column_idxs_;

  // Smem tile for sparse matrix values.
  Value *values_tile_base_;

  // Smem tile for sparse matrix indices.
  Index *column_idxs_tile_base_;

  // Constructor. Set the initial pointer offsets.
  __device__ __forceinline__
  SparseTile(int rhs_columns, int offset, int thread_idx_x,
             const ScalarValue *__restrict__ values,
             const ScalarIndex *__restrict__ column_idxs,
             ScalarValue *values_tile, ScalarIndex *column_idxs_tile)
      : rhs_columns_(rhs_columns * sizeof(ScalarValue)),
        values_(reinterpret_cast<const Value *>(values + offset) +
                thread_idx_x),
        column_idxs_(reinterpret_cast<const Index *>(column_idxs + offset) +
                     thread_idx_x),
        values_tile_base_(reinterpret_cast<Value *>(values_tile) +
                          thread_idx_x),
        column_idxs_tile_base_(reinterpret_cast<Index *>(column_idxs_tile) +
                               thread_idx_x) {}

  /**
   * @brief Strip-mine a 1-dimensional tile from the sparse matrix.
   */
  __device__ __forceinline__ void Load() {
    // Set the initial tile pointers.
    Value *values_tile = values_tile_base_;
    Index *column_idxs_tile = column_idxs_tile_base_;

#pragma unroll
    for (int k_item_idx = 0; k_item_idx < kThreadItemsK_; ++k_item_idx) {
      // Load the values into smem.
      Store(sputnik::Load(values_), values_tile);

      if (TypeUtils<ScalarValue>::IsMixed()) {
        // If we're using 16-bit indices, don't scale to avoid overflow.
        Store(sputnik::Load(column_idxs_), column_idxs_tile);
      } else {
        // Load the column indices into smem. Scale the index by the number
        // of columns in the rhs matrix for our lookups into the dense rhs.
        VectorCompute<Value>::Mul(rhs_columns_, sputnik::Load(column_idxs_),
                                  column_idxs_tile);
      }

      // Increment out pointers for the next iteration.
      values_ += kBlockWidth;
      column_idxs_ += kBlockWidth;

      // These increments should not occur if kThreadItemsK_ == 1.
      values_tile += kBlockWidth;
      column_idxs_tile += kBlockWidth;
    }
  }

  /**
   * @brief Zeros all values & indices in smem.
   *
   * This helper is useful for allowing us to unroll the dense residue handling
   * by a factor of 2 or 4. This in turn allows us to use vector smem operations
   * and reduce control (at the expense of computing 1-3 more values).
   */
  __device__ __forceinline__ void ZeroTiles() {
    // Set the initial tile pointers.
    Value *values_tile = values_tile_base_;
    Index *column_idxs_tile = column_idxs_tile_base_;

    const float kZeroValues[kValuesPerLoad_] = {};
    const int kZeroIndices[kValuesPerLoad_] = {};
#pragma unroll
    for (int k_item_idx = 0; k_item_idx < kThreadItemsK_; ++k_item_idx) {
      Store(*reinterpret_cast<const Value *>(kZeroValues), values_tile);
      Store(*reinterpret_cast<const Index *>(kZeroIndices), column_idxs_tile);
      values_tile += kBlockWidth;
      column_idxs_tile += kBlockWidth;
    }
  }

  /**
   * @brief Residue handling for loads from a sparse matrix.
   */
  __device__ __forceinline__ void Residue(int residue) {
    // Update our global and shared memory pointers for our switch to
    // scalar ops.
    constexpr int kResidueUpdateStride =
        -1 * static_cast<int>(sizeof(ScalarValue)) * (kValuesPerLoad_ - 1);
    const int kResidueUpdate = static_cast<int>(threadIdx.x) *
        kResidueUpdateStride;

    const ScalarValue *values =
        OffsetCast<const ScalarValue>(values_, kResidueUpdate);
    const ScalarIndex *column_idxs =
        OffsetCast<const ScalarIndex>(column_idxs_, kResidueUpdate);

    // Set the initial smem tile pointers.
    ScalarValue *values_tile =
        OffsetCast<ScalarValue>(values_tile_base_, kResidueUpdate);
    ScalarIndex *column_idxs_tile =
        OffsetCast<ScalarIndex>(column_idxs_tile_base_, kResidueUpdate);

    constexpr int kScalarThreadItemsK = kBlockItemsK / kBlockWidth;
#pragma unroll
    for (int k_item_idx = 0; k_item_idx < kScalarThreadItemsK; ++k_item_idx) {
      // The compiler doesn't like unrolling this loop with this bail-out,
      // but for some reason if we use "return" instead of "break", and we
      // have an asm block following the loop the compiler does it just fine.
      //
      // TODO(tgale): The empty asm block at the end of this loop is very
      // weird. Explore ways to unroll this loop without this block.
      if (residue <= static_cast<int>(threadIdx.x)) return;

      // Load the values into smem.
      Store(sputnik::Load(values), values_tile);

      if (TypeUtils<ScalarValue>::IsMixed()) {
        // If we're using 16-bit indices, don't scale to avoid overflow.
        Store(sputnik::Load(column_idxs), column_idxs_tile);
      } else {
        // Load the column indices into smem. Scale the index by the number
        // of columns in the rhs matrix for our lookups into the dense rhs.
        VectorCompute<ScalarValue>::Mul(
            rhs_columns_, sputnik::Load(column_idxs), column_idxs_tile);
      }

      // Increment our pointers for the next iteration.
      values += kBlockWidth;
      column_idxs += kBlockWidth;
      values_tile += kBlockWidth;
      column_idxs_tile += kBlockWidth;
      residue -= kBlockWidth;
    }
    // NOTE: See above TODO for why this empty asm block exists.
    asm("");
  }
};

}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_SPMM_SPARSE_TILE_H_
