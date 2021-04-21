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

#ifndef THIRD_PARTY_SPUTNIK_SPMM_DENSE_TILE_H_
#define THIRD_PARTY_SPUTNIK_SPMM_DENSE_TILE_H_

/**
 * @file @brief Defines functor for efficiently loading tiles from
 * a dense matrix based on sparse-matrix meta-data.
 */

#include "sputnik/cuda_utils.h"
#include "sputnik/load_store.h"
#include "sputnik/spmm/predicate_utils.h"
#include "sputnik/tiling_utils.h"
#include "sputnik/type_utils.h"
#include "sputnik/vector_utils.h"

namespace sputnik {

/**
 * @brief Functor for loading a 2-d tile from a dense matrix.
 *
 * Which rows the 1-d strips of the dense matrix are loaded from is
 * dependent on the indices passed in.
 */
template <typename Value, int kBlockItemsK, int kBlockItemsX, int kBlockWidth,
          int kResidueUnroll>
struct DenseTile {
  // The type of a single element of a Value.
  typedef typename TypeUtils<Value>::ScalarValue ScalarValue;

  // Compile-time checks on the k-dimension and x-dimension tile sizes.
  static_assert(kBlockItemsK * kBlockItemsX >= kBlockWidth,
                "Dense tile size must be >= thread block width.");
  static_assert(kBlockItemsK * kBlockItemsX % kBlockWidth == 0,
                "Dense tile size must be divisible by thread block width.");

  // Compile-time checks on the Value template parameter.
  static_assert(sizeof(Value) >= sizeof(ScalarValue),
                "Value size must be >= data type size.");
  static_assert(sizeof(Value) % sizeof(ScalarValue) == 0,
                "Value size must be divisbile by data type size.");

  //
  /// Static members.
  //

  // The type of a single element of an Index.
  typedef typename Value2Index<ScalarValue>::Index ScalarIndex;

  // The number of values that will be loaded per-thread, per-load.
  static constexpr int kValuesPerLoad_ = sizeof(Value) / sizeof(ScalarValue);

  // TODO(tgale): This version of the code does not support vector accesses
  // with a 1-dimension tiling of the output. Can we generalize it to support
  // both cases? Do we need to do this, or are 2D schemes always preferable?
  static_assert(kValuesPerLoad_ * kBlockWidth <= kBlockItemsX,
                "The number of values loaded from a row of rhs "
                "at once must not exceed kBlockItemsX.");

  // The number of data items in the x-dimension that each thread owns.
  // Equivalently, the number of outputs each thread owns.
  static constexpr int kThreadItemsX_ =
      kBlockItemsX / kBlockWidth / kValuesPerLoad_;

  // Compile time check on the residue unrolling parameter.
  static_assert(kBlockItemsK % kResidueUnroll == 0,
                "k-dimension tile size must be divisible by the residue"
                " unrolling factor.");

  // The number of outer loop iterations for the residue handling.
  static constexpr int kResidueOuterLimit_ = kBlockItemsK / kResidueUnroll;

  // The number of inner loop iterations for the residue handling.
  static constexpr int kResidueInnerLimit_ = kResidueUnroll;

  // The number of elements to compute on per-scalar rhs element.
  static constexpr int kElementsPerScalar_ =
      TypeUtils<Value>::kElementsPerScalar;

  // Data type of our accumulator registers.
  typedef typename TypeUtils<Value>::Accumulator Accumulator;

  // Shorthand for n-dim a predicate vector of the appropriate size.
  typedef PredicateVector<kThreadItemsX_> Predicates;

  //
  /// Member variables.
  //

  // The number of columns in the rhs matrix in bytes.
  const int rhs_columns_;

  // The dense matrix pointer in global memory.
  const Value *matrix_base_;

  // The loaded dense matrix row offsets in smem.
  const ScalarIndex *row_offsets_base_;

  // The register file fragment to load the dense values into.
  Value *matrix_fragment_;

  // Constructor. Set the initial pointer offsets.
  __device__ __forceinline__ DenseTile(int rhs_columns, int offset,
                                       int thread_idx_x,
                                       const ScalarValue *__restrict__ matrix,
                                       const ScalarIndex *row_offsets,
                                       ScalarValue *matrix_fragment)
      : rhs_columns_(rhs_columns * sizeof(ScalarValue)) {
    matrix_base_ =
        reinterpret_cast<const Value *>(matrix + offset) + thread_idx_x;
    row_offsets_base_ = row_offsets;
    matrix_fragment_ = reinterpret_cast<Value *>(matrix_fragment);
  }

  /**
   * @brief Strip mine a 2-dimensional tile from the dense matrix.
   */
  __device__ __forceinline__ void Load(const Predicates &predicates_n) {
    const ScalarIndex *row_offsets = row_offsets_base_;

#pragma unroll
    for (int k_item_idx = 0; k_item_idx < kBlockItemsK; ++k_item_idx) {
      // Load the row offsets and extract into 32-bit integer values.
      int scaled_indices[kElementsPerScalar_];
      Convert(row_offsets, scaled_indices);
#pragma unroll
      for (int elt_idx = 0; elt_idx < kElementsPerScalar_; ++elt_idx) {
        // Possibly scale the indices s.t. they properly index into the
        // right-hand size dense matrix.
        if (TypeUtils<ScalarValue>::IsMixed()) {
          scaled_indices[elt_idx] *= rhs_columns_;
        }

        // Increment the matrix pointer.
        const Value *matrix =
            OffsetCast<const Value>(matrix_base_, scaled_indices[elt_idx]);
#pragma unroll
        for (int x_item_idx = 0; x_item_idx < kThreadItemsX_; ++x_item_idx) {
          // NOTE: There are a few different ways we could have expressed
          // this loop while avoiding out-of-bounds memory accesses. See
          // the documentation for PredicateVector for more info.
          if (predicates_n.GetBit(x_item_idx)) {
            int fragment_offset =
                k_item_idx * kThreadItemsX_ * kElementsPerScalar_ +
                elt_idx * kThreadItemsX_ + x_item_idx;
            matrix_fragment_[fragment_offset] = sputnik::Load(matrix);

            // Increment our matrix pointer for the next iteration.
            matrix += kBlockWidth;
          }
        }
      }
      // Increment our row offsets pointer for the next iteration.
      ++row_offsets;
    }
  }

  /**
   * @brief Load the residue and compute the matrix product.
   *
   * If kResidueUnroll > 1, this function will assume that it can unroll
   * `kResidueUnroll` iterations of the inner load & compute loop. To
   * maintain correctness, this requires that the lhs tile and indices are
   * padded with zeros past the size of the residue to the nearest multiple
   * of `kResidueUnroll`. This potentially enables the compiler to use
   * wide shared memory loads.
   */
  __device__ __forceinline__ void ResidueLoadAndCompute(
      int residue, const Predicates &predicates_n, const ScalarValue *lhs_tile,
      float *output_fragment) {
    const ScalarIndex *row_offsets = row_offsets_base_;

    // If we're only going to perform a single iteration of the inner loop,
    // pull the predicate check out of the loop.
    if ((kThreadItemsX_ == 1) && !predicates_n.GetBit(0)) return;

#pragma unroll
    for (int k_outer_idx = 0; k_outer_idx < kResidueOuterLimit_;
         ++k_outer_idx) {
      // The compiler doesn't like unrolling this loop with this bail-out,
      // but for some reason if we use "return" instead of "break", and we
      // have an asm block following the loop the compiler does it just fine.
      //
      // TODO(tgale): The empty asm block at the end of this loop is very
      // weird. Explore ways to unroll this loop without this block.
      if (residue <= 0) return;

#pragma unroll
      for (int k_inner_idx = 0; k_inner_idx < kResidueInnerLimit_;
           ++k_inner_idx) {
        const int k_item_idx = k_inner_idx + k_outer_idx * kResidueInnerLimit_;

        // Load the row offsets and extract into 32-bit integer values.
        int scaled_indices[kElementsPerScalar_];
        Convert(row_offsets, scaled_indices);

        // Load the weight from smem and extract into 32-bit float values.
        float lhs_values[kElementsPerScalar_];
        Convert(lhs_tile + k_item_idx, lhs_values);
#pragma unroll
        for (int elt_idx = 0; elt_idx < kElementsPerScalar_; ++elt_idx) {
          // Possibly scale the indices s.t. they properly index into the
          // right-hand size dense matrix.
          if (TypeUtils<ScalarValue>::IsMixed()) {
            scaled_indices[elt_idx] *= rhs_columns_;
          }

          // Increment hte matrix pointer.
          const Value *matrix =
              OffsetCast<const Value>(matrix_base_, scaled_indices[elt_idx]);
#pragma unroll
          for (int x_item_idx = 0; x_item_idx < kThreadItemsX_; ++x_item_idx) {
            // NOTE: We special-case kThreadItemsX_ == 1 to generate cleaner
            // branching code in this unrolled loop.
            if ((kThreadItemsX_ == 1) || predicates_n.GetBit(x_item_idx)) {
              float *outputs = output_fragment + x_item_idx * kValuesPerLoad_ *
                                                     kElementsPerScalar_;

              // Load the rhs & lhs and compute immediately.
              VectorCompute<Value>::FMA(
                  lhs_values[elt_idx], sputnik::Load(matrix),
                  reinterpret_cast<Accumulator *>(outputs));

              // Increment our matrix pointer for the next iteration.
              matrix += kBlockWidth;
            }
          }
        }
        // Increment our row offsets pointer for the next iteration.
        ++row_offsets;
      }
      // Update the number of items left to process.
      residue -= kResidueInnerLimit_;
    }
    // NOTE: See above TODO for why this empty asm block exists.
    asm("");
  }
};

}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_SPMM_DENSE_TILE_H_
