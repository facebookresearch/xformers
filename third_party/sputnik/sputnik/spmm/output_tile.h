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

#ifndef THIRD_PARTY_SPUTNIK_SPMM_OUTPUT_TILE_H_
#define THIRD_PARTY_SPUTNIK_SPMM_OUTPUT_TILE_H_

/**
 * @file @brief Defines functor for efficiently storing results to the
 * output matrix tile.
 */

#include "sputnik/load_store.h"
#include "sputnik/spmm/predicate_utils.h"
#include "sputnik/type_utils.h"

namespace sputnik {

/**
 * @brief Functor for storing outputs efficiently from accumulator registers.
 */
template <typename Value, int kBlockItemsX, int kBlockWidth>
struct OutputTile {
  //
  /// Static members.
  //

  // The type of a single element of a Value.
  typedef typename TypeUtils<Value>::ScalarValue ScalarValue;

  // The number of values that will be stored per-thread, per-store.
  static constexpr int kValuesPerStore_ = sizeof(Value) / sizeof(ScalarValue);

  // The number of outputs that each thread owns and is responsible
  // for writing back to global memory.
  static constexpr int kThreadItemsX_ =
      kBlockItemsX / kBlockWidth / kValuesPerStore_;

  // Shorthand for a predicate vector of the appropriate size.
  typedef PredicateVector<kThreadItemsX_> Predicates;

  // The number of elements to compute on per-scalar rhs element.
  static constexpr int kElementsPerScalar_ =
      TypeUtils<ScalarValue>::kElementsPerScalar;

  //
  /// Member variables.
  //

  // The register file fragment with the results to store.
  const float* output_fragment_;

  // The output matrix pointer in global memory.
  Value* output_matrix_;

  __device__ __forceinline__ OutputTile(int row_offset, int column_offset,
                                        int cols, int thread_idx_x,
                                        const float* output_fragment,
                                        ScalarValue* output_matrix) {
    output_fragment_ = output_fragment;
    const int output_offset = row_offset * cols + column_offset;
    output_matrix_ =
        reinterpret_cast<Value*>(output_matrix + output_offset) + thread_idx_x;
  }

  __device__ __forceinline__ void Store(const Predicates& predicates_n) {
#pragma unroll
    for (int x_item_idx = 0; x_item_idx < kThreadItemsX_; ++x_item_idx) {
      // NOTE: There are a few different ways we could have expressed
      // this loop while avoiding out-of-bounds memory accesses. See
      // the documentation for PredicateVector for more info.
      if (predicates_n.GetBit(x_item_idx)) {
        // TODO(tgale): The below branch is a hack to avoid a slight increase
        // in register usage in the float32 variants of these kernels with
        // the mixed-precision expression. Figure out a way to express this
        // without the branch and without altering the register allocation
        // for the single-precision kernels.
        if (TypeUtils<Value>::IsMixed()) {
          // Convert the accumulated results into the output representation.
          Value out;
          const int fragment_offset =
              x_item_idx * kElementsPerScalar_ * kValuesPerStore_;
          Convert(output_fragment_ + fragment_offset, &out);
          sputnik::Store(out, output_matrix_);
        } else {
          const Value* output_fragment =
              reinterpret_cast<const Value*>(output_fragment_);
          *output_matrix_ = output_fragment[x_item_idx];
        }
        // Increment the pointers for the next iteration.
        output_matrix_ += kBlockWidth;
      }
    }
  }
};

}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_SPMM_OUTPUT_TILE_H_
