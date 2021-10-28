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

#ifndef THIRD_PARTY_SPUTNIK_MEMORY_ALIGNER_H_
#define THIRD_PARTY_SPUTNIK_MEMORY_ALIGNER_H_

/**
 * @file @brief Defines utilities for aligning rows of a sparse matrix
 * to enable the use of vector memory accesses.
 */

#include "sputnik/type_utils.h"
#include "sputnik/vector_utils.h"

namespace sputnik {

template <typename Value, int kBlockWidth>
struct MemoryAligner {
  //
  /// Static members.
  //

  // The type of a single element of a Value.
  typedef typename TypeUtils<Value>::ScalarValue ScalarValue;

  // The type of a single element of an Index.
  typedef typename Value2Index<ScalarValue>::Index ScalarIndex;

  // The number of values we need to align the pointers to.
  static constexpr int kValueAlignment = sizeof(Value) / sizeof(ScalarValue);

  // Pre-calculated mask used to efficiently align the row offset.
  static constexpr uint32_t kAlignmentMask = ~(kValueAlignment - 1);

  // The maximum number of values and indices that we could have to mask.
  static constexpr int kMaxValuesToMask = kValueAlignment - 1;

  // The number of masking iterations we need to perform. For most kernels,
  // this will be one and the loop in MaskPrefix should be compiled away.
  static constexpr int kMaskSteps =
      (kMaxValuesToMask + kBlockWidth - 1) / kBlockWidth;

  //
  /// Member variables.
  //

  // The row offset in the sparse matrix values & column indices buffers.
  int row_offset_;

  // The number of nonzeros in this row of the sparse matrix.
  int nonzeros_;

  // The number of values we need to mask out at the start of the first
  // computed tile.
  int values_to_mask_;

  // Constructor. Save the row offset and initialize the masked region size.
  __device__ __forceinline__ MemoryAligner(int row_offset, int nonzeros) {
    row_offset_ = row_offset;
    nonzeros_ = nonzeros;

    // NOTE: kValueAlignment is guaranteed to be 2 or 4, so we can express
    // modulo by kValueAlignment in this way. Switching to this expression
    // produced much cleaner code than relying on the compiler to optimize
    // away the modulo.
    values_to_mask_ = row_offset & (kValueAlignment - 1);
  }

  /**
   * @brief Potentially align the sparse matrix pointers to the vector width.
   *
   * NOTE: This code effectively reduces the row offset to the nearest 128
   * or 64-byte aligned value. All memory allocated with cudaMalloc is 128-
   * byte aligned, thus this code will never cause our kernels to issue out-
   * of-bounds memory accesses to the region before the allocations used to
   * store the sparse matrix.
   */
  __device__ __forceinline__ int AlignedRowOffset() {
    return row_offset_ & kAlignmentMask;
  }

  __device__ __forceinline__ int AlignedNonzeros() {
    return nonzeros_ + values_to_mask_;
  }

  /**
   * @brief Mask the first few values of the loaded sparse matrix tile in
   * the case that we offset the pointers.
   */
  __device__ __forceinline__ void MaskPrefix(
      ScalarValue* values_tile_sv, ScalarIndex* column_indices_tile_si) {
    // NOTE: The below masking code is data type agnostic. Cast input pointers
    // to float/int so that we efficiently operate on 4-byte words.
    float* values_tile = reinterpret_cast<float*>(values_tile_sv);
    int* column_indices_tile = reinterpret_cast<int*>(column_indices_tile_si);

    int mask_idx = threadIdx.x;
#pragma unroll
    for (int mask_step = 0; mask_step < kMaskSteps; ++mask_step) {
      if (mask_idx < values_to_mask_) {
        // NOTE: We set the column index for these out-of-bounds values to
        // a dummy values of zero. This will trigger a superfluous load into
        // the dense right-hand side matrix, but will never be out-of-bounds.
        // We set the value for this index to 0 s.t. the superfluous rhs value
        // is not accumulated into the output.
        values_tile[mask_idx] = 0.0f;
        column_indices_tile[mask_idx] = 0;
        mask_idx += kBlockWidth;
      }
    }
  }
};

// For scalar memory accesses, there is no need to align our pointers.
template <int kBlockWidth>
struct MemoryAligner<float, kBlockWidth> {
  //
  /// Member variables.
  //

  // The row offset in the sparse matrix values & column indices buffers.
  int row_offset_;

  // The number of nonzeros in this row of the sparse matrix.
  int nonzeros_;

  __device__ __forceinline__ MemoryAligner(int row_offset, int nonzeros) {
    row_offset_ = row_offset;
    nonzeros_ = nonzeros;
  }

  __device__ __forceinline__ int AlignedRowOffset() { return row_offset_; }

  __device__ __forceinline__ int AlignedNonzeros() { return nonzeros_; }

  __device__ __forceinline__ void MaskPrefix(float*, int*) { /* noop */
  }
};

template <int kBlockWidth>
struct MemoryAligner<half2, kBlockWidth> {
  //
  /// Member variables.
  //

  // The row offset in the sparse matrix values & column indices buffers.
  int row_offset_;

  // The number of nonzeros in this row of the sparse matrix.
  int nonzeros_;

  __device__ __forceinline__ MemoryAligner(int row_offset, int nonzeros) {
    row_offset_ = row_offset;
    nonzeros_ = nonzeros;
  }

  __device__ __forceinline__ int AlignedRowOffset() { return row_offset_; }

  __device__ __forceinline__ int AlignedNonzeros() { return nonzeros_; }

  __device__ __forceinline__ void MaskPrefix(half2*, short2*) { /* noop */
  }
};

}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_MEMORY_ALIGNER_H_
