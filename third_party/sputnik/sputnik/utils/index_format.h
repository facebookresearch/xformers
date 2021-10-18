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

#ifndef THIRD_PARTY_SPUTNIK_UTILS_INDEX_FORMAT_H_
#define THIRD_PARTY_SPUTNIK_UTILS_INDEX_FORMAT_H_

#include "sputnik/cuda_utils.h"

namespace sputnik {

/**
 * @brief Converts compressed sparse row indices to linear indices.
 *
 * Converts `column_index[i]` to `column_index[i] * row_index * n`, where
 * `row_index` is the row that this column index belongs to. We call this
 * "index format" or "1-dimensional coordinate format".
 *
 * @param m The number of rows in the sparse matrix.
 * @param n The number of columns in the sparse matrix.
 * @param noneros The number of nonzeros in the sparse matrix.
 * @param row_offsets The offsets of each row of nonzeros and column indices
 * in the sparse matrix. Device-side buffer of `m + 1` ints.
 * @param column_indices The column indices of each nonzero values in the
 * sparse matrix. Device-side buffer of `nonzeros` ints.
 * @param linear_indices Output buffer for the linear sparse matrix indices.
 * Device-side buffer of `nonzeros` ints.
 * @param stream The CUDA stream to launch the kernels in.
 */
cudaError_t Csr2Idx(int m, int n, int nonzeros,
                    const int* __restrict__ row_offsets,
                    const int* __restrict__ column_indices,
                    int* __restrict__ linear_indices, cudaStream_t stream);

}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_UTILS_INDEX_FORMAT_H_
