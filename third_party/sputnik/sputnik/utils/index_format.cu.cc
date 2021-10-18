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

#include "sputnik/load_store.h"
#include "sputnik/tiling_utils.h"
#include "sputnik/utils/index_format.h"

namespace sputnik {

namespace {

template <int kBlockWidth>
__global__ void Csr2IdxKernel(int m, int n, const int* __restrict__ row_offsets,
                              const int* __restrict__ column_indices,
                              int* __restrict__ linear_indices) {
  // Calculate this thread block's indices into the matrix.
  typedef TilingUtils<1, /* unused */ 1, kBlockWidth> Tiling;
  int m_index = Tiling::IndexM(), n_index = Tiling::IndexN();

  // If we're out of bounds in the m-dimension, exit.
  if (m_index >= m) return;

  // Load the row offset and calculate the number of nonzeros in the row.
  int row_offset = Load(row_offsets + m_index);
  int nonzeros = Load(row_offsets + m_index + 1) - row_offset;

  // If this thread has no work to do, exit.
  if ((n_index + threadIdx.x) >= nonzeros) return;

  // Calculate this threads index into the input/output sparse matrix.
  int thread_index = row_offset + n_index + threadIdx.x;

  // Load the column index, convert to a linear index, and store.
  Store(Load(column_indices + thread_index) + n * m_index,
        linear_indices + thread_index);
}

}  // namespace

cudaError_t Csr2Idx(int m, int n, int nonzeros,
                    const int* __restrict__ row_offsets,
                    const int* __restrict__ column_indices,
                    int* __restrict__ linear_indices, cudaStream_t stream) {
  // TODO(tgale): Tune this value for different matrix sizes.
  constexpr int kBlockWidth = 64;
  dim3 grid_dim(m, std::ceil(static_cast<float>(n) / kBlockWidth), 1);
  dim3 block_dim(kBlockWidth, 1, 1);

  Csr2IdxKernel<kBlockWidth><<<grid_dim, block_dim, 0, stream>>>(
      m, n, row_offsets, column_indices, linear_indices);
  return cudaGetLastError();
}

}  // namespace sputnik
