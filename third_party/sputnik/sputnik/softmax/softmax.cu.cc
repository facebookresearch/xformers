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

#include <cmath>

#include "sputnik/load_store.h"
#include "sputnik/softmax/softmax.h"

namespace sputnik {

namespace {

__global__ void SoftmaxKernel(int m, int n, const float* __restrict__ input,
                              float* __restrict__ output) {
  // Calculate the index of the row that this block will process.
  int m_index = blockIdx.x * blockDim.y + threadIdx.y;
  if (m_index >= m) return;

  // Step 1: Find the maximum value in our row.
  const float* in = input + m_index * n;
  float max = -INFINITY;
  for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
    float x = Load(in + idx);
    max = x > max ? x : max;
  }
  for (int idx = 1; idx < blockDim.x; idx *= 2) {
    float x = __shfl_xor_sync(0xffffffff, max, idx);
    max = x > max ? x : max;
  }

  // Step 2: Compute the normalization constant. Invert the norm
  // once so we don't need to do repeated division.
  float norm = 0.0f;
  for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
    norm += expf(Load(in + idx) - max);
  }
  for (int idx = 1; idx < blockDim.x; idx *= 2) {
    norm += __shfl_xor_sync(0xffffffff, norm, idx);
  }
  norm = 1.0f / norm;

  // step 3: Normalize the exponentials of the input and store the
  // results.
  float* out = output + m_index * n;
  for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
    Store(expf(Load(in + idx) - max) * norm, out + idx);
  }
}

}  // namespace

cudaError_t Softmax(int m, int n, const float* __restrict__ input,
                    float* __restrict__ output, cudaStream_t stream) {
  // NOTE: SoftmaxKernel currently only supports 1 warp per row
  // of the input matrix. We launch two warps per block, with each
  // mapped to different rows to enable us to hit max occupancy.
  constexpr int kBlockWidth = 32;
  constexpr int kWarpsPerBlock = 2;
  dim3 grid_dim(std::ceil(static_cast<float>(m) / kWarpsPerBlock));
  dim3 block_dim(kBlockWidth, kWarpsPerBlock);

  SoftmaxKernel<<<grid_dim, block_dim, 0, stream>>>(m, n, input, output);
  return cudaGetLastError();
}

}  // namespace sputnik
