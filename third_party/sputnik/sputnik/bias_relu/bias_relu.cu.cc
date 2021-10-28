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

#include "sputnik/bias_relu/bias_relu.h"
#include "sputnik/load_store.h"

namespace sputnik {

__global__ void BiasReluKernel(int n, int c, int d,
                               const float* __restrict__ in,
                               const float* __restrict__ bias,
                               float* __restrict__ out) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= (n * c * d)) return;

  // TODO(tgale): We could encode this in the grid dimensions.
  // We should also do specializations that can use vector
  // loads/stores.
  int channel_index = (index / d) % c;

  // Load the input and bias value and add them together. Apply
  // the ReLU and store the result.
  float x = Load(in + index) + Load(bias + channel_index);
  Store(x > 0 ? x : 0, out + index);
}

cudaError_t BiasRelu(int n, int c, int d, const float* __restrict__ in,
                     const float* __restrict__ bias, float* __restrict__ out,
                     cudaStream_t stream) {
  // Put 1k threads in each block by default.
  constexpr int kThreadsPerBlock = 1024;
  const int kNumBlocks =
      std::ceil(static_cast<float>(n * c * d) / kThreadsPerBlock);
  BiasReluKernel<<<kNumBlocks, kThreadsPerBlock, 0, stream>>>(n, c, d, in, bias,
                                                              out);
  return cudaGetLastError();
}

}  // namespace sputnik
