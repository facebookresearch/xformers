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

#include <cstdint>

#include "sputnik/cuda_utils.h"
#include "sputnik/depthwise/cuda_depthwise.h"
#include "sputnik/depthwise/depthwise_config.h"
#include "sputnik/matrix_utils.h"
#include "sputnik/test_utils.h"

#include "absl/random/random.h"
#include "benchmark/benchmark.h"

namespace sputnik {

void ReportThroughput(benchmark::State& state) {
  const int kDimN = state.range(0);
  const int kDimC = state.range(1);
  const int kDimH = state.range(2);
  const int kDimW = state.range(3);
  const int kKernelSize = state.range(4);
  const int kPadding = state.range(5);
  const int kStride = state.range(6);

  // Output spatial dimensions.
  const int kDimOutH = (kDimH - kKernelSize + 2 * kPadding) / kStride + 1;
  const int kDimOutW = (kDimW - kKernelSize + 2 * kPadding) / kStride + 1;

  const int kFlopsPerIteration = kKernelSize * kKernelSize *
                                 kDimOutH * kDimOutW *
                                 kDimN * kDimC * 2;
  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations()) *
      kFlopsPerIteration);
}

void BenchmarkArgs(benchmark::internal::Benchmark* b) {
  const std::vector<std::vector<int64_t>> benchmarks = {
      // TVM benchmark.
      {1, 256, 96, 96, 3, 1, 1},
      // MobileNetV1 width 1.8
      {1, 56, 112, 112, 3, 1, 1},
      {1, 115, 112, 112, 3, 1, 2},
      {1, 232, 56, 56, 3, 1, 1},
      {1, 232, 56, 56, 3, 1, 2},
      {1, 464, 28, 28, 3, 1, 1},
      {1, 464, 28, 28, 3, 1, 2},
      {1, 920, 14, 14, 3, 1, 1},
      {1, 920, 14, 14, 3, 1, 2},
      {1, 1840, 7, 7, 3, 1, 1},
      // MobileNetV1 width 1.7
      {1, 56, 112, 112, 3, 1, 1},
      {1, 108, 112, 112, 3, 1, 2},
      {1, 216, 56, 56, 3, 1, 1},
      {1, 216, 56, 56, 3, 1, 2},
      {1, 432, 28, 28, 3, 1, 1},
      {1, 432, 28, 28, 3, 1, 2},
      {1, 872, 14, 14, 3, 1, 1},
      {1, 872, 14, 14, 3, 1, 2},
      {1, 1744, 7, 7, 3, 1, 1},
      // MobileNet width 1.6
      {1, 48, 112, 112, 3, 1, 1},
      {1, 102, 112, 112, 3, 1, 2},
      {1, 208, 56, 56, 3, 1, 1},
      {1, 208, 56, 56, 3, 1, 2},
      {1, 408, 28, 28, 3, 1, 1},
      {1, 408, 28, 28, 3, 1, 2},
      {1, 816, 14, 14, 3, 1, 1},
      {1, 816, 14, 14, 3, 1, 2},
      {1, 1640, 7, 7, 3, 1, 1},
      // MobileNet width 1.5
      {1, 48, 112, 112, 3, 1, 1},
      {1, 96, 112, 112, 3, 1, 2},
      {1, 192, 56, 56, 3, 1, 1},
      {1, 192, 56, 56, 3, 1, 2},
      {1, 384, 28, 28, 3, 1, 1},
      {1, 384, 28, 28, 3, 1, 2},
      {1, 768, 14, 14, 3, 1, 1},
      {1, 768, 14, 14, 3, 1, 2},
      {1, 1536, 7, 7, 3, 1, 1},
      // MobileNet width 1.4
      {1, 48, 112, 112, 3, 1, 1},
      {1, 89, 112, 112, 3, 1, 2},
      {1, 176, 56, 56, 3, 1, 1},
      {1, 176, 56, 56, 3, 1, 2},
      {1, 360, 28, 28, 3, 1, 1},
      {1, 360, 28, 28, 3, 1, 2},
      {1, 720, 14, 14, 3, 1, 1},
      {1, 720, 14, 14, 3, 1, 2},
      {1, 1432, 7, 7, 3, 1, 1},
      // MobileNet width 1.3
      {1, 40, 112, 112, 3, 1, 1},
      {1, 83, 112, 112, 3, 1, 2},
      {1, 168, 56, 56, 3, 1, 1},
      {1, 168, 56, 56, 3, 1, 2},
      {1, 336, 28, 28, 3, 1, 1},
      {1, 336, 28, 28, 3, 1, 2},
      {1, 664, 14, 14, 3, 1, 1},
      {1, 664, 14, 14, 3, 1, 2},
      {1, 1328, 7, 7, 3, 1, 1},
  };

  for (const auto& a : benchmarks) b->Args(a);
}

void BM_CudaDepthwise_GenericFloat(benchmark::State& state) {
  const int kDimN = state.range(0);
  const int kDimC = state.range(1);
  const int kDimH = state.range(2);
  const int kDimW = state.range(3);
  const int kKernelSize = state.range(4);
  const int kPadding = state.range(5);
  const int kStride = state.range(6);

  // Output spatial dimensions.
  const int kDimOutH = (kDimH - kKernelSize + 2 * kPadding) / kStride + 1;
  const int kDimOutW = (kDimW - kKernelSize + 2 * kPadding) / kStride + 1;

  // We currently don't predicate on the input loads, so to be safe we
  // just allocate way off the end.
  const int kInputMult = 2;

  absl::BitGen generator;
  CudaMatrix<float> in(kDimN * kDimC, kDimH * kDimW * kInputMult, &generator);
  CudaMatrix<float> filters(kDimC, kKernelSize * kKernelSize, &generator);
  CudaMatrix<float> out(kDimN * kDimC, kDimOutH * kDimOutW, &generator);

  int batch_size = 10;
  while (state.KeepRunningBatch(batch_size)) {
    for (int i = 0; i < batch_size; ++i) {
      CUDA_CALL(CudaDepthwise(kDimN, kDimC, kDimH, kDimW, in.Values(),
                              kKernelSize, kPadding, kStride, filters.Values(),
                              out.Values(), /*stream=*/0));
    }
    CUDA_CALL(cudaStreamSynchronize(/*stream=*/0));
  }
  ReportThroughput(state);
}

BENCHMARK(BM_CudaDepthwise_GenericFloat)->Apply(BenchmarkArgs)->UseRealTime();

}  // namespace sputnik
