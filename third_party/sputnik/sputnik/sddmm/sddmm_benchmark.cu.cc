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

#include "sputnik/cuda_utils.h"
#include "sputnik/matrix_utils.h"
#include "sputnik/sddmm/cuda_sddmm.h"

#include "absl/random/random.h"
#include "benchmark/benchmark.h"

namespace sputnik {

void ReportThroughput(benchmark::State& state) {
  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations()) *
      state.range(3) * state.range(1) * 2);
}

void BenchmarkArgs(benchmark::internal::Benchmark* b) {
  std::vector<int> dims = {512, 1024, 2048, 4096, 8192};
  std::vector<float> sparsities = {.25f, .2f, .15f, .1f, .05f};

  for (const auto& d : dims) {
    for (const auto& s : sparsities) {
      b->Args({d, d, d, static_cast<int>(d * d * s)});
    }
  }
}

void BM_CudaSddmm_GenericFloat(benchmark::State& state) {
  const int kDimM = state.range(0);
  const int kDimK = state.range(1);
  const int kDimN = state.range(2);
  const int kNonZeros = state.range(3);

  const int kRowPadding = 0;

  // Create the sparse matrix on the gpu.
  absl::BitGen generator;
  CudaSparseMatrix<float> output_matrix(kDimM, kDimN, kNonZeros, RANDOM_UNIFORM,
                                        &generator, SORTED, kRowPadding);

  // Create the dense matrix on the gpu.
  CudaMatrix<float> lhs_matrix(kDimM, kDimK, &generator);
  CudaMatrix<float> rhs_matrix(kDimN, kDimK, &generator);

  int batch_size = 10;
  while (state.KeepRunningBatch(batch_size)) {
    for (int i = 0; i < batch_size; ++i) {
      CUDA_CALL(CudaSddmm(
          output_matrix.Rows(),
          lhs_matrix.Columns(),
          output_matrix.Columns(),
          output_matrix.NumElementsWithPadding(),
          output_matrix.RowIndices(),
          output_matrix.RowOffsets(),
          output_matrix.ColumnIndices(),
          lhs_matrix.Values(),
          rhs_matrix.Values(),
          output_matrix.Values(), 0));
    }
    CUDA_CALL(cudaStreamSynchronize(nullptr));
  }
  ReportThroughput(state);
}

BENCHMARK(BM_CudaSddmm_GenericFloat)->Apply(BenchmarkArgs)->UseRealTime();

}  // namespace sputnik
