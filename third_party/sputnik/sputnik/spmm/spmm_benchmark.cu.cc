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
#include <cstdint>

#include "sputnik/cuda_utils.h"
#include "sputnik/matrix_utils.h"
#include "sputnik/spmm/cuda_spmm.h"
#include "sputnik/spmm/spmm_config.h"
#include "sputnik/test_utils.h"

#include "absl/random/random.h"
#include "benchmark/benchmark.h"

namespace sputnik {

void ReportThroughput(benchmark::State& state) {
  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations()) *
      state.range(3) * state.range(2) * 2);
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

void BM_CudaSpmm_GenericFloat(benchmark::State& state) {
  const int kDimM = state.range(0);
  const int kDimK = state.range(1);
  const int kDimN = state.range(2);
  const int kNonZeros = state.range(3);

  const int kRowPadding = 0;

  // Create the sparse matrix on the gpu.
  absl::BitGen generator;
  CudaSparseMatrix<float> sparse_matrix_gpu(
      kDimM, kDimK, kNonZeros, RANDOM_UNIFORM, &generator, SORTED, kRowPadding);

  // Create the dense matrix on the gpu.
  CudaMatrix<float> matrix_gpu(kDimK, kDimN, &generator);

  // Create the output matrix on the gpu.
  CudaMatrix<float> output_matrix_gpu(kDimM, kDimN, &generator);

  int batch_size = 10;
  while (state.KeepRunningBatch(batch_size)) {
    for (int i = 0; i < batch_size; ++i) {
      CUDA_CALL(CudaSpmm(
          kDimM, kDimK, kDimN, sparse_matrix_gpu.NumElementsWithPadding(),
          sparse_matrix_gpu.RowIndices(), sparse_matrix_gpu.Values(),
          sparse_matrix_gpu.RowOffsets(), sparse_matrix_gpu.ColumnIndices(),
          matrix_gpu.Values(), output_matrix_gpu.Values(), 0));
    }
    CUDA_CALL(cudaStreamSynchronize(nullptr));
  }
  ReportThroughput(state);
}

BENCHMARK(BM_CudaSpmm_GenericFloat)->Apply(BenchmarkArgs)->UseRealTime();

void ReportGemmThroughput(benchmark::State& state) {
  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations()) *
      state.range(0) * state.range(1) * state.range(2) * 2);
}

void GemmBenchmarkArgs(benchmark::internal::Benchmark* b) {
  std::vector<int> dims = {512, 1024, 2048, 4096, 8192};

  for (const auto& d : dims) {
    b->Args({d, d, d});
  }
}

void BM_CublasGemm_ColumnMajor(benchmark::State& state) {
  int m = state.range(0);
  int k = state.range(1);
  int n = state.range(2);

  // Create the lhs, rhs, and output matrices on gpu.
  absl::BitGen generator;
  CudaMatrix<float> lhs_gpu(m, k, &generator);
  CudaMatrix<float> rhs_gpu(k, n, &generator);
  CudaMatrix<float> output_gpu(m, n, &generator);

  // Setup CuBLAS specific data structures.
  cublasHandle_t handle;
  CUBLAS_CALL(cublasCreate(&handle));
  CUBLAS_CALL(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  float alpha = 1.0, beta = 0.0;
  cudaDataType_t data_type = CUDA_R_32F;
  cudaDataType_t compute_type = CUDA_R_32F;
  cublasGemmAlgo_t gemm_algorithm = CUBLAS_GEMM_DEFAULT;

  int batch_size = 10;
  while (state.KeepRunningBatch(batch_size)) {
    for (int i = 0; i < batch_size; ++i) {
      // Run the cublas kernel.
      CUBLAS_CALL(cublasGemmEx(
          handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, lhs_gpu.Values(),
          data_type, m, rhs_gpu.Values(), data_type, k, &beta,
          output_gpu.Values(), data_type, m, compute_type, gemm_algorithm));
    }
    CUDA_CALL(cudaStreamSynchronize(0));
  }
  ReportGemmThroughput(state);
}

BENCHMARK(BM_CublasGemm_ColumnMajor)->Apply(GemmBenchmarkArgs)->UseRealTime();

cublasStatus_t RowMajorGemm(cublasHandle_t handle, bool trans_a,
                            const CudaMatrix<float>& a, bool trans_b,
                            const CudaMatrix<float>& b, CudaMatrix<float>* c) {
  int m = trans_b ? b.Rows() : b.Columns();
  int k = trans_b ? b.Columns() : b.Rows();
  int n = trans_a ? a.Columns() : a.Rows();

  int ldb = trans_b ? k : m;
  int lda = trans_a ? n : k;
  cublasOperation_t transpose_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transpose_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;

  float alpha = 1.0, beta = 0.0;
  return cublasGemmEx(handle, transpose_b, transpose_a, m, n, k, &alpha,
                      b.Values(), CUDA_R_32F, ldb, a.Values(), CUDA_R_32F, lda,
                      &beta, c->Values(), CUDA_R_32F, c->Columns(), CUDA_R_32F,
                      CUBLAS_GEMM_DEFAULT);
}

void BM_CublasGemm_RowMajor(benchmark::State& state) {
  int m = state.range(0);
  int k = state.range(1);
  int n = state.range(2);

  // Create the lhs, rhs, and output matrices on gpu.
  absl::BitGen generator;
  CudaMatrix<float> lhs_gpu(m, k, &generator);
  CudaMatrix<float> rhs_gpu(k, n, &generator);
  CudaMatrix<float> output_gpu(m, n, &generator);

  // Setup CuBLAS specific data structures.
  cublasHandle_t handle;
  CUBLAS_CALL(cublasCreate(&handle));
  CUBLAS_CALL(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

  int batch_size = 10;
  while (state.KeepRunningBatch(batch_size)) {
    for (int i = 0; i < batch_size; ++i) {
      // Run the cublas kernel.
      CUBLAS_CALL(
          RowMajorGemm(handle, false, lhs_gpu, false, rhs_gpu, &output_gpu));
    }
    CUDA_CALL(cudaStreamSynchronize(0));
  }
  ReportGemmThroughput(state);
}

BENCHMARK(BM_CublasGemm_RowMajor)->Apply(GemmBenchmarkArgs)->UseRealTime();

}  // namespace sputnik
