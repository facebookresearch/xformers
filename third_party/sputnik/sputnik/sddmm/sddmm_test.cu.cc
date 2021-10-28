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

#include "sputnik/cuda_utils.h"
#include "sputnik/matrix_utils.h"
#include "sputnik/sddmm/cuda_sddmm.h"
#include "sputnik/test_utils.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/random/random.h"

namespace sputnik {

using ::testing::NanSensitiveFloatNear;
using ::testing::Pointwise;

/**
 * @brief Defines the properties of a problem for testing.
 */
template <int kDimM_, int kDimK_, int kDimN_, int kNonZeros_>
struct Problem {
  static_assert(kNonZeros_ <= kDimM_ * kDimN_,
                "The number of nonzeros must fit into the matrix.");

  static const int kDimM = kDimM_;
  static const int kDimK = kDimK_;
  static const int kDimN = kDimN_;
  static const int kNonZeros = kNonZeros_;
};

template <typename Problem>
class SddmmTest : public ::testing::Test {
 public:
  const int kDimM_ = Problem::kDimM;
  const int kDimK_ = Problem::kDimK;
  const int kDimN_ = Problem::kDimN;
  const int kNonZeros_ = Problem::kNonZeros;

  // Random number generator for creating matrices.
  absl::BitGen generator_;

  /**
   * @brief Basic sampled matrix multiplicatoin routine for testing.
   */
  void Sddmm(int m, int k, int n, const int* row_offsets,
             const int* column_indices, const float* lhs_matrix,
             const float* rhs_matrix, float* output_values) {
    for (int i = 0; i < m; ++i) {
      for (int j = row_offsets[i]; j < row_offsets[i + 1]; ++j) {
        int idx_n = column_indices[j];
        double accumulator = 0.0;
        for (int l = 0; l < k; ++l) {
          accumulator += static_cast<double>(lhs_matrix[i * k + l]) *
                         static_cast<double>(rhs_matrix[idx_n * k + l]);
        }
        output_values[j] = static_cast<float>(accumulator);
      }
    }
  }
};

typedef ::testing::Types<
    Problem<1, 32, 32, 32>,            // Basic functionality
    Problem<1, 64, 32, 32>,            // Issues with multiple k-dim tiles
    Problem<1, 32, 64, 64>,            // Issues with multiple thread blocks
    Problem<1, 32, 64, 32>,            // 50% sparsity
    Problem<1, 32, 64, 16>,            // 75% sparsity
    Problem<2, 32, 32, 64>,            // multiple rows
    Problem<1024, 80, 1024, 131072>,   // 1k rnn bwd wrt w, bs 80
    Problem<1, 28, 32, 32>,            // non-tile-size divisible
    Problem<1024, 128, 1024, 131072>,  // 1k rnn bwd wrt w, bs 128
    Problem<1024, 256, 1024, 131072>,  // 1k rnn bwd wrt w, bs 256
    Problem<1024, 512, 1024, 131072>,  // 1k rnn bwd wrt w, bs 512
    Problem<2048, 128, 2048, 1024 * 1024>,          // 2k rnn bwd wrt w, bs 128
    Problem<2048 * 3, 128, 2048, 1024 * 1024 * 3>,  // 2k gru bwd wrt w, bs 128
    Problem<2048 * 4, 128, 2048, 1024 * 1024 * 4>>   // 2k lstm bwd wrt w, bs 128
    TestProblems;

TYPED_TEST_SUITE(SddmmTest, TestProblems);

typedef std::function<cudaError_t(
    int,
    int,
    int,
    int,
    const int*,
    const int*,
    const int*,
    const float*,
    const float*,
    float*,
    cudaStream_t)>
    SddmmFn;

template <typename Problem>
void TestFn(SddmmFn sddmm_fn, SddmmTest<Problem>* obj) {
  // No padding required for correctness.
  const int kRowPadding = 0;

  // Create the sparse matrix on cpu & gpu.
  SparseMatrix output_matrix(obj->kDimM_, obj->kDimN_, obj->kNonZeros_,
                             RANDOM_UNIFORM, &obj->generator_, SORTED,
                             kRowPadding);
  CudaSparseMatrix<float> output_matrix_gpu(output_matrix);

  // Create the dense matrices on cpu & gpu
  Matrix lhs_matrix(obj->kDimM_, obj->kDimK_, &obj->generator_);
  Matrix rhs_matrix(obj->kDimN_, obj->kDimK_, &obj->generator_);
  CudaMatrix<float> lhs_matrix_gpu(lhs_matrix), rhs_matrix_gpu(rhs_matrix);

  // Run the GPU kernel.
  CUDA_CALL(sddmm_fn(
      output_matrix_gpu.Rows(), lhs_matrix_gpu.Columns(),
      output_matrix_gpu.Columns(), output_matrix_gpu.NumElementsWithPadding(),
      output_matrix_gpu.RowIndices(), output_matrix_gpu.RowOffsets(),
      output_matrix_gpu.ColumnIndices(), lhs_matrix_gpu.Values(),
      rhs_matrix_gpu.Values(), output_matrix_gpu.Values(), 0));
  CUDA_CALL(cudaStreamSynchronize(nullptr));

  obj->Sddmm(obj->kDimM_, obj->kDimK_, obj->kDimN_, output_matrix.RowOffsets(),
             output_matrix.ColumnIndices(), lhs_matrix.Values(),
             rhs_matrix.Values(), output_matrix.Values());

  // Copy the results to the host and compare with the ground truth.
  SparseMatrix results(output_matrix_gpu);
  auto comparator =
      Pointwise(NanSensitiveFloatNear(1e-04), ToVector(output_matrix));
  ASSERT_THAT(ToVector(results), comparator);
}

#define CONCAT_(x, y) x##y
#define CONCAT(x, y) CONCAT_(x, y)
#define ANONYMOUS_NAME(x) CONCAT(x, __COUNTER__)

#define REGISTER_TESTCASE(name, fn) \
  TYPED_TEST(SddmmTest, name) { TestFn(fn, this); }

#define REGISTER_TILED_TESTCASE_HELPER(name, tname, fn, ltype, mt, kt, nt, bs) \
  const auto& tname = fn<ltype, mt, kt, nt, bs>;                               \
  REGISTER_TESTCASE(name##_##ltype##x##mt##x##kt##x##nt##x##bs, tname)

#define REGISTER_TILED_TESTCASE(name, fn, ltype, mt, kt, nt, bs)            \
  REGISTER_TILED_TESTCASE_HELPER(name, ANONYMOUS_NAME(sddmm_fn), fn, ltype, \
                                 mt, kt, nt, bs)

REGISTER_TILED_TESTCASE(CudaSddmmEx, CudaSddmmEx, float, 1, 32, 32, 32);
REGISTER_TILED_TESTCASE(CudaSddmmEx, CudaSddmmEx, float2, 2, 32, 32, 16);
REGISTER_TILED_TESTCASE(CudaSddmmEx, CudaSddmmEx, float4, 4, 32, 32, 8);

}  // namespace sputnik
