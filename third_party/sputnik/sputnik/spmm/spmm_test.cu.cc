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
#include <functional>

#include "sputnik/cuda_utils.h"
#include "sputnik/matrix_utils.h"
#include "sputnik/spmm/cuda_spmm.h"
#include "sputnik/spmm/spmm_config.h"
#include "sputnik/test_utils.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/random/random.h"

namespace sputnik {

using ::testing::NanSensitiveFloatNear;
using ::testing::Pointwise;

template <int kDimM_, int kDimK_, int kDimN_, int kNonZeros_>
struct Problem {
  static_assert(kNonZeros_ <= kDimM_ * kDimK_,
                "Number of non-zero must fit in the matrix.");

  static const int kDimM = kDimM_;
  static const int kDimK = kDimK_;
  static const int kDimN = kDimN_;
  static const int kNonZeros = kNonZeros_;
};

template <typename Problem>
class SpmmTest : public ::testing::Test {
 public:
  const int kDimM = Problem::kDimM;
  const int kDimK = Problem::kDimK;
  const int kDimN = Problem::kDimN;
  const int kNonZeros = Problem::kNonZeros;

  // Random number generator for creating matrices.
  absl::BitGen generator_;

  /**
   * @brief Basic matrix-multiplication routine for testing.
   */
  void Spmm(int m, int k, int n, const float *a_values,
            const int *a_row_offsets, const int *a_column_indices,
            const float *b, float *c) {
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        double accum = 0.0;
        for (int l = a_row_offsets[i]; l < a_row_offsets[i + 1]; ++l) {
          float a_val = a_values[l];
          int a_col = a_column_indices[l];
          accum += static_cast<double>(a_val) *
                   static_cast<double>(b[a_col * n + j]);
        }
        c[i * n + j] = static_cast<float>(accum);
      }
    }
  }
};

// NOTE: To use the float4/float2 variants of the kernels the N dim
// must be a multiple of 4/2 respectively. To use the half8/half4
// variants of the kernel the N dim must be a multiple of 8/4
// respectively. All of our test cases meet these criteria.
typedef ::testing::Types<
    Problem<4, 32, 32, 32>,
    Problem<4, 32, 48, 128>,
    Problem<7, 96, 40, 333>,
    Problem<11, 55, 8, 293>,
    Problem<4, 128, 128, 512>,
    Problem<64, 512, 512, 16384>,
    /* Some actualy problem sizes that we benchmark */
    Problem<1024, 1024, 512, 131072>,
    Problem<1024, 1024, 256, 131072>,
    Problem<1024, 1024, 128, 131072>,
    /* Some more problems we commonly benchmark */
    Problem<2048, 2048, 128, 1024 * 1024>,
    Problem<2048 * 3, 2048, 128, 1024 * 1024 * 3>,
    Problem<2048 * 4, 2048, 128, 1024 * 1024 * 4>>
    TestProblems;

TYPED_TEST_SUITE(SpmmTest, TestProblems);

typedef std::function<cudaError_t(
    int,            // m: number of rows in lhs & output.
    int,            // k: number of cols in lhs and rows in rhs.
    int,            // n: number of cols in rhs/output.
    int,            // nonzeros: number of nonzero values in lhs.
    const int *,    // row_indices: ptr to row index swizzle map.
    const float *,  // values: ptr to lhs values.
    const int *,    // row_offsets: ptr to lhs row offsets.
    const int *,    // column_indices: ptr to lhs column indices.
    const float *,  // dense_matrix: ptr to rhs matrix.
    const float *,  // bias: ptr to bias.
    float *,        // output_matrix: ptr to output matrix.
    cudaStream_t)>  // stream: stream to execute in.
    FloatSpmmFn;

template <typename Problem>
void TestFn(FloatSpmmFn spmm_fn, SpmmTest<Problem> *obj) {
  // With the addition of the reverse offset memory alignment trick we
  // no longer need to pad rows of the sparse matrix to use vector memory
  // instruction.
  const int kRowPadding = 0;

  // Create the sparse matrix on cpu & gpu.
  SparseMatrix sparse_matrix(obj->kDimM, obj->kDimK, obj->kNonZeros,
                             RANDOM_UNIFORM, &obj->generator_, SORTED,
                             kRowPadding);
  CudaSparseMatrix<float> sparse_matrix_gpu(sparse_matrix);

  // Create the dense matrix on cpu & gpu
  Matrix matrix(obj->kDimK, obj->kDimN, &obj->generator_);
  CudaMatrix<float> matrix_gpu(matrix);

  // Create the output matrix on gpu & gpu.
  Matrix output_matrix(obj->kDimM, obj->kDimN, &obj->generator_);
  CudaMatrix<float> output_matrix_gpu(output_matrix);

  // Run the gpu kernel.
  CUDA_CALL(spmm_fn(obj->kDimM, obj->kDimK, obj->kDimN,
                    sparse_matrix_gpu.NumElementsWithPadding(),
                    sparse_matrix_gpu.RowIndices(), sparse_matrix_gpu.Values(),
                    sparse_matrix_gpu.RowOffsets(),
                    sparse_matrix_gpu.ColumnIndices(), matrix_gpu.Values(),
                    /* bias = */ nullptr, output_matrix_gpu.Values(), 0));
  CUDA_CALL(cudaStreamSynchronize(nullptr));

  obj->Spmm(obj->kDimM, obj->kDimK, obj->kDimN, sparse_matrix.Values(),
            sparse_matrix.RowOffsets(), sparse_matrix.ColumnIndices(),
            matrix.Values(), output_matrix.Values());

  // Copy the results to the host and compare with the ground truth.
  Matrix results(output_matrix_gpu);
  auto comparator =
      Pointwise(NanSensitiveFloatNear(1e-04), ToVector(output_matrix));
  ASSERT_THAT(ToVector(results), comparator);
}

typedef std::function<cudaError_t(
    int,             // m: number of rows in lhs & output.
    int,             // k: number of cols in lhs and rows in rhs.
    int,             // n: number of cols in rhs/output.
    int,             // nonzeros: number of nonzero values in lhs.
    const int *,     // row_indices: ptr to row index swizzle map.
    const half2 *,   // values: ptr to lhs values.
    const int *,     // row_offsets: ptr to lhs row offsets.
    const short2 *,  // column_indices: ptr to lhs column indices.
    const half2 *,   // dense_matrix: ptr to rhs matrix.
    const float *,   // bias: ptr to bias.
    half2 *,         // output_matrix: ptr to output matrix.
    cudaStream_t)>   // stream: stream to execute in.
    HalfSpmmFn;

template <typename Problem>
void HalfTestFn(HalfSpmmFn spmm_fn, SpmmTest<Problem> *obj) {
  // Pad to the nearest 2 nonzero elements so we can use the `half2`
  // datatype.
  const int kRowPadding = 2;

  // Create the sparse matrix on cpu & gpu.
  SparseMatrix sparse_matrix(obj->kDimM, obj->kDimK, obj->kNonZeros,
                             RANDOM_UNIFORM, &obj->generator_, SORTED,
                             kRowPadding);
  CudaSparseMatrix<half2> sparse_matrix_gpu(sparse_matrix);

  // Create the dense matrix on cpu & gpu
  Matrix matrix(obj->kDimK, obj->kDimN, &obj->generator_);
  CudaMatrix<half2> matrix_gpu(matrix);

  // Create the output matrix on gpu & gpu.
  Matrix output_matrix(obj->kDimM, obj->kDimN, &obj->generator_);
  CudaMatrix<half2> output_matrix_gpu(output_matrix);

  // Run the gpu kernel.
  CUDA_CALL(spmm_fn(obj->kDimM, obj->kDimK, obj->kDimN,
                    sparse_matrix_gpu.NumElementsWithPadding(),
                    sparse_matrix_gpu.RowIndices(), sparse_matrix_gpu.Values(),
                    sparse_matrix_gpu.RowOffsets(),
                    sparse_matrix_gpu.ColumnIndices(), matrix_gpu.Values(),
                    /* bias = */ nullptr, output_matrix_gpu.Values(), 0));
  CUDA_CALL(cudaStreamSynchronize(nullptr));

  obj->Spmm(obj->kDimM, obj->kDimK, obj->kDimN, sparse_matrix.Values(),
            sparse_matrix.RowOffsets(), sparse_matrix.ColumnIndices(),
            matrix.Values(), output_matrix.Values());

  // Copy the results to the host and compare with the ground truth.
  Matrix results(output_matrix_gpu);

  auto comparator =
      Pointwise(NanSensitiveFloatNear(5e-02), ToVector(output_matrix));
  ASSERT_THAT(ToVector(results), comparator);
}

#define CONCAT_(x, y) x##y
#define CONCAT(x, y) CONCAT_(x, y)
#define ANONYMOUS_NAME(x) CONCAT(x, __COUNTER__)

#define REGISTER_TESTCASE(name, fn) \
  TYPED_TEST(SpmmTest, name) { TestFn(fn, this); }

#define REGISTER_FLOAT_TESTCASE_HELPER(name, tname, fn, stype, dtype, mt, kt, \
                                       nt, bs)                                \
  const auto &tname = fn<SpmmConfig<float, stype, dtype, mt, kt, nt, bs>>;    \
  REGISTER_TESTCASE(name##_##stype##x##dtype##x##mt##x##kt##x##nt##x##bs, tname)

#define REGISTER_FLOAT_TESTCASE(name, fn, stype, dtype, mt, kt, nt, bs)    \
  REGISTER_FLOAT_TESTCASE_HELPER(name, ANONYMOUS_NAME(spmm_fn), fn, stype, \
                                 dtype, mt, kt, nt, bs)

/* 1-d tiling with blocksize 64 */
REGISTER_FLOAT_TESTCASE(CudaSpmmEx, CudaSpmmEx, float, float, 1, 32, 64, 32);

/* 2-d tiling with blocksize 64 and vector loads */
REGISTER_FLOAT_TESTCASE(CudaSpmmEx, CudaSpmmEx, float2, float2, 2, 32, 64, 16);
REGISTER_FLOAT_TESTCASE(CudaSpmmEx, CudaSpmmEx, float4, float4, 4, 32, 64, 8);

/* 1-d tilings with blocksize 32 */
REGISTER_FLOAT_TESTCASE(CudaSpmmEx, CudaSpmmEx, float, float, 1, 32, 32, 32);

/* 2-d tilings with 32 n-dim and vector loads */
REGISTER_FLOAT_TESTCASE(CudaSpmmEx, CudaSpmmEx, float2, float2, 2, 32, 32, 16);
REGISTER_FLOAT_TESTCASE(CudaSpmmEx, CudaSpmmEx, float4, float4, 4, 32, 32, 8);

REGISTER_FLOAT_TESTCASE(CudaSpmmEx, CudaSpmmEx, float2, float4, 4, 16, 32, 8);
REGISTER_FLOAT_TESTCASE(CudaSpmmEx, CudaSpmmEx, float, float4, 4, 8, 32, 8);
REGISTER_FLOAT_TESTCASE(CudaSpmmEx, CudaSpmmEx, float, float2, 2, 16, 32, 16);

REGISTER_FLOAT_TESTCASE(CudaSpmmEx, CudaSpmmEx, float2, float4, 2, 32, 64, 16);
REGISTER_FLOAT_TESTCASE(CudaSpmmEx, CudaSpmmEx, float, float4, 1, 32, 128, 32);
REGISTER_FLOAT_TESTCASE(CudaSpmmEx, CudaSpmmEx, float, float2, 1, 32, 64, 32);

REGISTER_FLOAT_TESTCASE(CudaSpmmEx, CudaSpmmEx, float4, float4, 2, 64, 64, 16);
REGISTER_FLOAT_TESTCASE(CudaSpmmEx, CudaSpmmEx, float2, float4, 1, 64, 128, 32);
REGISTER_FLOAT_TESTCASE(CudaSpmmEx, CudaSpmmEx, float2, float2, 1, 64, 64, 32);

/* 2-d tilings with 16 n-dim and vector loads */
REGISTER_FLOAT_TESTCASE(CudaSpmmEx, CudaSpmmEx, float2, float2, 4, 32, 16, 8);
REGISTER_FLOAT_TESTCASE(CudaSpmmEx, CudaSpmmEx, float2, float, 2, 32, 16, 16);
REGISTER_FLOAT_TESTCASE(CudaSpmmEx, CudaSpmmEx, float4, float4, 8, 32, 16, 4);
REGISTER_FLOAT_TESTCASE(CudaSpmmEx, CudaSpmmEx, float4, float2, 4, 32, 16, 8);

/* 2-d tilings with 8 n-dim and vector loads */
REGISTER_FLOAT_TESTCASE(CudaSpmmEx, CudaSpmmEx, float2, float2, 8, 32, 8, 4);
REGISTER_FLOAT_TESTCASE(CudaSpmmEx, CudaSpmmEx, float4, float4, 16, 32, 8, 2);
REGISTER_FLOAT_TESTCASE(CudaSpmmEx, CudaSpmmEx, float4, float2, 8, 32, 8, 4);
REGISTER_FLOAT_TESTCASE(CudaSpmmEx, CudaSpmmEx, float4, float, 4, 32, 8, 8);

#undef REGISTER_FLOAT_TESTCASE
#undef REGISTER_FLOAT_TESTCASE_HELPER
#undef REGISTER_TESTCASE

#define REGISTER_TESTCASE(name, fn) \
  TYPED_TEST(SpmmTest, name) { HalfTestFn(fn, this); }

#define REGISTER_HALF_TESTCASE_HELPER(name, tname, fn, stype, dtype, mt, kt, \
                                      nt, bs)                                \
  const auto &tname = fn<SpmmConfig<half2, stype, dtype, mt, kt, nt, bs>>;   \
  REGISTER_TESTCASE(name##_##stype##x##dtype##x##mt##x##kt##x##nt##x##bs, tname)

#define REGISTER_HALF_TESTCASE(name, fn, stype, dtype, mt, kt, nt, bs)    \
  REGISTER_HALF_TESTCASE_HELPER(name, ANONYMOUS_NAME(spmm_fn), fn, stype, \
                                dtype, mt, kt, nt, bs)

/* 1-d tiling with blocksize 64 */
REGISTER_HALF_TESTCASE(CudaSpmmEx, CudaSpmmEx, half2, half2, 1, 32, 64, 32);

/* 2-d tiling with blocksize 64 and vector loads */
REGISTER_HALF_TESTCASE(CudaSpmmEx, CudaSpmmEx, half4, half4, 2, 32, 64, 16);
REGISTER_HALF_TESTCASE(CudaSpmmEx, CudaSpmmEx, half8, half8, 4, 32, 64, 8);

/* 1-d tilings with blocksize 32 */
REGISTER_HALF_TESTCASE(CudaSpmmEx, CudaSpmmEx, half2, half2, 1, 32, 32, 32);

// /* 2-d tilings with 32 n-dim and vector loads */
REGISTER_HALF_TESTCASE(CudaSpmmEx, CudaSpmmEx, half4, half4, 2, 32, 32, 16);
REGISTER_HALF_TESTCASE(CudaSpmmEx, CudaSpmmEx, half8, half8, 4, 32, 32, 8);

/* 2-d tilings with 16 n-dim and vector loads */
REGISTER_HALF_TESTCASE(CudaSpmmEx, CudaSpmmEx, half4, half4, 4, 32, 16, 8);
REGISTER_HALF_TESTCASE(CudaSpmmEx, CudaSpmmEx, half4, half2, 2, 32, 16, 16);
REGISTER_HALF_TESTCASE(CudaSpmmEx, CudaSpmmEx, half8, half8, 8, 32, 16, 4);
REGISTER_HALF_TESTCASE(CudaSpmmEx, CudaSpmmEx, half8, half4, 4, 32, 16, 8);

/* 2-d tilings with 8 n-dim and vector loads */
REGISTER_HALF_TESTCASE(CudaSpmmEx, CudaSpmmEx, half4, half4, 8, 32, 8, 4);
REGISTER_HALF_TESTCASE(CudaSpmmEx, CudaSpmmEx, half8, half8, 16, 32, 8, 2);
REGISTER_HALF_TESTCASE(CudaSpmmEx, CudaSpmmEx, half8, half4, 8, 32, 8, 4);
REGISTER_HALF_TESTCASE(CudaSpmmEx, CudaSpmmEx, half8, half2, 4, 32, 8, 8);

#undef ANONYMOUS_NAME
#undef CONCAT
#undef CONCAT_

}  // namespace sputnik
