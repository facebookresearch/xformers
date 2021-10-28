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

#ifndef THIRD_PARTY_SPUTNIK_MATRIX_UTILS_H_
#define THIRD_PARTY_SPUTNIK_MATRIX_UTILS_H_

/**
 * @file @brief Utilities for creating sparse and dense matrices for
 * tests and benchmarks.
 */

#include <vector>

#include "sputnik/cuda_utils.h"
#include "sputnik/test_utils.h"
#include "sputnik/type_utils.h"
#include "absl/random/random.h"

namespace sputnik {

/**
 * @brief Type conversion utilities.
 */
template <typename In, typename Out>
cudaError_t Convert(const In *in, Out *out, int n);

/**
 * @brief Matrix creation utilities.
 */
template <typename ValueType, typename IndexType>
void MakeSparseMatrixRandomUniform(int rows, int columns, int nonzeros,
                                   ValueType *values, IndexType *row_offsets,
                                   IndexType *column_indices,
                                   absl::BitGen *generator,
                                   int row_padding = 4);

template <typename ValueType, typename IndexType>
void MakeSparseMatrixPerfectUniform(int rows, int columns, int nonzeros_per_row,
                                    ValueType *values, IndexType *row_offsets,
                                    IndexType *column_indices,
                                    absl::BitGen *generator);

/**
 * @brief Create a row swizzle that maps thread blocks to rows in order.
 */
void IdentityRowSwizzle(int rows, const int* row_offsets, int* row_indices);

/**
 * @brief Create a row swizzle that maps thread blocks to rows in order of
 * decreasing size.
 */
void SortedRowSwizzle(int rows, const int* row_offsets, int* row_indices);

/**
 * @brief Enumeration of different row swizzles we can create.
 */
enum Swizzle {
  // Do not reorder the rows at all.
  IDENTITY = 0,
  // Sort the rows by size s.t. we process larger rows first.
  SORTED = 1
};

/**
 * @brief Enumeration of different sparse matrices that we can create.
 */
enum ElementDistribution {
  // Randomly sample which weights to zero with uniform probability.
  // Rows can have different numbers of non-zero weights, but they
  // have the same expected number of non-zero weights.
  RANDOM_UNIFORM = 0,
  // Divide the weights evenly across the rows and then randomly sample
  // which weights should be sets to zero in each row. All rows have
  // exactly the same number of non-zero weights.
  PERFECT_UNIFORM = 1
};

// Prototype for CudaSparseMatrix s.t. we can reference in SparseMatrix API.
template <typename Value>
class CudaSparseMatrix;

/**
 * @brief Simple sparse-matrix class to for managing pointers and memory
 * allocation/deallocation.
 */
class SparseMatrix {
 public:
  /**
   * @brief Create a sparse matrix with the specified properties.
   *
   * @param rows The number of rows in the matrix.
   * @param columns The number of columns in the matrix.
   * @param nonzeros The number of nonzero values in the matrix.
   * @param weight_distribution The distribution of non-zero weights
   * across the rows of the matrix.
   * @param row_swizzle The type of row swizzle to apply. Defaults to
   * IDENTITY.
   * @param pad_to_rows Each row in the sparse matrix will be padded to a
   * multiple of this value. Defaults to 4, which enables the user of
   * 4-element vector loads and stores. For best performance, pad to
   * `kBlockItemsK`.
   */
  SparseMatrix(int rows, int columns, int nonzeros,
               ElementDistribution weight_distribution, absl::BitGen* generator,
               Swizzle row_swizzle = IDENTITY, int pad_rows_to = 4);

  /**
   * @brief Construct a sparse matrix from a CUDA sparse matrix.
   */
  explicit SparseMatrix(const CudaSparseMatrix<float>& sparse_matrix);

  /**
   * @brief Construct a sparse matrix with the specified sparsity pattern.
   */
  SparseMatrix(int rows, int columns, int nonzeros,
               const std::vector<int>& row_offsets,
               const std::vector<int>& column_indices, absl::BitGen* generator,
               Swizzle row_swizzle = IDENTITY, int pad_rows_to = 4);

  /**
   * @brief Cleanup the underlying storage.
   */
  ~SparseMatrix() {
    delete[] values_;
    delete[] row_offsets_;
    delete[] column_indices_;
    delete[] row_indices_;
  }

  SparseMatrix(const SparseMatrix&) = delete;
  SparseMatrix& operator=(const SparseMatrix&) = delete;
  SparseMatrix(SparseMatrix&&) = delete;
  SparseMatrix& operator=(SparseMatrix&&) = delete;

  const float* Values() const { return values_; }
  float* Values() { return values_; }

  const int* RowOffsets() const { return row_offsets_; }
  int* RowOffsets() { return row_offsets_; }

  const int* ColumnIndices() const { return column_indices_; }
  int* ColumnIndices() { return column_indices_; }

  const int* RowIndices() const { return row_indices_; }
  int* RowIndices() { return row_indices_; }

  int Rows() const { return rows_; }

  int Columns() const { return columns_; }

  int Nonzeros() const { return nonzeros_; }

  int PadRowsTo() const { return pad_rows_to_; }

  int NumElementsWithPadding() const { return num_elements_with_padding_; }

  ElementDistribution WeightDistribution() const {
    return weight_distribution_;
  }

  Swizzle RowSwizzle() const { return row_swizzle_; }

 protected:
  SparseMatrix() : values_(nullptr),
                   row_offsets_(nullptr),
                   column_indices_(nullptr),
                   row_indices_(nullptr),
                   rows_(0),
                   columns_(0),
                   nonzeros_(0),
                   pad_rows_to_(0),
                   num_elements_with_padding_(0),
                   weight_distribution_(RANDOM_UNIFORM),
                   row_swizzle_(IDENTITY) {}

  // Matrix value and index storage.
  float* values_;
  int* row_offsets_;
  int* column_indices_;

  // Swizzled row indices for load balancing.
  int* row_indices_;

  // Matrix meta-data.
  int rows_, columns_, nonzeros_;
  int pad_rows_to_, num_elements_with_padding_;
  ElementDistribution weight_distribution_;
  Swizzle row_swizzle_;

  void InitFromCudaSparseMatrix(const CudaSparseMatrix<float>& sparse_matrix);
};

/**
 * @brief Simple gpu sparse-matrix class to for managing pointers and
 * memory allocation/deallocation.
 */
template <typename Value>
class CudaSparseMatrix {
 public:
  /**
   * @brief Create a sparse matrix with the specified properties.
   *
   * @param row The number of rows in the matrix.
   * @param columns The number of columns in the matrix.
   * @param nonzeros The number of nonzero values in the matrix.
   * @param weight_distribution The distribution of non-zero weights
   * across the rows of the matrix.
   * @param row_swizzle The type of row swizzle to apply. Defaults to
   * IDENTITY.
   * @param pad_to_rows Each row in the sparse matrix will be padded to a
   * multiple of this value. Defaults to 4, which enables the user of
   * 4-element vector loads and stores. For best performance, pad to
   * `kBlockItemsK`.
   */
  CudaSparseMatrix(int rows_, int columns_, int nonzeros_,
                   ElementDistribution weight_distribution_,
                   absl::BitGen* generator, Swizzle row_swizzle = IDENTITY,
                   int pad_rows_to_ = 4);

  /**
   * @brief Construct a CUDA sparse matrix from a host sparse matrix.
   */
  explicit CudaSparseMatrix(const SparseMatrix& sparse_matrix);

  /**
   * @brief Cleanup the underlying storage.
   */
  ~CudaSparseMatrix() {
    CUDA_CALL(cudaFree(values_));
    CUDA_CALL(cudaFree(row_offsets_));
    CUDA_CALL(cudaFree(column_indices_));
    CUDA_CALL(cudaFree(row_indices_));
  }

  CudaSparseMatrix(const CudaSparseMatrix&) = delete;
  CudaSparseMatrix& operator=(const CudaSparseMatrix&) = delete;
  CudaSparseMatrix(CudaSparseMatrix&&) = delete;
  CudaSparseMatrix& operator=(CudaSparseMatrix&&) = delete;

  // Datatype for indices in this matrix.
  typedef typename Value2Index<Value>::Index Index;

  const Value* Values() const { return values_; }
  Value* Values() { return values_; }

  const int* RowOffsets() const { return row_offsets_; }
  int* RowOffsets() { return row_offsets_; }

  const Index* ColumnIndices() const { return column_indices_; }
  Index* ColumnIndices() { return column_indices_; }

  const int* RowIndices() const { return row_indices_; }
  int* RowIndices() { return row_indices_; }

  int Rows() const { return rows_; }

  int Columns() const { return columns_; }

  int Nonzeros() const { return nonzeros_; }

  int PadRowsTo() const { return pad_rows_to_; }

  int NumElementsWithPadding() const { return num_elements_with_padding_; }

  ElementDistribution WeightDistribution() const {
    return weight_distribution_;
  }

  Swizzle RowSwizzle() const { return row_swizzle_; }

 protected:
  CudaSparseMatrix() : values_(nullptr),
                       row_offsets_(nullptr),
                       column_indices_(nullptr),
                       row_indices_(nullptr),
                       rows_(0),
                       columns_(0),
                       nonzeros_(0),
                       pad_rows_to_(0),
                       num_elements_with_padding_(0),
                       weight_distribution_(RANDOM_UNIFORM),
                       row_swizzle_(IDENTITY) {}

  // Matrix value and index storage.
  Value* values_;
  int* row_offsets_;
  Index* column_indices_;

  // Swizzled row indices for load balancing.
  int* row_indices_;

  // Matrix meta-data.
  int rows_, columns_, nonzeros_;
  int pad_rows_to_, num_elements_with_padding_;
  ElementDistribution weight_distribution_;
  Swizzle row_swizzle_;

  void InitFromSparseMatrix(const SparseMatrix& sparse_matrix);
};

// Prototype for CudaMatrix s.t. we can reference in Matrix API.
template <typename Value>
class CudaMatrix;

/**
 * @brief Simple dense matrix class for managing pointers and memory.
 */
class Matrix {
 public:
  /**
   * @brief Construct a dense matrix with random values.
   *
   * @param rows The number of rows in the matrix.
   * @param columns The number of columns in the matrix.
   */
  Matrix(int rows, int columns, absl::BitGen* generator);

  /**
   * @brief Construct a matrix from a CUDA matrix.
   */
  template <typename Value>
  explicit Matrix(const CudaMatrix<Value>& matrix) {
    InitFromCudaMatrix(matrix);
  }

  ~Matrix() { delete[] values_; }

  Matrix(const Matrix&) = delete;
  Matrix& operator=(const Matrix&) = delete;
  Matrix(Matrix&&) = delete;
  Matrix& operator=(Matrix&&) = delete;

  const float* Values() const { return values_; }
  float* Values() { return values_; }

  int Rows() const { return rows_; }

  int Columns() const { return columns_; }

 protected:
  // Matrix value storage.
  float* values_;

  // Matrix meta-data.
  int rows_, columns_;

  template <typename Value>
  void InitFromCudaMatrix(const CudaMatrix<Value>& matrix);
};

/**
 * @brief Simple dense gpu matrix class for managing pointers and memory.
 */
template <typename Value>
class CudaMatrix {
 public:
  /**
   * @brief Construct a dense matrix on gpu with random values.
   *
   * @param rows The number of rows in the matrix.
   * @param columns The number of columns in the matrix.
   */
  CudaMatrix(int rows, int columns, absl::BitGen* generator);

  /**
   * @brief Construct a dense matrix on gpu from one on the host.
   */
  explicit CudaMatrix(const Matrix& matrix);

  ~CudaMatrix() { CUDA_CALL(cudaFree(values_)); }

  CudaMatrix(const CudaMatrix&) = delete;
  CudaMatrix& operator=(const CudaMatrix&) = delete;
  CudaMatrix(CudaMatrix&&) = delete;
  CudaMatrix& operator=(CudaMatrix&&) = delete;

  const Value* Values() const { return values_; }
  Value* Values() { return values_; }

  int Rows() const { return rows_; }

  int Columns() const { return columns_; }

 protected:
  // Matrix value storage.
  Value* values_;

  // Matrix meta-data.
  int rows_, columns_;

  void InitFromMatrix(const Matrix& matrix);
};

/**
 * @brief Helper to load sparse matrix values into a std::vector.
 */
inline std::vector<float> ToVector(const SparseMatrix& sparse_matrix) {
  int num = sparse_matrix.NumElementsWithPadding();
  std::vector<float> out(sparse_matrix.Values(), sparse_matrix.Values() + num);
  return out;
}

/**
 * @brief Helper to load matrix values into a std::vector.
 */
inline std::vector<float> ToVector(const Matrix& matrix) {
  int num = matrix.Rows() * matrix.Columns();
  std::vector<float> out(matrix.Values(), matrix.Values() + num);
  return out;
}

}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_MATRIX_UTILS_H_
