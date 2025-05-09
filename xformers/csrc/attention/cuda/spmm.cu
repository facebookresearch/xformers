// This file was modified from sputnik to implement batch support for
// spmm directly in the kernels
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
#include <string>
#include <unordered_map>

#include "sputnik/barrier.h"
#include "sputnik/cuda_utils.h"
#include "sputnik/load_store.h"
#include "sputnik/memory_aligner.h"
#include "sputnik/spmm/compute_utils.h"
#include "sputnik/spmm/cuda_spmm.h"
#include "sputnik/spmm/dense_tile.h"
#include "sputnik/spmm/output_tile.h"
#include "sputnik/spmm/predicate_utils.h"
#include "sputnik/spmm/sparse_tile.h"
#include "sputnik/spmm/spmm_config.h"
#include "sputnik/tiling_utils.h"
#include "sputnik/vector_utils.h"

#include <ATen/ATen.h>
#include <torch/types.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace sputnik {

template <typename Config>
cudaError_t CudaSpmmEx2(
    int m,
    int k,
    int n,
    int nonzeros,
    const int* __restrict__ row_indices,
    const typename Config::ScalarValue* __restrict__ values,
    const int* __restrict__ row_offsets,
    const typename Config::ScalarIndex* __restrict__ column_indices,
    const typename Config::ScalarValue* __restrict__ dense_matrix,
    const float* __restrict__ bias,
    typename Config::ScalarValue* __restrict__ output_matrix,
    cudaStream_t stream,
    int batch_size);

namespace {

template <typename Config>
struct SpmmKernel2 {
  //
  /// Shortcuts for commonly used specialized types.
  //

  typedef TilingUtils<
      Config::kBlockItemsY,
      Config::kBlockItemsK,
      Config::kBlockItemsX>
      Tiling;

  typedef PredicateVector<Config::kThreadItemsX> PredicateVector;

  typedef PredicatesN<
      typename Config::DenseValue,
      Config::kBlockItemsX,
      Config::kBlockWidth>
      PredicatesN;

  typedef MemoryAligner<typename Config::SparseValue, Config::kBlockWidth>
      MemoryAligner;

  typedef SparseTile<
      typename Config::SparseValue,
      Config::kBlockItemsK,
      Config::kBlockWidth>
      SparseTile;

  typedef DenseTile<
      typename Config::DenseValue,
      Config::kBlockItemsK,
      Config::kBlockItemsX,
      Config::kBlockWidth,
      Config::kResidueUnroll>
      DenseTile;

  typedef ComputeUtils<
      typename Config::DenseValue,
      Config::kBlockItemsK,
      Config::kBlockItemsX,
      Config::kBlockWidth>
      Computer;

  typedef Barrier<Config::kBlockItemsY, Config::kBlockWidth> Barrier;

  typedef OutputTile<
      typename Config::DenseValue,
      Config::kBlockItemsX,
      Config::kBlockWidth>
      OutputTile;

  typedef typename Config::ScalarValue ScalarValue;
  typedef typename Config::DenseValue DenseValue;
  typedef typename Config::SparseValue SparseValue;
  typedef typename Config::ScalarIndex ScalarIndex;
  typedef typename Config::Index Index;

  /**
   * @brief Main function for SpMM kernel.
   */
  static __device__ __forceinline__ void KernelFn(
      int m,
      int k,
      int n,
      const int* __restrict__ row_indices,
      const ScalarValue* __restrict__ values,
      const int* __restrict__ row_offsets,
      const ScalarIndex* __restrict__ column_indices,
      const ScalarValue* __restrict__ dense_matrix,
      const float* __restrict__ bias,
      ScalarValue* __restrict__ out,
      int nnz) {
    // Calculate this thread block's indices into the M and N dimensions.
    int m_index = Tiling::IndexM(), n_index = Tiling::IndexN();

    // Threads that work on different m-dim indices are independent. If
    // we're out of bounds in the m-dimension we can just return.
    if (m_index >= m)
      return;
    m_index = Load(row_indices + m_index);

    // Divide some of our constant problem dimensions and indices by
    // the number of elements that are packed into each scalar.
    n /= Config::kElementsPerScalar;

    // Initialize the n-dimension predicates for this thread.
    PredicateVector predicates_n;
    if (Config::kPredicateLoads) {
      PredicatesN::Set(n_index, n, &predicates_n);
    }

    // Load the row offset and calculate the number of non-zeros in the row.
    int row_offset = Load(row_offsets + m_index);
    int nonzeros = Load(row_offsets + m_index + 1) - row_offset;

    // Divide some of our constant values by the number of elements that
    // are packed into a single scalar.
    nonzeros /= Config::kElementsPerScalar;
    row_offset /= Config::kElementsPerScalar;

    // Possibly align the row offset s.t. it's safe to use vector memory ops.
    //
    // Note that if we only have residue to process, we do not align the row
    // offset. This lets us not worry about masking in the residue handling,
    // where we use scalar memory operations anyways.
    MemoryAligner memory_aligner(row_offset, nonzeros);
    int aligned_nonzeros = memory_aligner.AlignedNonzeros();
    if (aligned_nonzeros >= Config::kBlockItemsK) {
      nonzeros = aligned_nonzeros;
      row_offset = memory_aligner.AlignedRowOffset();
    }

    // Shared memory tiles for the lhs values and indices.
    constexpr int kTileSize = Config::kBlockItemsK * Config::kBlockItemsY;
    __shared__ ScalarValue values_tile_array[kTileSize];
    __shared__ ScalarIndex column_indices_tile_array[kTileSize];

    // Possibly increment our tile pointers for 2D tiling schemes.
    ScalarValue* values_tile = Tiling::MaybeOffset(
        values_tile_array, Config::kBlockItemsK * threadIdx.y);
    ScalarIndex* column_indices_tile = Tiling::MaybeOffset(
        column_indices_tile_array, Config::kBlockItemsK * threadIdx.y);

    // Create a loader for the sparse lhs matrix.
    SparseTile sparse_tile_loader(
        n,
        row_offset,
        threadIdx.x,
        values + blockIdx.z * nnz,
        column_indices,
        values_tile,
        column_indices_tile);

    // Register fragment for the dense_matrix values.
    constexpr int kDenseFragmentSize = Config::kElementsPerScalar *
        Config::kBlockItemsK * Config::kBlockItemsX / Config::kBlockWidth;
    __align__(16) ScalarValue dense_matrix_fragment[kDenseFragmentSize];

    // Create a loader for the dense dense_matrix matrix.
    DenseTile dense_tile_loader(
        n,
        n_index,
        threadIdx.x,
        dense_matrix + blockIdx.z * k * n,
        column_indices_tile,
        dense_matrix_fragment);

    // Accumulator registers for the output values. Initialize the
    // registers to zero s.t. we can always accumulate in-place.
    constexpr int kOutputFragmentSize =
        Config::kBlockItemsX / Config::kBlockWidth * Config::kElementsPerScalar;
    __align__(16) float output_fragment[kOutputFragmentSize] = {};

    // Helper for computing tile-level partial matmuls.
    Computer computer(values_tile, dense_matrix_fragment, output_fragment);

    // Helper for managing synchronization between collaborating threads.
    Barrier barrier(threadIdx.y);

    //
    /// Begin kernel main loop.
    //

    // For the first iteration of our main loop, we need to possibly mask
    // the first few values from the sparse matrix in case we aligned our
    // values and column indices pointers.
    if (nonzeros >= Config::kBlockItemsK) {
      // Load a tile from the sparse lhs matrix and synchronize the cta.
      sparse_tile_loader.Load();
      barrier.Sync();

      // Mask any values we loaded that aren't from our row of the sparse
      // matrix. Threads could potentially mask values in smem that they
      // were not responsible for loading. Synchronize again to make sure
      // the masking occurs after the previous loads have completed.
      //
      // TODO(tgale): We don't need to synchronize here for the scalar
      // variants of the kernels. We also don't need to handle the first
      // iteration specially. This kernel has now become very complex. It
      // would be nice to break it out into an SpMM class where we can
      // break each of these sections out into helper functions.
      memory_aligner.MaskPrefix(values_tile, column_indices_tile);
      barrier.Sync();

      // Load a tile from the sparse dense_matrix matrix.
      dense_tile_loader.Load(predicates_n);

      // Multiply the tiles and accumulate the results.
      computer.TileMAC();
      nonzeros -= Config::kBlockItemsK;
    }

    // Loop over the tiles in the k-dimension of the dense_matrix/lhs matrices.
    for (; nonzeros >= Config::kBlockItemsK; nonzeros -= Config::kBlockItemsK) {
      // Synchronize s.t. we don't overwrite our shared memory tiles while
      // other warps have not completed using them in computation.
      barrier.Sync();

      // Load a tile from the sparse lhs matrix and synchronize the cta.
      sparse_tile_loader.Load();
      barrier.Sync();

      // Load a tile from the sparse dense_matrix matrix.
      dense_tile_loader.Load(predicates_n);

      // Multiply the tiles and accumulate the results.
      computer.TileMAC();
    }

    //
    /// Begin spmm residue computation.
    //

    // Synchronize s.t. we don't overwrite our shared memory tiles while
    // other warps have not completed using them in computation.
    barrier.Sync();

    // Zero the shared memory tiles s.t. we can operate on sets of 2/4
    // values safely in the dense tile loads and computation.
    if (Config::kResidueUnroll > 1) {
      sparse_tile_loader.ZeroTiles();
      barrier.Sync();
    }

    // Load a tile from the sparse lhs matrix and synchronize the cta.
    sparse_tile_loader.Residue(nonzeros);
    barrier.Sync();

    // Load a tile from the dense dense_matrix matrix and compute immediately.
    dense_tile_loader.ResidueLoadAndCompute(
        nonzeros, predicates_n, values_tile, output_fragment);

    //
    /// Write results to the output.
    //

    // Possibly apply the bias and RelU.
    if (bias != nullptr) {
      // Bias value is shared across all outputs.
      const float bias_value = Load(bias + m_index);
#pragma unroll
      for (int out_idx = 0; out_idx < kOutputFragmentSize; ++out_idx) {
        output_fragment[out_idx] += bias_value;
        output_fragment[out_idx] =
            output_fragment[out_idx] > 0 ? output_fragment[out_idx] : 0;
      }
    }

    // Create a storer for the output matrix.
    OutputTile output_tile_storer(
        m_index,
        n_index,
        n,
        threadIdx.x,
        output_fragment,
        out + blockIdx.z * m * n);
    output_tile_storer.Store(predicates_n);
  }
};

template <typename Config>
__global__ void __launch_bounds__(Config::kThreadsPerBlock) Kernel2(
    int m,
    int k,
    int n,
    const int* __restrict__ row_indices,
    const typename Config::ScalarValue* __restrict__ values,
    const int* __restrict__ row_offsets,
    const typename Config::ScalarIndex* __restrict__ column_indices,
    const typename Config::ScalarValue* __restrict__ dense_matrix,
    const float* __restrict__ bias,
    typename Config::ScalarValue* __restrict__ out,
    int nnz) {
  SpmmKernel2<Config>::KernelFn(
      m,
      k,
      n,
      row_indices,
      values,
      row_offsets,
      column_indices,
      dense_matrix,
      bias,
      out,
      nnz);
}

template <typename Config>
__global__ void __launch_bounds__(
    Config::kThreadsPerBlock,
    Config::kMinOccupancy)
    KernelWithBounds2(
        int m,
        int k,
        int n,
        const int* __restrict__ row_indices,
        const typename Config::ScalarValue* __restrict__ values,
        const int* __restrict__ row_offsets,
        const typename Config::ScalarIndex* __restrict__ column_indices,
        const typename Config::ScalarValue* __restrict__ dense_matrix,
        const float* __restrict__ bias,
        typename Config::ScalarValue* __restrict__ out,
        int nnz) {
  SpmmKernel2<Config>::KernelFn(
      m,
      k,
      n,
      row_indices,
      values,
      row_offsets,
      column_indices,
      dense_matrix,
      bias,
      out,
      nnz);
}

typedef std::function<cudaError_t(
    int, // m: number of rows in lhs & output.
    int, // k: number of cols in lhs and rows in rhs.
    int, // n: number of cols in rhs/output.
    int, // nonzeros: number of nonzero values in lhs.
    const int*, // row_indices: ptr to row index swizzle map.
    const float*, // values: ptr to lhs values.
    const int*, // row_offsets: ptr to lhs row offsets.
    const int*, // column_indices: ptr to lhs column indices.
    const float*, // dense_matrix: ptr to rhs matrix.
    const float*, // bias: bias pointer.
    float*, // output_matrix: ptr to output matrix.
    cudaStream_t,
    int)> // stream: stream to execute in.
    FloatSpmmFn2;

// Lookup table for kernel selection.
using FloatTable2 = std::unordered_map<std::string, FloatSpmmFn2>;

std::string MakeHandle(int m, int k, int n, int nonzeros) {
  // NOTE: We don't include the number of nonzeros currently.
  return std::to_string(m) + "_" + std::to_string(k) + "_" + std::to_string(n);
}

FloatTable2* GetFloatTable2() {
  static FloatTable2 kernel_table = {
      // MBV1 W1.8
      {MakeHandle(920, 920, 196, -1),
       CudaSpmmEx2<SpmmConfig<float, float, float4, 1, 32, 128, 32>>},
      {MakeHandle(920, 464, 196, -1),
       CudaSpmmEx2<SpmmConfig<float, float, float4, 1, 32, 128, 32>>},
      {MakeHandle(232, 115, 3136, -1),
       CudaSpmmEx2<SpmmConfig<float, float, float4, 4, 8, 32, 8, 4, false>>},
      {MakeHandle(232, 232, 3136, -1),
       CudaSpmmEx2<SpmmConfig<float, float2, float4, 4, 16, 32, 8, 4, false>>},
      // MBV1 W1.7
      {MakeHandle(872, 872, 196, -1),
       CudaSpmmEx2<SpmmConfig<float, float, float4, 1, 32, 128, 32>>},
      {MakeHandle(872, 432, 196, -1),
       CudaSpmmEx2<SpmmConfig<float, float, float4, 1, 32, 128, 32>>},
      {MakeHandle(216, 108, 3136, -1),
       CudaSpmmEx2<SpmmConfig<float, float, float4, 4, 8, 32, 8, 4, false>>},
      {MakeHandle(216, 216, 3136, -1),
       CudaSpmmEx2<SpmmConfig<float, float2, float4, 4, 16, 32, 8, 4, false>>},
      // MBV1 W1.6
      {MakeHandle(816, 816, 196, -1),
       CudaSpmmEx2<SpmmConfig<float, float, float4, 1, 32, 128, 32>>},
      {MakeHandle(816, 408, 196, -1),
       CudaSpmmEx2<SpmmConfig<float, float, float4, 1, 32, 128, 32>>},
      {MakeHandle(208, 102, 3136, -1),
       CudaSpmmEx2<SpmmConfig<float, float, float4, 4, 8, 32, 8, 4, false>>},
      {MakeHandle(208, 208, 3136, -1),
       CudaSpmmEx2<SpmmConfig<float, float2, float4, 4, 16, 32, 8, 4, false>>},
      // MBV1 W1.5
      {MakeHandle(768, 768, 196, -1),
       CudaSpmmEx2<SpmmConfig<float, float, float4, 1, 32, 128, 32>>},
      {MakeHandle(768, 384, 196, -1),
       CudaSpmmEx2<SpmmConfig<float, float, float4, 1, 32, 128, 32>>},
      {MakeHandle(192, 96, 3136, -1),
       CudaSpmmEx2<SpmmConfig<float, float, float4, 4, 8, 32, 8, 4, false>>},
      {MakeHandle(192, 192, 3136, -1),
       CudaSpmmEx2<SpmmConfig<float, float, float4, 4, 8, 32, 8, 4, false>>},
      // MBV1 W1.4
      {MakeHandle(720, 720, 196, -1),
       CudaSpmmEx2<SpmmConfig<float, float, float4, 1, 32, 128, 32>>},
      {MakeHandle(720, 360, 196, -1),
       CudaSpmmEx2<SpmmConfig<float, float, float4, 1, 32, 128, 32>>},
      {MakeHandle(176, 89, 3136, -1),
       CudaSpmmEx2<SpmmConfig<float, float, float4, 4, 8, 32, 8, 4, false>>},
      {MakeHandle(176, 176, 3136, -1),
       CudaSpmmEx2<SpmmConfig<float, float, float4, 4, 8, 32, 8, 4, false>>},
      // MBV1 W1.3
      {MakeHandle(664, 664, 196, -1),
       CudaSpmmEx2<SpmmConfig<float, float, float4, 1, 32, 128, 32>>},
      {MakeHandle(664, 336, 196, -1),
       CudaSpmmEx2<SpmmConfig<float, float, float4, 1, 32, 128, 32>>},
      {MakeHandle(168, 83, 3136, -1),
       CudaSpmmEx2<SpmmConfig<float, float, float4, 4, 8, 32, 8, 4, false>>},
      {MakeHandle(168, 168, 3136, -1),
       CudaSpmmEx2<SpmmConfig<float, float, float4, 4, 8, 32, 8, 4, false>>}};
  return &kernel_table;
}

FloatSpmmFn2 GetKernel(int m, int k, int n, int nonzeros) {
  FloatTable2* kernel_table = GetFloatTable2();
  auto it = kernel_table->find(MakeHandle(m, k, n, nonzeros));
  if (it == kernel_table->end()) {
    // Return uninitialized function to defer to the standard heuristic.
    FloatSpmmFn2 nullfn;
    return nullfn;
  }
  return it->second;
}

} // namespace

template <typename Config>
cudaError_t CudaSpmmEx2(
    int m,
    int k,
    int n,
    int nonzeros,
    const int* __restrict__ row_indices,
    const typename Config::ScalarValue* __restrict__ values,
    const int* __restrict__ row_offsets,
    const typename Config::ScalarIndex* __restrict__ column_indices,
    const typename Config::ScalarValue* __restrict__ dense_matrix,
    const float* __restrict__ bias,
    typename Config::ScalarValue* __restrict__ output_matrix,
    cudaStream_t stream,
    int batch_size) {
  dim3 grid_dim(
      ceil(static_cast<float>(m) / Config::kBlockItemsY),
      ceil(
          static_cast<float>(n) / Config::kBlockItemsX /
          Config::kElementsPerScalar),
      batch_size);
  dim3 block_dim(Config::kBlockWidth, Config::kBlockItemsY, 1);

  if (Config::kLaunchBounds) {
    KernelWithBounds2<Config><<<grid_dim, block_dim, 0, stream>>>(
        m,
        k,
        n,
        row_indices,
        values,
        row_offsets,
        column_indices,
        dense_matrix,
        bias,
        output_matrix,
        nonzeros);
  } else {
    Kernel2<Config><<<grid_dim, block_dim, 0, stream>>>(
        m,
        k,
        n,
        row_indices,
        values,
        row_offsets,
        column_indices,
        dense_matrix,
        bias,
        output_matrix,
        nonzeros);
  }
  return cudaGetLastError();
}

cudaError_t CudaSpmmBiasRelu2(
    int m,
    int k,
    int n,
    int nonzeros,
    const int* __restrict__ row_indices,
    const float* __restrict__ values,
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const float* __restrict__ dense_matrix,
    const float* __restrict__ bias,
    float* __restrict__ output_matrix,
    cudaStream_t stream,
    int batch_size) {
  // Try finding a specific kernel in the table. If we find a valid
  // one, call it and return.
  auto spmm_kernel = GetKernel(m, k, n, nonzeros);
  if (spmm_kernel) {
    return spmm_kernel(
        m,
        k,
        n,
        nonzeros,
        row_indices,
        values,
        row_offsets,
        column_indices,
        dense_matrix,
        bias,
        output_matrix,
        stream,
        batch_size);
  }

  // A very simple kernel selection heuristic. For small batch sizes,
  // we use the hybrid kernel variants with float4 sparse matrix loads.
  // For mid to large batch sizes, we use the standard float4 kernel with
  // and n-dimension tile of 32. On our synthetic RNN problem data this
  // gives us about 96% of the performance of a kernel selection oracle.
  //
  // TODO(tgale): We should improve the code here to make it more extensible
  // and less repetitive. We should also improve this heuristic to improve
  // performance on a wider range of problems.
  //
  // TODO(tgale): Update these heuristics to take batch size vector alignment
  // into account. This is currently not a perfectly general API.
  if ((n % 4) == 0) {
    if (n == 8) {
      // No predicates in the n-dimension.
      typedef SpmmConfig<float, float4, float, 4, 32, 8, 8, 4, false> Config;
      return CudaSpmmEx2<Config>(
          m,
          k,
          n,
          nonzeros,
          row_indices,
          values,
          row_offsets,
          column_indices,
          dense_matrix,
          bias,
          output_matrix,
          stream,
          batch_size);
    } else if (n < 8) {
      typedef SpmmConfig<float, float4, float, 4, 32, 8, 8> Config;
      return CudaSpmmEx2<Config>(
          m,
          k,
          n,
          nonzeros,
          row_indices,
          values,
          row_offsets,
          column_indices,
          dense_matrix,
          bias,
          output_matrix,
          stream,
          batch_size);
    } else if (n == 16) {
      // No predicates in the n-dimension.
      typedef SpmmConfig<float, float4, float2, 4, 32, 16, 8, 4, false> Config;
      return CudaSpmmEx2<Config>(
          m,
          k,
          n,
          nonzeros,
          row_indices,
          values,
          row_offsets,
          column_indices,
          dense_matrix,
          bias,
          output_matrix,
          stream,
          batch_size);
    } else if (n < 16) {
      typedef SpmmConfig<float, float4, float2, 4, 32, 16, 8> Config;
      return CudaSpmmEx2<Config>(
          m,
          k,
          n,
          nonzeros,
          row_indices,
          values,
          row_offsets,
          column_indices,
          dense_matrix,
          bias,
          output_matrix,
          stream,
          batch_size);
    } else if (n == 32) {
      // No predicates in the n-dimension.
      typedef SpmmConfig<float, float4, float4, 4, 32, 32, 8, 4, false> Config;
      return CudaSpmmEx2<Config>(
          m,
          k,
          n,
          nonzeros,
          row_indices,
          values,
          row_offsets,
          column_indices,
          dense_matrix,
          bias,
          output_matrix,
          stream,
          batch_size);
    } else if ((n % 64) == 0) {
      // No predicates in n-dimension. Set kMinOccupancy to 8 to avoid
      // register spilling. Note that we only use this `large-tile` variant
      // if the batch size is divisble by 64.
      typedef SpmmConfig<float, float4, float4, 4, 32, 64, 8, 4, false, true, 8>
          Config;
      return CudaSpmmEx2<Config>(
          m,
          k,
          n,
          nonzeros,
          row_indices,
          values,
          row_offsets,
          column_indices,
          dense_matrix,
          bias,
          output_matrix,
          stream,
          batch_size);
    } else {
      // Default kernel. 32-wide tile dimensions with 4-wide vector loads and
      // 4-way subwarp tiling. Run for all batch sizes greater than 16, unless
      // the batch size is divisible by 64.
      typedef SpmmConfig<float, float4, float4, 4, 32, 32, 8> Config;
      return CudaSpmmEx2<Config>(
          m,
          k,
          n,
          nonzeros,
          row_indices,
          values,
          row_offsets,
          column_indices,
          dense_matrix,
          bias,
          output_matrix,
          stream,
          batch_size);
    }
  } else if ((n % 2) == 0) {
    typedef SpmmConfig<float, float2, float2, 2, 32, 32, 16> Config;
    return CudaSpmmEx2<Config>(
        m,
        k,
        n,
        nonzeros,
        row_indices,
        values,
        row_offsets,
        column_indices,
        dense_matrix,
        bias,
        output_matrix,
        stream,
        batch_size);
  } else {
    // Scalar kernel.
    typedef SpmmConfig<float, float, float, 1, 32, 32, 32> Config;
    return CudaSpmmEx2<Config>(
        m,
        k,
        n,
        nonzeros,
        row_indices,
        values,
        row_offsets,
        column_indices,
        dense_matrix,
        bias,
        output_matrix,
        stream,
        batch_size);
  }
}

cudaError_t CudaSpmm2(
    int m,
    int k,
    int n,
    int nonzeros,
    const int* __restrict__ row_indices,
    const float* __restrict__ values,
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const float* __restrict__ dense_matrix,
    float* __restrict__ output_matrix,
    cudaStream_t stream,
    int batch_size) {
  return CudaSpmmBiasRelu2(
      m,
      k,
      n,
      nonzeros,
      row_indices,
      values,
      row_offsets,
      column_indices,
      dense_matrix,
      /* bias = */ nullptr,
      output_matrix,
      stream,
      batch_size);
}

} // namespace sputnik

at::Tensor spmm_sputnik(
    const at::Tensor& b,
    const at::Tensor& row_indices,
    const at::Tensor& values,
    const at::Tensor& row_offsets,
    const at::Tensor& column_indices,
    int64_t m) {
  TORCH_CHECK(b.dim() == 3);
  TORCH_CHECK(values.dim() == 2);
  TORCH_CHECK(b.size(0) == values.size(0));
  TORCH_CHECK(row_indices.dim() == 1);
  TORCH_CHECK(row_offsets.dim() == 1);
  TORCH_CHECK(column_indices.dim() == 1);
  TORCH_CHECK(values.size(1) == column_indices.size(0));

  TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
  TORCH_CHECK(row_indices.is_cuda(), "row_indices must be a CUDA tensor");
  TORCH_CHECK(values.is_cuda(), "values must be a CUDA tensor");
  TORCH_CHECK(row_offsets.is_cuda(), "row_offsets must be a CUDA tensor");
  TORCH_CHECK(column_indices.is_cuda(), "column_offsets must be a CUDA tensor");

  TORCH_CHECK(b.is_contiguous(), "b must be a contiguous tensor");
  TORCH_CHECK(
      row_indices.is_contiguous(), "row_indices must be a contiguous tensor");
  TORCH_CHECK(values.is_contiguous(), "values must be a contiguous tensor");
  TORCH_CHECK(
      row_offsets.is_contiguous(), "row_offsets must be a contiguous tensor");
  TORCH_CHECK(
      column_indices.is_contiguous(),
      "column_offsets must be a contiguous tensor");

  TORCH_CHECK(!b.is_sparse(), "b must be a dense tensor");
  TORCH_CHECK(!row_indices.is_sparse(), "row_indices must be a dense tensor");
  TORCH_CHECK(!values.is_sparse(), "values must be a dense tensor");
  TORCH_CHECK(!row_offsets.is_sparse(), "row_offsets must be a dense tensor");
  TORCH_CHECK(
      !column_indices.is_sparse(), "column_offsets must be a dense tensor");

  TORCH_CHECK(
      values.device() == b.device(),
      "values should be in the same device as b");
  TORCH_CHECK(
      values.device() == row_indices.device(),
      "a should be in the same device as row_indices");
  TORCH_CHECK(
      values.device() == row_offsets.device(),
      "a should be in the same device as row_offsets");
  TORCH_CHECK(
      values.device() == column_indices.device(),
      "a should be in the same device as column_indices");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int batch = b.size(0);
  int k = b.size(1);
  int n = b.size(2);

  int nonzeros = column_indices.size(0);
  TORCH_CHECK(
      batch == 1 || nonzeros % 4 == 0,
      "If batch size > 1 then number of nonzeros should be a multiple of 4");

  at::Tensor output = at::empty({batch, m, n}, b.options());

  // TODO investigate misaligned address errors in values ptr
  AT_CUDA_CHECK(
      sputnik::CudaSpmm2(
          m,
          k,
          n,
          nonzeros,
          row_indices.data_ptr<int>(),
          values.data_ptr<float>(),
          row_offsets.data_ptr<int>(),
          column_indices.data_ptr<int>(),
          b.data_ptr<float>(),
          output.data_ptr<float>(),
          stream,
          batch));

  return output;
}

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::spmm_sputnik"), TORCH_FN(spmm_sputnik));
}
