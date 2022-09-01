#include <ATen/ATen.h>
#include <torch/library.h>
#include <cmath>
#include <vector>

#include <cuda_fp16.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/vector.h"
#include "cutlass/numeric_types.h"

#include "debug_utils.h"
#include "gemm_kernel_utils.h"

#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/epilogue/threadblock/epilogue_smem_accumulator.h"
#include "cutlass/epilogue/warp/fragment_iterator_tensor_op.h"
#include "cutlass/epilogue/warp/tile_iterator_tensor_op.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/platform/platform.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/transform/threadblock/vector_iterator.h"

#include "find_default_mma.h"
#include "mma_from_smem.h"

#include <inttypes.h>

using namespace gemm_kernel_utils;

namespace {
template <typename scalar_t, typename Arch>
constexpr int getWarpsPerSm() {
  bool is_half = !std::is_same<scalar_t, float>::value;
  if (Arch::kMinComputeCapability >= 80) {
    return is_half ? 12 : 8;
  }
  return 8;
}

template <typename scalar_t_, bool kIsAligned_, typename ArchTag>
struct AttentionBackwardKernel {
  using scalar_t = scalar_t_;
  using output_t = scalar_t;
  using lse_scalar_t = float;
  using accum_t = float;
  static constexpr bool kIsAligned = kIsAligned_;

  struct Params {
    // Input tensors
    scalar_t* query_ptr; // [num_queries, head_dim]
    scalar_t* key_ptr; // [num_keys, head_dim]
    scalar_t* value_ptr; // [num_keys, head_dim_value]
    lse_scalar_t* logsumexp_ptr; // [num_queries]
    scalar_t* output_ptr; // [num_queries, head_dim_value]
    scalar_t* grad_output_ptr; // [num_queries, head_dim_value]

    // Output tensors
    scalar_t* grad_query_ptr; // [num_queries, head_dim]
    scalar_t* grad_key_ptr; // [num_keys, head_dim]
    scalar_t* grad_value_ptr; // [num_keys, head_dim_value]

    // Dimensions/strides
    int32_t head_dim;
    int32_t head_dim_value;
    int32_t num_queries;
    int32_t num_keys;

    __device__ void advance_batches(int32_t batch_id) {
      constexpr int32_t kAlignLSE = 32; // block size of backward
      auto lse_dim = ceil_div((int32_t)num_queries, kAlignLSE) * kAlignLSE;

      query_ptr += batch_id * head_dim * num_queries;
      key_ptr += batch_id * head_dim * num_keys;
      value_ptr += batch_id * head_dim_value * num_keys;
      logsumexp_ptr += batch_id * lse_dim;
      output_ptr += batch_id * head_dim_value * num_queries;
      grad_output_ptr += batch_id * head_dim_value * num_queries;

      grad_query_ptr += batch_id * head_dim * num_queries;
      grad_key_ptr += batch_id * head_dim * num_keys;
      grad_value_ptr += batch_id * head_dim_value * num_keys;
    }
  };

  // Blocks & grid
  static constexpr int64_t kWarpSize = 32;
  static constexpr int64_t kNumWarpsPerBlock = 4;
  static constexpr int64_t kBlockSizeI = 64;
  static constexpr int64_t kBlockSizeJ = 64;

  // Launch bounds
  static constexpr int64_t kNumThreads = kWarpSize * kNumWarpsPerBlock;
  static constexpr int64_t kMinBlocksPerSm =
      getWarpsPerSm<scalar_t, ArchTag>() / kNumWarpsPerBlock;

  static int64_t getNumBlocksY(int64_t num_queries) {
    return 1; // ceil_div(num_queries, kBlockSizeI);
  }

  static int64_t getNumBlocksX(int64_t num_values) {
    return 1; // ceil_div(num_values, kBlockSizeJ);
  }

  using GemmType = DefaultGemmType<ArchTag, scalar_t>;
  using DefaultConfig =
      typename cutlass::gemm::device::DefaultGemmConfiguration<
          typename GemmType::OpClass,
          ArchTag,
          scalar_t,
          scalar_t,
          scalar_t, // ElementC
          accum_t // ElementAccumulator
          >;
  static constexpr auto kOptimalAlignement =
      std::max(DefaultConfig::kAlignmentA, DefaultConfig::kAlignmentB);
  static constexpr auto kMinimumAlignment = GemmType::kMinimumAlignment;

  struct MatmulQK {
    /*
    attn_T = k_j @ q_i.transpose(-2, -1) # matmul
    attn_T = (attn_T - logsumexp[i_start:i_end].unsqueeze(1).transpose(-2,
    -1)).exp() # epilogue

    with attn_T.shape = (kBlockSizeJ, kBlockSizeI)
    */
    using ThreadblockShape =
        cutlass::gemm::GemmShape<kBlockSizeJ, kBlockSizeI, GemmType::ThreadK>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, GemmType::WarpK>;
    using DefaultMma = typename cutlass::gemm::threadblock::DefaultMma<
        scalar_t, // ElementA
        cutlass::layout::RowMajor, // LayoutA
        kIsAligned ? DefaultConfig::kAlignmentA : GemmType::kMinimumAlignment,
        scalar_t, // ElementB
        cutlass::layout::ColumnMajor, // LayoutB
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment,
        accum_t, // ElementC
        cutlass::layout::RowMajor, // LayoutC
        typename GemmType::OpClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        typename GemmType::InstructionShape,
        DefaultConfig::kStages,
        typename GemmType::Operator,
        false, // AccumulatorsInRowMajor = false,
        ArchTag::kMinComputeCapability >= 80
            ? cutlass::gemm::SharedMemoryClearOption::kZfill
            : cutlass::gemm::SharedMemoryClearOption::kNone>;
    using MmaCore = typename DefaultMma::MmaCore;
    using Mma = typename DefaultMma::ThreadblockMma;

    // Epilogue to store to shared-memory in a format that we can use later for
    // the second matmul
    using B2bGemm = typename cutlass::gemm::threadblock::B2bGemm<
        typename Mma::Operator::IteratorC,
        typename Mma::Operator,
        scalar_t,
        WarpShape,
        ThreadblockShape>;
    using AccumulatorSharedStorage = typename B2bGemm::AccumulatorSharedStorage;
  };

  struct MatmulGradV {
    /*
    grad_v[j_start:j_end] += attn_T @ do_i # matmul

    Dimensions: (kBlockSizeJ * kNumWarpsPerBlock, kBlockSizeI, K)
    (we might need to iterate multiple times on K)
    */
    using ThreadblockShape =
        cutlass::gemm::GemmShape<kBlockSizeJ, kBlockSizeI, GemmType::ThreadK>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, GemmType::WarpK>;
    using InstructionShape = typename GemmType::InstructionShape;

    using DefaultGemm = cutlass::gemm::kernel::DefaultGemm<
        scalar_t, // ElementA,
        cutlass::layout::RowMajor, // LayoutA,
        DefaultConfig::kAlignmentA,
        scalar_t, // ElementB,
        cutlass::layout::RowMajor, // LayoutB,
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment,
        output_t,
        cutlass::layout::RowMajor, // LayoutC,
        accum_t,
        typename GemmType::OpClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        typename GemmType::InstructionShape,
        typename DefaultConfig::EpilogueOutputOp,
        void, // ThreadblockSwizzle - not used
        DefaultConfig::kStages,
        false, // SplitKSerial
        typename GemmType::Operator>;

    using DefaultMmaFromSmem =
        typename cutlass::gemm::threadblock::DefaultMmaFromSharedMemory<
            typename DefaultGemm::Mma,
            typename MatmulQK::AccumulatorSharedStorage>;
    using Mma = typename DefaultMmaFromSmem::Mma;
    using IteratorB = typename Mma::IteratorB;
    using WarpCount = typename Mma::WarpCount;

    // Epilogue
    using DefaultOutputOp = typename DefaultConfig::EpilogueOutputOp;
    using DefaultEpilogue = typename DefaultGemm::Epilogue;
    using OutputTileIterator = typename DefaultEpilogue::OutputTileIterator;

    struct SharedStorage {
      union {
        // Storing parts of `V` during the matmul
        typename Mma::SharedStorage mm;
        // Used by the Epilogue (so we can reuse the same memory space)
        typename DefaultEpilogue::SharedStorage epilogue;
      };
    };
  };

  struct MatmulDOIVJ {
    /*
    doi_t_vj = do_i @ v_j.transpose(-2, -1) # matmul
    tmp = (doi_t_vj - Di.unsqueeze(1)) * attn # inplace / epilogue?
    */
    using ThreadblockShape =
        cutlass::gemm::GemmShape<kBlockSizeI, kBlockSizeJ, GemmType::ThreadK>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, GemmType::WarpK>;
    using DefaultMma = typename cutlass::gemm::threadblock::DefaultMma<
        scalar_t, // ElementA
        cutlass::layout::RowMajor, // LayoutA
        kIsAligned ? DefaultConfig::kAlignmentA : GemmType::kMinimumAlignment,
        scalar_t, // ElementB
        cutlass::layout::ColumnMajor, // LayoutB
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment,
        accum_t, // ElementC
        cutlass::layout::RowMajor, // LayoutC
        typename GemmType::OpClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        typename GemmType::InstructionShape,
        DefaultConfig::kStages,
        typename GemmType::Operator,
        false, // AccumulatorsInRowMajor = false,
        ArchTag::kMinComputeCapability >= 80
            ? cutlass::gemm::SharedMemoryClearOption::kZfill
            : cutlass::gemm::SharedMemoryClearOption::kNone>;
    using MmaCore = typename DefaultMma::MmaCore;
    using Mma = typename DefaultMma::ThreadblockMma;

    // Epilogue to store to shared-memory in a format that we can use later for
    // the second matmul
    using B2bGemm = typename cutlass::gemm::threadblock::B2bGemm<
        typename Mma::Operator::IteratorC,
        typename Mma::Operator,
        scalar_t,
        WarpShape,
        ThreadblockShape>;
    using AccumulatorSharedStorage = typename B2bGemm::AccumulatorSharedStorage;
  };

  struct MatmulGradQ {
    // grad_q <- tmp @ k_j
    using ThreadblockShape =
        cutlass::gemm::GemmShape<kBlockSizeI, kBlockSizeJ, GemmType::ThreadK>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, GemmType::WarpK>;
    using InstructionShape = typename GemmType::InstructionShape;

    using DefaultGemm = cutlass::gemm::kernel::DefaultGemm<
        scalar_t, // ElementA,
        cutlass::layout::RowMajor, // LayoutA,
        DefaultConfig::kAlignmentA,
        scalar_t, // ElementB,
        cutlass::layout::RowMajor, // LayoutB,
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment,
        output_t,
        cutlass::layout::RowMajor, // LayoutC,
        accum_t,
        typename GemmType::OpClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        typename GemmType::InstructionShape,
        typename DefaultConfig::EpilogueOutputOp,
        void, // ThreadblockSwizzle - not used
        DefaultConfig::kStages,
        false, // SplitKSerial
        typename GemmType::Operator>;

    using DefaultMmaFromSmem =
        typename cutlass::gemm::threadblock::DefaultMmaFromSharedMemory<
            typename DefaultGemm::Mma,
            typename MatmulDOIVJ::AccumulatorSharedStorage>;
    using Mma = typename DefaultMmaFromSmem::Mma;
    using IteratorB = typename Mma::IteratorB;
    using WarpCount = typename Mma::WarpCount;

    // Epilogue
    using DefaultOutputOp = typename DefaultConfig::EpilogueOutputOp;
    using DefaultEpilogue = typename DefaultGemm::Epilogue;
    using OutputTileIterator = typename DefaultEpilogue::OutputTileIterator;

    struct SharedStorage {
      union {
        // Storing parts of `V` during the matmul
        typename Mma::SharedStorage mm;
        // Used by the Epilogue (so we can reuse the same memory space)
        typename DefaultEpilogue::SharedStorage epilogue;
      };
    };
  };
  struct MatmulGradK {
    // grad_k <- tmp.transpose(-2, -1) @ q_i
    using ThreadblockShape =
        cutlass::gemm::GemmShape<kBlockSizeJ, kBlockSizeI, GemmType::ThreadK>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, GemmType::WarpK>;
    using InstructionShape = typename GemmType::InstructionShape;

    using DefaultGemm = cutlass::gemm::kernel::DefaultGemm<
        scalar_t, // ElementA,
        cutlass::layout::RowMajor, // LayoutA,
        DefaultConfig::kAlignmentA,
        scalar_t, // ElementB,
        cutlass::layout::RowMajor, // LayoutB,
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment,
        output_t,
        cutlass::layout::RowMajor, // LayoutC,
        accum_t,
        typename GemmType::OpClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        typename GemmType::InstructionShape,
        typename DefaultConfig::EpilogueOutputOp,
        void, // ThreadblockSwizzle - not used
        DefaultConfig::kStages,
        false, // SplitKSerial
        typename GemmType::Operator>;

    using DefaultMmaFromSmem =
        typename cutlass::gemm::threadblock::DefaultMmaFromSharedMemory<
            typename DefaultGemm::Mma,
            typename MatmulQK::AccumulatorSharedStorage>;
    using Mma = typename DefaultMmaFromSmem::Mma;
    using IteratorB = typename Mma::IteratorB;
    using WarpCount = typename Mma::WarpCount;

    // Epilogue
    using DefaultOutputOp = typename DefaultConfig::EpilogueOutputOp;
    using DefaultEpilogue = typename DefaultGemm::Epilogue;
    using OutputTileIterator = typename DefaultEpilogue::OutputTileIterator;

    struct SharedStorage {
      union {
        typename Mma::SharedStorage mm;
        typename DefaultEpilogue::SharedStorage epilogue;
      };
    };
  };

  struct SharedStorage {
    struct AfterDOIJV {
      typename MatmulDOIVJ::AccumulatorSharedStorage doivj_shared_storage;
      union {
        typename MatmulGradQ::SharedStorage mm_gradQ;
        typename MatmulGradK::SharedStorage mm_gradK;
      };
    };
    struct AfterQK {
      typename MatmulQK::AccumulatorSharedStorage attn_shared_storage;
      union {
        typename MatmulGradV::SharedStorage mm_gradV;
        typename MatmulDOIVJ::Mma::SharedStorage mm_doivj;
        AfterDOIJV after_doivj;
      };
    };
    cutlass::Array<accum_t, kBlockSizeI> di; // (do_i * o_i).sum(-1)
    union {
      typename MatmulQK::Mma::SharedStorage qk;
      AfterQK after_qk;
    };
  };

  // OLD VERSION - a3f257389709
  template <int kElementsPerAccess>
  static __device__ void _computeDi(
      cutlass::Array<accum_t, kBlockSizeI>& di,
      Params const& p,
      int32_t query_start) {
    __syncthreads();
    using AccessType = cutlass::Array<scalar_t, kElementsPerAccess>;
    static constexpr int kNumThreadsPerLine = 4;
    static constexpr int kParallelRowsPerWarp = kWarpSize / kNumThreadsPerLine;

    int32_t laneCol = (get_lane_id() % kNumThreadsPerLine);
    int32_t laneRow = (get_lane_id() / kNumThreadsPerLine) +
        get_warp_id() * kBlockSizeI / kNumWarpsPerBlock;

    int32_t dO_s0 = p.head_dim_value / AccessType::kElements;
    int32_t out_s0 = p.head_dim_value / AccessType::kElements;
    cutlass::
        Array<accum_t, kBlockSizeI / kParallelRowsPerWarp / kNumWarpsPerBlock>
            di_frag;
    di_frag.clear();
    assert(p.head_dim_value % AccessType::kElements == 0);
    CUTLASS_PRAGMA_UNROLL
    for (int firstCol = 0; firstCol < p.head_dim_value;
         firstCol += kNumThreadsPerLine * AccessType::kElements) {
      const __restrict__ AccessType* dO =
          reinterpret_cast<const __restrict__ AccessType*>(
              p.grad_output_ptr + (query_start + laneRow) * p.head_dim_value +
              firstCol);
      const __restrict__ AccessType* out =
          reinterpret_cast<const __restrict__ AccessType*>(
              p.output_ptr + (query_start + laneRow) * p.head_dim_value +
              firstCol);
      int32_t rowEnd = (p.num_queries - query_start);
      int32_t colEnd = p.head_dim_value / AccessType::kElements;

      AccessType frag_dO;
      AccessType frag_out;
      AccessType result;
      frag_dO.clear();
      frag_out.clear();
      dO += laneCol;
      out += laneCol;

      bool withinBounds =
          firstCol + laneCol * AccessType::kElements < p.head_dim_value;

      CUTLASS_PRAGMA_UNROLL
      for (int frag_idx = 0; frag_idx < di_frag.size(); ++frag_idx) {
        int32_t fetching_index = laneRow + frag_idx * kParallelRowsPerWarp;
        if (fetching_index >= rowEnd) {
          break;
        }
        if (withinBounds) {
          frag_dO = *dO;
          frag_out = *out;
          dO += dO_s0 * kParallelRowsPerWarp;
          out += out_s0 * kParallelRowsPerWarp;
          cutlass::multiplies<AccessType> multiply;
          result = multiply(frag_dO, frag_out);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < AccessType::kElements; ++i) {
            di_frag[frag_idx] = di_frag[frag_idx] + accum_t(result[i]);
          }
        }
      }
    }
    // Store everything in smem
    CUTLASS_PRAGMA_UNROLL
    for (int frag_idx = 0; frag_idx < di_frag.size(); ++frag_idx) {
      int32_t fetching_index = laneRow + frag_idx * kParallelRowsPerWarp;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 1; i < kNumThreadsPerLine; i *= 2) {
        di_frag[frag_idx] = di_frag[frag_idx] +
            __shfl_xor_sync(0xffffffff, di_frag[frag_idx], i);
      }
      di[fetching_index] = di_frag[frag_idx];
    }
    __syncthreads();
  }

  static __device__ void computeDi(
      cutlass::Array<accum_t, kBlockSizeI>& di,
      Params const& p,
      int32_t query_start) {
    constexpr int kOptimalElements =
        128 / cutlass::sizeof_bits<scalar_t>::value;
    if (p.head_dim_value % kOptimalElements == 0) {
      _computeDi<kOptimalElements>(di, p, query_start);
    } else {
      _computeDi<1>(di, p, query_start);
    }
  }

  static __device__ void kernel(Params& p_) {
    // Hint to nvcc to store points & tensor shapes in registers
    // as we use them a lot
    register const Params p = p_;

    extern __shared__ char smem_buffer[];
    SharedStorage& shared_storage = *((SharedStorage*)smem_buffer);
    // int32_t key_start = blockIdx.x * kBlockSizeJ;
    // int32_t query_start = blockIdx.y * kBlockSizeI;
    int32_t thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    int32_t warp_id = threadIdx.y;

    auto clearSmem = [&]() {
      // Initialize shared-memory. It can contain `nans` otherwise that screw up
      // everything (only seens on Sm75+ tho)
      uint32_t* smem = (uint32_t*)smem_buffer;
      for (int i = 0; i < sizeof(SharedStorage) / sizeof(uint32_t) -
               kWarpSize * kNumWarpsPerBlock;
           i += kWarpSize * kNumWarpsPerBlock) {
        smem[i + thread_id] = 0;
      }
    };

    int32_t query_end = p.num_queries / kBlockSizeI * kBlockSizeI;
    int32_t key_end = p.num_keys / kBlockSizeJ * kBlockSizeJ;
    int32_t query_start = 0;
    for (; query_start < query_end; query_start += kBlockSizeI) {
      clearSmem();
      computeDi(shared_storage.di, p, query_start);
      int32_t key_start = 0;
      for (; key_start < key_end; key_start += kBlockSizeJ) {
        processBlockIJ<true>(shared_storage, p, query_start, key_start);
      }
      // last (partial) key
      if (key_start != p.num_keys) {
        processBlockIJ<false>(shared_storage, p, query_start, key_start);
      }
    }
    // Last (partial) query block
    if (query_start != p.num_queries) {
      computeDi(shared_storage.di, p, query_start);
      for (int32_t key_start = 0; key_start < p.num_keys;
           key_start += kBlockSizeJ) {
        processBlockIJ<false>(shared_storage, p, query_start, key_start);
      }
    }
  }

  // Compute threadblock location
  template <bool skipBoundsChecks>
  static __device__ __forceinline__ void processBlockIJ(
      SharedStorage& shared_storage,
      Params const& p,
      int32_t query_start,
      int32_t key_start) {
    cutlass::MatrixCoord no_offset{0, 0};
    accum_t scale = accum_t(1.0 / std::sqrt(float(p.head_dim)));
    int32_t thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    int32_t warp_id = threadIdx.y;
    int32_t lane_id = threadIdx.x;

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // MatmulQK
    /////////////////////////////////////////////////////////////////////////////////////////////////
    {
      using Mma = typename MatmulQK::Mma;

      cutlass::gemm::GemmCoord problem_size(
          skipBoundsChecks ? MatmulQK::ThreadblockShape::kM
                           : p.num_keys - key_start,
          skipBoundsChecks ? MatmulQK::ThreadblockShape::kN
                           : p.num_queries - query_start,
          p.head_dim // k
      );

      // k_j
      typename Mma::IteratorA iterator_A(
          {int32_t(p.head_dim)},
          p.key_ptr + key_start * p.head_dim,
          {problem_size.m(), problem_size.k()},
          thread_id,
          no_offset);

      // q_i.transpose(-2, -1)
      typename Mma::IteratorB iterator_B(
          {int32_t(p.head_dim)},
          p.query_ptr + query_start * p.head_dim,
          {problem_size.k(), problem_size.n()},
          thread_id,
          no_offset);

      Mma mma(shared_storage.qk, thread_id, warp_id, lane_id);

      typename Mma::FragmentC accum;

      accum.clear();

      auto gemm_k_iterations =
          (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // Compute threadblock-scoped matrix multiply-add
      mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);
      accum = cutlass::multiplies<typename Mma::FragmentC>()(scale, accum);

      // Epilogue: add LSE + exp and store that to our shared memory buffer
      // shmem <- (matmul_result -
      // logsumexp[i_start:i_end].unsqueeze(1)).exp()
      int warp_idx_mn_0 =
          warp_id % (Mma::Base::WarpCount::kM * Mma::Base::WarpCount::kN);
      auto output_tile_coords = cutlass::MatrixCoord{
          warp_idx_mn_0 % Mma::Base::WarpCount::kM,
          warp_idx_mn_0 / Mma::Base::WarpCount::kM};
      __syncthreads();
      MatmulQK::B2bGemm::accumApplyLSEToSmem(
          shared_storage.after_qk.attn_shared_storage,
          accum,
          p.logsumexp_ptr + query_start,
          problem_size.n(),
          thread_id,
          warp_id,
          lane_id,
          output_tile_coords);
      __syncthreads();
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // GradV matmul
    //
    // grad_v[j_start:j_end] += attn_T @ do_i
    /////////////////////////////////////////////////////////////////////////////////////////////////
    for (int col = 0; col < p.head_dim_value;
         col += MatmulGradV::ThreadblockShape::kN) {
      using Mma = typename MatmulGradV::Mma;

      cutlass::gemm::GemmCoord problem_size(
          skipBoundsChecks ? MatmulGradV::ThreadblockShape::kM
                           : p.num_keys - key_start,
          p.head_dim_value - col,
          skipBoundsChecks ? MatmulQK::Mma::Shape::kN
                           : std::min(
                                 (int32_t)MatmulQK::Mma::Shape::kN,
                                 p.num_queries - query_start));

      // q_i.transpose(-2, -1)
      typename Mma::IteratorB iterator_B(
          {int32_t(p.head_dim_value)},
          p.grad_output_ptr + query_start * p.head_dim_value + col,
          {problem_size.k(), problem_size.n()},
          thread_id,
          no_offset);

      Mma mma(
          shared_storage.after_qk.mm_gradV.mm,
          shared_storage.after_qk.attn_shared_storage,
          thread_id,
          warp_id,
          lane_id,
          problem_size.k());

      typename Mma::FragmentC accum;

      accum.clear();

      auto gemm_k_iterations =
          (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // Compute threadblock-scoped matrix multiply-add
      __syncthreads();

      mma(gemm_k_iterations, accum, iterator_B, accum);
      __syncthreads();

      // Output results
      typename MatmulGradV::OutputTileIterator output_read_it(
          typename MatmulGradV::OutputTileIterator::Params{p.head_dim_value},
          p.grad_value_ptr + key_start * p.head_dim_value + col,
          {skipBoundsChecks ? MatmulGradV::ThreadblockShape::kM
                            : p.num_keys - key_start,
           p.head_dim_value - col},
          thread_id);
      typename MatmulGradV::OutputTileIterator output_write_it = output_read_it;

      DISPATCH_BOOL(
          query_start == 0, kIsFirst, ([&]() {
            using DefaultEpilogue = typename MatmulGradV::DefaultEpilogue;
            using DefaultOutputOp = typename MatmulGradV::DefaultOutputOp;
            static constexpr auto ScaleType = kIsFirst
                ? cutlass::epilogue::thread::ScaleType::Nothing
                : cutlass::epilogue::thread::ScaleType::NoBetaScaling;
            using EpilogueOutputOp =
                typename cutlass::epilogue::thread::LinearCombination<
                    typename DefaultOutputOp::ElementOutput,
                    DefaultOutputOp::kCount,
                    typename DefaultOutputOp::ElementAccumulator,
                    typename DefaultOutputOp::ElementCompute,
                    ScaleType>;
            using Epilogue = typename cutlass::epilogue::threadblock::Epilogue<
                typename DefaultEpilogue::Shape,
                typename Mma::Operator,
                DefaultEpilogue::kPartitionsK,
                typename DefaultEpilogue::OutputTileIterator,
                typename DefaultEpilogue::AccumulatorFragmentIterator,
                typename DefaultEpilogue::WarpTileIterator,
                typename DefaultEpilogue::SharedLoadIterator,
                EpilogueOutputOp,
                typename DefaultEpilogue::Padding,
                DefaultEpilogue::kFragmentsPerIteration,
                true // IterationsUnroll
                >;
            EpilogueOutputOp rescale({1, 1});
            Epilogue epilogue(
                shared_storage.after_qk.mm_gradV.epilogue,
                thread_id,
                warp_id,
                lane_id);
            epilogue(rescale, output_write_it, accum, output_read_it);
          }));
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // MatmulDOIVJ
    /////////////////////////////////////////////////////////////////////////////////////////////////
    {
      using Mma = typename MatmulDOIVJ::Mma;

      cutlass::gemm::GemmCoord problem_size(
          skipBoundsChecks ? MatmulDOIVJ::ThreadblockShape::kM
                           : p.num_queries - query_start,
          skipBoundsChecks ? MatmulDOIVJ::ThreadblockShape::kN
                           : p.num_keys - key_start,
          p.head_dim_value // k
      );

      // do_i
      typename Mma::IteratorA iterator_A(
          {int32_t(p.head_dim_value)},
          p.grad_output_ptr + query_start * p.head_dim_value,
          {problem_size.m(), problem_size.k()},
          thread_id,
          no_offset);

      // v_j.transpose(-2, -1)
      typename Mma::IteratorB iterator_B(
          {int32_t(p.head_dim_value)},
          p.value_ptr + key_start * p.head_dim_value,
          {problem_size.k(), problem_size.n()},
          thread_id,
          no_offset);

      Mma mma(shared_storage.after_qk.mm_doivj, thread_id, warp_id, lane_id);

      typename Mma::FragmentC accum;

      accum.clear();

      auto gemm_k_iterations =
          (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // Compute threadblock-scoped matrix multiply-add
      mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);

      int warp_idx_mn_0 =
          warp_id % (Mma::Base::WarpCount::kM * Mma::Base::WarpCount::kN);
      auto output_tile_coords = cutlass::MatrixCoord{
          warp_idx_mn_0 % Mma::Base::WarpCount::kM,
          warp_idx_mn_0 / Mma::Base::WarpCount::kM};
      // TODO: This must be terribly inefficient. There must be a better way
      // tmp [RF] <- (accum [RF] - Di [smem] ) * attn_T.T [smem]
      // attn_shared_storage  [smem] <- tmp.T
      // doivj_shared_storage [smem] <- tmp
      {
        using RegistersIter = typename DefaultAttentionScalingCoefsUpdater<
            typename Mma::Operator::IteratorC,
            typename MatmulDOIVJ::DefaultMma::MmaCore::ElementC,
            kWarpSize>::Updater;
        auto lane_offset = RegistersIter::get_lane_offset(
            lane_id, warp_id, output_tile_coords);
        auto attn_T = shared_storage.after_qk.attn_shared_storage.accum_ref();
        accum_t current_di;
        typename Mma::FragmentC fragment_attn, fragment_di, fragment_pos;
        RegistersIter::iterateRows(
            lane_offset,
            [&](int accum_m) { current_di = shared_storage.di[accum_m]; },
            [&](int accum_m, int accum_n, int idx) {
              // TODO: Otherwise we can get nans as we
              // might have infs here (only seen on f16 tho)
              if (skipBoundsChecks ||
                  (accum_m < problem_size.m() && accum_n < problem_size.n())) {
                fragment_attn[idx] = attn_T.at({accum_n, accum_m});
              } else {
                fragment_attn[idx] = 0;
              }
              fragment_di[idx] = current_di;
              fragment_pos[idx] = 100 * accum_m + accum_n;
            },
            [&](int accum_m) {

            });
        accum = (accum - fragment_di) * fragment_attn * scale;
        __syncthreads();
        // attn <- attn_T.T
        RegistersIter::iterateRows(
            lane_offset,
            [&](int accum_m) {},
            [&](int accum_m, int accum_n, int idx) {
              // How does this even work?! We need to change the layout
              attn_T.at({accum_n, accum_m}) = scalar_t(accum[idx]);
            },
            [&](int accum_m) {});
      }

      MatmulDOIVJ::B2bGemm::accumToSmem(
          shared_storage.after_qk.after_doivj.doivj_shared_storage,
          accum,
          lane_id,
          output_tile_coords);
      __syncthreads();
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // GradQ matmul
    //
    // grad_q[i_start:i_end] += tmp @ k_j
    /////////////////////////////////////////////////////////////////////////////////////////////////
    for (int col = 0; col < p.head_dim;
         col += MatmulGradQ::ThreadblockShape::kN) {
      using Mma = typename MatmulGradQ::Mma;

      cutlass::gemm::GemmCoord problem_size(
          skipBoundsChecks
              ? MatmulGradQ::ThreadblockShape::kM
              : std::min((int32_t)Mma::Shape::kM, p.num_queries - query_start),
          false ? MatmulGradQ::ThreadblockShape::kN : p.head_dim - col,
          skipBoundsChecks
              ? MatmulQK::Mma::Shape::kM
              : std::min(
                    (int32_t)MatmulQK::Mma::Shape::kM, p.num_keys - key_start));

      // k_j
      typename Mma::IteratorB iterator_B(
          {int32_t(p.head_dim)},
          p.key_ptr + key_start * p.head_dim + col,
          {problem_size.k(), problem_size.n()},
          thread_id,
          no_offset);

      Mma mma(
          shared_storage.after_qk.after_doivj.mm_gradQ.mm,
          shared_storage.after_qk.after_doivj.doivj_shared_storage,
          thread_id,
          warp_id,
          lane_id,
          problem_size.k());

      typename Mma::FragmentC accum;

      accum.clear();

      auto gemm_k_iterations =
          (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // Compute threadblock-scoped matrix multiply-add
      __syncthreads();
      mma(gemm_k_iterations, accum, iterator_B, accum);
      __syncthreads();

      // Output results
      typename MatmulGradQ::OutputTileIterator output_read_it(
          typename MatmulGradQ::OutputTileIterator::Params{p.head_dim},
          p.grad_query_ptr + query_start * p.head_dim + col,
          {problem_size.m(), problem_size.n()},
          thread_id);
      typename MatmulGradQ::OutputTileIterator output_write_it = output_read_it;

      DISPATCH_BOOL(
          key_start == 0, kIsFirst, ([&]() {
            using DefaultEpilogue = typename MatmulGradQ::DefaultEpilogue;
            using DefaultOutputOp = typename MatmulGradQ::DefaultOutputOp;
            static constexpr auto ScaleType = kIsFirst
                ? cutlass::epilogue::thread::ScaleType::Nothing
                : cutlass::epilogue::thread::ScaleType::NoBetaScaling;
            using EpilogueOutputOp =
                typename cutlass::epilogue::thread::LinearCombination<
                    typename DefaultOutputOp::ElementOutput,
                    DefaultOutputOp::kCount,
                    typename DefaultOutputOp::ElementAccumulator,
                    typename DefaultOutputOp::ElementCompute,
                    ScaleType>;
            using Epilogue = typename cutlass::epilogue::threadblock::Epilogue<
                typename DefaultEpilogue::Shape,
                typename Mma::Operator,
                DefaultEpilogue::kPartitionsK,
                typename DefaultEpilogue::OutputTileIterator,
                typename DefaultEpilogue::AccumulatorFragmentIterator,
                typename DefaultEpilogue::WarpTileIterator,
                typename DefaultEpilogue::SharedLoadIterator,
                EpilogueOutputOp,
                typename DefaultEpilogue::Padding,
                DefaultEpilogue::kFragmentsPerIteration,
                true // IterationsUnroll
                >;
            EpilogueOutputOp rescale({1, 1});
            Epilogue epilogue(
                shared_storage.after_qk.after_doivj.mm_gradQ.epilogue,
                thread_id,
                warp_id,
                lane_id);
            epilogue(rescale, output_write_it, accum, output_read_it);
          }));
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // GradK matmul
    //
    // grad_k[i_start:i_end] += tmp.transpose(-2, -1) @ q_i
    /////////////////////////////////////////////////////////////////////////////////////////////////
    for (int col = 0; col < p.head_dim;
         col += MatmulGradK::ThreadblockShape::kN) {
      using Mma = typename MatmulGradK::Mma;

      cutlass::gemm::GemmCoord problem_size(
          skipBoundsChecks
              ? MatmulGradK::ThreadblockShape::kM
              : std::min((int32_t)Mma::Shape::kM, p.num_keys - key_start),
          false ? MatmulGradK::ThreadblockShape::kN : p.head_dim - col,
          skipBoundsChecks ? MatmulQK::Mma::Shape::kN
                           : std::min(
                                 (int32_t)MatmulQK::Mma::Shape::kN,
                                 p.num_queries - query_start));

      // q_i
      typename Mma::IteratorB iterator_B(
          {int32_t(p.head_dim)},
          p.query_ptr + query_start * p.head_dim + col,
          {problem_size.k(), problem_size.n()},
          thread_id,
          no_offset);

      Mma mma(
          shared_storage.after_qk.after_doivj.mm_gradK.mm,
          shared_storage.after_qk.attn_shared_storage, // storing tmp.T
          thread_id,
          warp_id,
          lane_id,
          problem_size.k());

      typename Mma::FragmentC accum;

      accum.clear();

      auto gemm_k_iterations =
          (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // Compute threadblock-scoped matrix multiply-add
      __syncthreads();
      mma(gemm_k_iterations, accum, iterator_B, accum);
      __syncthreads();

      // Output results
      typename MatmulGradK::OutputTileIterator output_read_it(
          typename MatmulGradK::OutputTileIterator::Params{p.head_dim},
          p.grad_key_ptr + key_start * p.head_dim + col,
          {skipBoundsChecks
               ? MatmulGradK::ThreadblockShape::kM
               : std::min((int32_t)Mma::Shape::kM, p.num_keys - key_start),
           false ? MatmulGradK::ThreadblockShape::kN : p.head_dim - col},
          thread_id);
      typename MatmulGradK::OutputTileIterator output_write_it = output_read_it;

      DISPATCH_BOOL(
          query_start == 0, kIsFirst, ([&]() {
            using DefaultEpilogue = typename MatmulGradK::DefaultEpilogue;
            using DefaultOutputOp = typename MatmulGradK::DefaultOutputOp;
            static constexpr auto ScaleType = kIsFirst
                ? cutlass::epilogue::thread::ScaleType::Nothing
                : cutlass::epilogue::thread::ScaleType::NoBetaScaling;
            using EpilogueOutputOp =
                typename cutlass::epilogue::thread::LinearCombination<
                    typename DefaultOutputOp::ElementOutput,
                    DefaultOutputOp::kCount,
                    typename DefaultOutputOp::ElementAccumulator,
                    typename DefaultOutputOp::ElementCompute,
                    ScaleType>;
            using Epilogue = typename cutlass::epilogue::threadblock::Epilogue<
                typename DefaultEpilogue::Shape,
                typename Mma::Operator,
                DefaultEpilogue::kPartitionsK,
                typename DefaultEpilogue::OutputTileIterator,
                typename DefaultEpilogue::AccumulatorFragmentIterator,
                typename DefaultEpilogue::WarpTileIterator,
                typename DefaultEpilogue::SharedLoadIterator,
                EpilogueOutputOp,
                typename DefaultEpilogue::Padding,
                DefaultEpilogue::kFragmentsPerIteration,
                true // IterationsUnroll
                >;
            EpilogueOutputOp rescale({1, 1});
            Epilogue epilogue(
                shared_storage.after_qk.after_doivj.mm_gradK.epilogue,
                thread_id,
                warp_id,
                lane_id);
            epilogue(rescale, output_write_it, accum, output_read_it);
          }));
    }
  }

  static __device__ __forceinline__ int8_t get_lane_id() {
    return threadIdx.x;
  }
  static __device__ __forceinline__ int8_t get_warp_id() {
    return threadIdx.y;
  }
  static __device__ __forceinline__ int16_t get_thread_id() {
    return threadIdx.x + threadIdx.y * blockDim.x;
  }
};

template <typename AK>
__global__ void __launch_bounds__(AK::kNumThreads, AK::kMinBlocksPerSm)
    attention_kernel_backward_batched(typename AK::Params params) {
#ifndef __CUDA_ARCH__
  using CurrentArch = cutlass::arch::Sm80;
#elif (__CUDA_ARCH__ >= 800)
  using CurrentArch = cutlass::arch::Sm80;
#elif (__CUDA_ARCH__ >= 750)
  using CurrentArch = cutlass::arch::Sm75;
#elif (__CUDA_ARCH__ >= 700)
  using CurrentArch = cutlass::arch::Sm70;
#elif (__CUDA_ARCH__ >= 500)
  using CurrentArch = cutlass::arch::Sm50;
#else
#error "Unsupported architecture in __CUDA_ARCH__"
#endif

  auto batch_id = blockIdx.z;
  params.advance_batches(batch_id);
  AK::kernel(params);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
mem_efficient_attention_backward_generic(
    const at::Tensor& grad_out_,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& logsumexp,
    const at::Tensor& out,
    const c10::optional<at::Tensor>& attn_bias_,
    double p,
    int64_t rng_seed,
    int64_t rng_offset) {
  TORCH_CHECK(query.dim() == grad_out_.dim());
  TORCH_CHECK(query.dim() == key.dim());
  TORCH_CHECK(query.dim() == 3);

  TORCH_CHECK(query.size(0) == grad_out_.size(0));
  TORCH_CHECK(query.size(1) == grad_out_.size(1));
  TORCH_CHECK(query.size(2) == grad_out_.size(2));

  TORCH_CHECK(query.size(2) == key.size(2));
  TORCH_CHECK(query.size(0) == key.size(0));

  TORCH_CHECK(query.size(0) == value.size(0));
  TORCH_CHECK(key.size(1) == value.size(1));

  at::Tensor attn_bias;
  if (attn_bias_.has_value()) {
    attn_bias = *attn_bias_;
    TORCH_CHECK(query.dim() == attn_bias.dim());
    TORCH_CHECK(query.size(0) == attn_bias.size(0));
    TORCH_CHECK(query.size(1) == attn_bias.size(1));
    TORCH_CHECK(key.size(1) == attn_bias.size(2));
    TORCH_CHECK(attn_bias.stride(1) == 0);
  }

  TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
  TORCH_CHECK(key.is_cuda(), "key must be a CUDA tensor");
  TORCH_CHECK(value.is_cuda(), "value must be a CUDA tensor");
  TORCH_CHECK(grad_out_.is_cuda(), "grad_out must be a CUDA tensor");

  TORCH_CHECK(!query.is_sparse(), "query must be a dense tensor");
  TORCH_CHECK(!key.is_sparse(), "key must be a dense tensor");
  TORCH_CHECK(!value.is_sparse(), "value must be a dense tensor");
  TORCH_CHECK(!grad_out_.is_sparse(), "grad_out must be a dense tensor");

  // TODO drop this limitation in the future
  TORCH_CHECK(query.is_contiguous());
  TORCH_CHECK(key.is_contiguous());
  TORCH_CHECK(value.is_contiguous());

  // TODO: support other dtypes in the future
  // TORCH_CHECK(
  //     query.scalar_type() == at::ScalarType::Half,
  //     "Only f16 type is supported for now");

  at::cuda::CUDAGuard device_guard(query.device());

  // handle potentially non-contiguous grad_out through a copy
  auto grad_out = grad_out_.contiguous();

  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);
  int64_t K = query.size(2);

  at::Tensor grad_q = at::zeros_like(query);
  at::Tensor grad_k = at::zeros_like(key);
  at::Tensor grad_v = at::zeros_like(value);

  cudaDeviceProp* properties =
      at::cuda::getDeviceProperties(query.device().index());
  const int computeCapability = properties->major * 10 + properties->minor;
// set at compile time to avoid having to generate host code for all kernels
// eg -DATTN_ONLY_BUILD_ARCH=Sm80
#ifdef ATTN_ONLY_BUILD_ARCH
#define DISPATCH_ARCHTAG(func)                         \
  using ArchTag = cutlass::arch::ATTN_ONLY_BUILD_ARCH; \
  func();
#else
#define DISPATCH_ARCHTAG(func)                                            \
  {                                                                       \
    if (computeCapability >= 80) {                                        \
      using ArchTag = cutlass::arch::Sm80;                                \
      func();                                                             \
    } else if (computeCapability >= 75) {                                 \
      using ArchTag = cutlass::arch::Sm75;                                \
      func();                                                             \
    } else if (computeCapability >= 70) {                                 \
      using ArchTag = cutlass::arch::Sm70;                                \
      func();                                                             \
    } else if (computeCapability >= 50) {                                 \
      using ArchTag = cutlass::arch::Sm50;                                \
      func();                                                             \
    } else {                                                              \
      TORCH_CHECK(                                                        \
          false,                                                          \
          "Your device is too old. We require compute capability >= 50"); \
    }                                                                     \
  }
#endif

#define DISPATCH_TYPES(func)                                          \
  {                                                                   \
    if (query.scalar_type() == at::ScalarType::Float) {               \
      using scalar_t = float;                                         \
      func();                                                         \
    } else if (query.scalar_type() == at::ScalarType::Half) {         \
      using scalar_t = cutlass::half_t;                               \
      func();                                                         \
    } else {                                                          \
      TORCH_CHECK(false, "Only fp32 & half supported at the moment"); \
    }                                                                 \
  }

  DISPATCH_TYPES(([&]() {
    bool isAligned;
    DISPATCH_ARCHTAG(([&]() {
      using AlignedAK = AttentionBackwardKernel<scalar_t, true, ArchTag>;
      isAligned =
          (query.stride(1) % AlignedAK::kOptimalAlignement == 0 &&
           key.stride(1) % AlignedAK::kOptimalAlignement == 0 &&
           value.stride(1) % AlignedAK::kOptimalAlignement == 0);
      // TODO: Should we warn or log somewhere when we use a less efficient
      // kernel due to wrong alignment?

      DISPATCH_BOOL(
          isAligned, kIsAligned, ([&]() {
            using AK = AttentionBackwardKernel<scalar_t, kIsAligned, ArchTag>;
            size_t smem_bytes = sizeof(typename AK::SharedStorage);
            // Might happen on Sm80/half, where the minimum alignment is 32bits
            TORCH_CHECK(
                query.stride(1) % AK::kMinimumAlignment == 0,
                "query is not correctly aligned");
            TORCH_CHECK(
                key.stride(1) % AK::kMinimumAlignment == 0,
                "key is not correctly aligned");
            TORCH_CHECK(
                value.stride(1) % AK::kMinimumAlignment == 0,
                "value is not correctly aligned");

            AK::Params params;
            params.query_ptr = (scalar_t*)query.data_ptr();
            params.key_ptr = (scalar_t*)key.data_ptr();
            params.value_ptr = (scalar_t*)value.data_ptr();
            params.logsumexp_ptr =
                (typename AK::lse_scalar_t*)logsumexp.data_ptr();
            params.output_ptr = (scalar_t*)out.data_ptr();
            params.grad_output_ptr = (scalar_t*)grad_out.data_ptr();
            params.grad_query_ptr = (scalar_t*)grad_q.data_ptr();
            params.grad_key_ptr = (scalar_t*)grad_k.data_ptr();
            params.grad_value_ptr = (scalar_t*)grad_v.data_ptr();
            params.head_dim = query.size(2);
            params.head_dim_value = value.size(2);
            params.num_queries = query.size(1);
            params.num_keys = key.size(1);

            dim3 grid(AK::getNumBlocksX(N), AK::getNumBlocksY(M), B);
            dim3 block(AK::kWarpSize, AK::kNumWarpsPerBlock, 1);
            constexpr auto kernel_fn = attention_kernel_backward_batched<AK>;

            if (smem_bytes > 0xc000) {
              TORCH_INTERNAL_ASSERT(
                  computeCapability >= 70,
                  "This kernel requires too much shared memory on this machine!");
              cudaFuncSetAttribute(
                  kernel_fn,
                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                  smem_bytes);
            }

            kernel_fn<<<grid, block, smem_bytes>>>(params);
            AT_CUDA_CHECK(cudaGetLastError());
          }));
    }));
  }));
  return std::make_tuple(grad_q, grad_k, grad_v);
}

} // namespace

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::efficient_attention_backward_generic"),
      TORCH_FN(mem_efficient_attention_backward_generic));
}
