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
// For half types
template <typename S>
struct WarpsPerSm {
  static constexpr int value = 12;
};
// Specialization for fp32
template <>
struct WarpsPerSm<float> {
  static constexpr int value = 8;
};

template <typename scalar_t_, bool kIsAligned_>
struct AttentionBackwardKernelInfo {
  using scalar_t = scalar_t_;
  using output_t = scalar_t;
  using lse_scalar_t = float;
  using accum_t = float;
  static constexpr bool kIsAligned = kIsAligned_;

  struct BatchedParams {
    at::PackedTensorAccessor32<scalar_t, 3> query;
    at::PackedTensorAccessor32<scalar_t, 3> key;
    at::PackedTensorAccessor32<scalar_t, 3> value;
    at::PackedTensorAccessor32<lse_scalar_t, 2> logsumexp;
    at::PackedTensorAccessor32<scalar_t, 3> output;
    at::PackedTensorAccessor32<scalar_t, 3> grad_output;

    // Outputs
    at::PackedTensorAccessor32<scalar_t, 3> grad_query;
    at::PackedTensorAccessor32<scalar_t, 3> grad_key;
    at::PackedTensorAccessor32<scalar_t, 3> grad_value;
  };

  // Blocks & grid
  static constexpr int64_t kWarpSize = 32;
  static constexpr int64_t kNumWarpsPerBlock = 4;
  static constexpr int64_t kBlockSizeI = 64;
  static constexpr int64_t kBlockSizeJ = 64;

  // Launch bounds
  static constexpr int64_t kNumThreads = kWarpSize * kNumWarpsPerBlock;
  static constexpr int64_t kMinBlocksPerSm =
      WarpsPerSm<scalar_t>::value / kNumWarpsPerBlock;

  static int64_t getNumBlocksY(int64_t num_queries) {
    return 1; // ceil_div(num_queries, kBlockSizeI);
  }

  static int64_t getNumBlocksX(int64_t num_values) {
    return 1; // ceil_div(num_values, kBlockSizeJ);
  }
};

template <typename AKI, typename ArchTag>
struct AttentionBackwardKernel {
  using scalar_t = typename AKI::scalar_t;
  using output_t = typename AKI::output_t;
  using lse_scalar_t = typename AKI::lse_scalar_t;
  using accum_t = typename AKI::accum_t;

  static constexpr bool kIsAligned = AKI::kIsAligned;

  static constexpr int64_t kWarpSize = AKI::kWarpSize;
  static constexpr int64_t kNumWarpsPerBlock = AKI::kNumWarpsPerBlock;
  static constexpr int64_t kBlockSizeI = AKI::kBlockSizeI;
  static constexpr int64_t kBlockSizeJ = AKI::kBlockSizeJ;

  struct Params {
    at::TensorAccessor<scalar_t, 2, at::DefaultPtrTraits, int32_t> query;
    at::TensorAccessor<scalar_t, 2, at::DefaultPtrTraits, int32_t> key;
    at::TensorAccessor<scalar_t, 2, at::DefaultPtrTraits, int32_t> value;
    at::TensorAccessor<lse_scalar_t, 1, at::DefaultPtrTraits, int32_t>
        logsumexp;
    at::TensorAccessor<scalar_t, 2, at::DefaultPtrTraits, int32_t> output;
    at::TensorAccessor<scalar_t, 2, at::DefaultPtrTraits, int32_t> grad_output;

    // Outputs
    at::TensorAccessor<scalar_t, 2, at::DefaultPtrTraits, int32_t> grad_query;
    at::TensorAccessor<scalar_t, 2, at::DefaultPtrTraits, int32_t> grad_key;
    at::TensorAccessor<scalar_t, 2, at::DefaultPtrTraits, int32_t> grad_value;
  };

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
      Params& p,
      int32_t query_start) {
    __syncthreads();
    using AccessType = cutlass::Array<scalar_t, kElementsPerAccess>;
    static constexpr int kNumThreadsPerLine = 4;
    static constexpr int kParallelRowsPerWarp = kWarpSize / kNumThreadsPerLine;

    int32_t laneCol = (get_lane_id() % kNumThreadsPerLine);
    int32_t laneRow = (get_lane_id() / kNumThreadsPerLine) +
        get_warp_id() * kBlockSizeI / kNumWarpsPerBlock;

    int32_t dO_s0 = p.grad_output.stride(0) / AccessType::kElements;
    int32_t out_s0 = p.output.stride(0) / AccessType::kElements;
    cutlass::
        Array<accum_t, kBlockSizeI / kParallelRowsPerWarp / kNumWarpsPerBlock>
            di_frag;
    di_frag.clear();
    assert(p.output.stride(0) % AccessType::kElements == 0);
    CUTLASS_PRAGMA_UNROLL
    for (int firstCol = 0; firstCol < p.output.size(1);
         firstCol += kNumThreadsPerLine * AccessType::kElements) {
      const __restrict__ AccessType* dO = reinterpret_cast<AccessType*>(
          p.grad_output[query_start + laneRow].data() + firstCol);
      const __restrict__ AccessType* out = reinterpret_cast<AccessType*>(
          p.output[query_start + laneRow].data() + firstCol);
      int32_t rowEnd = (p.grad_output.size(0) - query_start);
      int32_t colEnd = p.grad_output.size(1) / AccessType::kElements;

      AccessType frag_dO;
      AccessType frag_out;
      AccessType result;
      frag_dO.clear();
      frag_out.clear();
      dO += laneCol;
      out += laneCol;

      bool withinBounds =
          firstCol + laneCol * AccessType::kElements < p.grad_output.size(1);

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
      Params& p,
      int32_t query_start) {
    constexpr int kOptimalElements =
        128 / cutlass::sizeof_bits<scalar_t>::value;
    if (p.output.size(1) % kOptimalElements == 0) {
      _computeDi<kOptimalElements>(di, p, query_start);
    } else {
      _computeDi<1>(di, p, query_start);
    }
  }

  static __device__ void kernel(Params& p) {
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

      static constexpr int64_t kWarpSize = AKI::kWarpSize;
      static constexpr int64_t kNumWarpsPerBlock = AKI::kNumWarpsPerBlock;
      for (int i = 0; i < sizeof(SharedStorage) / sizeof(uint32_t) -
               kWarpSize * kNumWarpsPerBlock;
           i += kWarpSize * kNumWarpsPerBlock) {
        smem[i + thread_id] = 0;
      }
    };

    int32_t query_end = p.query.size(0) / kBlockSizeI * kBlockSizeI;
    int32_t key_end = p.key.size(0) / kBlockSizeJ * kBlockSizeJ;
    int32_t query_start = 0;
    for (; query_start < query_end; query_start += kBlockSizeI) {
      clearSmem();
      computeDi(shared_storage.di, p, query_start);
      int32_t key_start = 0;
      for (; key_start < key_end; key_start += kBlockSizeJ) {
        processBlockIJ<true>(shared_storage, p, query_start, key_start);
      }
      // last (partial) key
      if (key_start != p.key.size(0)) {
        processBlockIJ<false>(shared_storage, p, query_start, key_start);
      }
    }
    // Last (partial) query block
    if (query_start != p.query.size(0)) {
      computeDi(shared_storage.di, p, query_start);
      for (int32_t key_start = 0; key_start < p.key.size(0);
           key_start += kBlockSizeJ) {
        processBlockIJ<false>(shared_storage, p, query_start, key_start);
      }
    }
  }

  // Compute threadblock location
  template <bool skipBoundsChecks>
  static __device__ __forceinline__ void processBlockIJ(
      SharedStorage& shared_storage,
      Params& p,
      int32_t query_start,
      int32_t key_start) {
    cutlass::MatrixCoord no_offset{0, 0};
    accum_t scale = accum_t(1.0 / std::sqrt(float(p.query.size(1))));
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
                           : p.key.size(0) - key_start,
          skipBoundsChecks ? MatmulQK::ThreadblockShape::kN
                           : p.query.size(0) - query_start,
          p.key.size(1) // k
      );

      // k_j
      typename Mma::IteratorA iterator_A(
          {p.key.stride(0)},
          &p.key[key_start][0],
          {problem_size.m(), problem_size.k()},
          thread_id,
          no_offset);

      // q_i.transpose(-2, -1)
      typename Mma::IteratorB iterator_B(
          {p.query.stride(0)},
          &p.query[query_start][0],
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
          &p.logsumexp[query_start],
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
    for (int col = 0; col < p.grad_output.size(1);
         col += MatmulGradV::ThreadblockShape::kN) {
      using Mma = typename MatmulGradV::Mma;

      cutlass::gemm::GemmCoord problem_size(
          skipBoundsChecks ? MatmulGradV::ThreadblockShape::kM
                           : p.grad_value.size(0) - key_start,
          p.grad_output.size(1) - col,
          skipBoundsChecks ? MatmulQK::Mma::Shape::kN
                           : std::min(
                                 (int32_t)MatmulQK::Mma::Shape::kN,
                                 p.grad_output.size(0) - query_start));

      // q_i.transpose(-2, -1)
      typename Mma::IteratorB iterator_B(
          {p.grad_output.stride(0)},
          &p.grad_output[query_start][col],
          {problem_size.k(), problem_size.n()},
          thread_id,
          no_offset);

      Mma mma(
          shared_storage.after_qk.mm_gradV.mm,
          shared_storage.after_qk.attn_shared_storage,
          thread_id,
          warp_id,
          lane_id);

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
          typename MatmulGradV::OutputTileIterator::Params{
              p.grad_value.stride(0)},
          &p.grad_value[key_start][col],
          {skipBoundsChecks ? MatmulGradV::ThreadblockShape::kM
                            : p.grad_value.size(0) - key_start,
           p.grad_output.size(1) - col},
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
                           : p.grad_output.size(0) - query_start,
          skipBoundsChecks ? MatmulDOIVJ::ThreadblockShape::kN
                           : p.value.size(0) - key_start,
          p.value.size(1) // k
      );

      // do_i
      typename Mma::IteratorA iterator_A(
          {p.grad_output.stride(0)},
          &p.grad_output[query_start][0],
          {problem_size.m(), problem_size.k()},
          thread_id,
          no_offset);

      // v_j.transpose(-2, -1)
      typename Mma::IteratorB iterator_B(
          {p.value.stride(0)},
          &p.value[key_start][0],
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
    for (int col = 0; col < p.key.size(1);
         col += MatmulGradQ::ThreadblockShape::kN) {
      using Mma = typename MatmulGradQ::Mma;

      cutlass::gemm::GemmCoord problem_size(
          skipBoundsChecks ? MatmulGradQ::ThreadblockShape::kM
                           : std::min(
                                 (int32_t)Mma::Shape::kM,
                                 p.grad_query.size(0) - query_start),
          false ? MatmulGradQ::ThreadblockShape::kN : p.key.size(1) - col,
          skipBoundsChecks ? MatmulQK::Mma::Shape::kM
                           : std::min(
                                 (int32_t)MatmulQK::Mma::Shape::kM,
                                 p.key.size(0) - key_start));

      // k_j
      typename Mma::IteratorB iterator_B(
          {p.key.stride(0)},
          &p.key[key_start][col],
          {problem_size.k(), problem_size.n()},
          thread_id,
          no_offset);

      Mma mma(
          shared_storage.after_qk.after_doivj.mm_gradQ.mm,
          shared_storage.after_qk.after_doivj.doivj_shared_storage,
          thread_id,
          warp_id,
          lane_id);

      typename Mma::FragmentC accum;

      accum.clear();

      auto gemm_k_iterations =
          (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // Compute threadblock-scoped matrix multiply-add
      mma(gemm_k_iterations, accum, iterator_B, accum);

      // Output results
      typename MatmulGradQ::OutputTileIterator output_read_it(
          typename MatmulGradQ::OutputTileIterator::Params{
              p.grad_query.stride(0)},
          &p.grad_query[query_start][col],
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
    for (int col = 0; col < p.query.size(1);
         col += MatmulGradK::ThreadblockShape::kN) {
      using Mma = typename MatmulGradK::Mma;

      cutlass::gemm::GemmCoord problem_size(
          skipBoundsChecks
              ? MatmulGradK::ThreadblockShape::kM
              : std::min(
                    (int32_t)Mma::Shape::kM, p.grad_key.size(0) - key_start),
          false ? MatmulGradK::ThreadblockShape::kN : p.query.size(1) - col,
          skipBoundsChecks ? MatmulQK::Mma::Shape::kN
                           : std::min(
                                 (int32_t)MatmulQK::Mma::Shape::kN,
                                 p.query.size(0) - query_start));

      // q_i
      typename Mma::IteratorB iterator_B(
          {p.query.stride(0)},
          &p.query[query_start][col],
          {problem_size.k(), problem_size.n()},
          thread_id,
          no_offset);

      Mma mma(
          shared_storage.after_qk.after_doivj.mm_gradK.mm,
          shared_storage.after_qk.attn_shared_storage, // storing tmp.T
          thread_id,
          warp_id,
          lane_id);

      typename Mma::FragmentC accum;

      accum.clear();

      auto gemm_k_iterations =
          (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // Compute threadblock-scoped matrix multiply-add
      mma(gemm_k_iterations, accum, iterator_B, accum);

      // Output results
      typename MatmulGradK::OutputTileIterator output_read_it(
          typename MatmulGradK::OutputTileIterator::Params{
              p.grad_key.stride(0)},
          &p.grad_key[key_start][col],
          {skipBoundsChecks
               ? MatmulGradK::ThreadblockShape::kM
               : std::min(
                     (int32_t)Mma::Shape::kM, p.grad_key.size(0) - key_start),
           false ? MatmulGradK::ThreadblockShape::kN
                 : p.grad_key.size(1) - col},
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

template <typename AttentionBackwardKernelInfo>
__global__ void __launch_bounds__(
    AttentionBackwardKernelInfo::kNumThreads,
    AttentionBackwardKernelInfo::kMinBlocksPerSm)
    attention_kernel_backward_batched(
        typename AttentionBackwardKernelInfo::BatchedParams batched_params) {
  auto batch_id = blockIdx.z;

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

  using AK = AttentionBackwardKernel<AttentionBackwardKernelInfo, CurrentArch>;
  typename AK::Params params{
      batched_params.query[batch_id],
      batched_params.key[batch_id],
      batched_params.value[batch_id],
      batched_params.logsumexp[batch_id],
      batched_params.output[batch_id],
      batched_params.grad_output[batch_id],
      // Outputs
      batched_params.grad_query[batch_id],
      batched_params.grad_key[batch_id],
      batched_params.grad_value[batch_id]};
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
  TORCH_CHECK(
      query.size(2) ==
      value.size(2)); // TODO: drop this limitation in the future

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
      using AlignedAK = AttentionBackwardKernel<
          AttentionBackwardKernelInfo<scalar_t, true>,
          ArchTag>;
      isAligned =
          (query.stride(1) % AlignedAK::kOptimalAlignement == 0 &&
           key.stride(1) % AlignedAK::kOptimalAlignement == 0 &&
           value.stride(1) % AlignedAK::kOptimalAlignement == 0);
      // TODO: Should we warn or log somewhere when we use a less efficient
      // kernel due to wrong alignment?
    }));
    DISPATCH_BOOL(
        isAligned, kIsAligned, ([&]() {
          using AKI = AttentionBackwardKernelInfo<scalar_t, kIsAligned>;

          size_t smem_bytes = 0;
          DISPATCH_ARCHTAG(([&]() {
            using AK = AttentionBackwardKernel<AKI, ArchTag>;
            smem_bytes = sizeof(typename AK::SharedStorage);
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
          }));

          using m = TypeTraits<scalar_t>;

          AKI::BatchedParams params{
              m::packed_accessor<3>(query),
              m::packed_accessor<3>(key),
              m::packed_accessor<3>(value),
              TypeTraits<AKI::lse_scalar_t>::packed_accessor<2>(logsumexp),
              m::packed_accessor<3>(out),
              m::packed_accessor<3>(grad_out),

              // Outputs
              m::packed_accessor<3>(grad_q),
              m::packed_accessor<3>(grad_k),
              m::packed_accessor<3>(grad_v)};

          dim3 grid(AKI::getNumBlocksX(N), AKI::getNumBlocksY(M), B);
          dim3 block(AKI::kWarpSize, AKI::kNumWarpsPerBlock, 1);
          constexpr auto kernel_fn = attention_kernel_backward_batched<AKI>;

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
  return std::make_tuple(grad_q, grad_k, grad_v);
}

} // namespace

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::efficient_attention_backward_generic"),
      TORCH_FN(mem_efficient_attention_backward_generic));
}
