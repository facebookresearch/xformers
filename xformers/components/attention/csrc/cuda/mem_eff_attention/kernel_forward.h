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

#include "attention_scaling_coefs_updater.h"
#include "cutlass/epilogue/threadblock/default_epilogue_simt.h"
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/platform/platform.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "debug_utils.h"
#include "epilogue_rescale_output.h"
#include "find_default_mma.h"
#include "gemm_kernel_utils.h"
#include "mma_from_smem.h"

#include <inttypes.h>

using namespace gemm_kernel_utils;

namespace {
template <typename scalar_t, typename Arch>
constexpr int getWarpsPerSm() {
  bool is_half = !std::is_same<scalar_t, float>::value;
  if (Arch::kMinComputeCapability >= 80) {
    return is_half ? 16 : 12;
  }
  return 12;
}
} // namespace

template <
    // The datatype of Q/K/V
    typename scalar_t_,
    // Architecture we are targeting (eg `cutlass::arch::Sm80`)
    typename ArchTag,
    // If Q/K/V are correctly aligned in memory and we can run a fast kernel
    bool isAligned_,
    int64_t kQueriesPerBlock,
    int64_t kKeysPerBlock,
    typename output_t_ = float>
struct AttentionKernel {
  using scalar_t = scalar_t_;
  using accum_t = float;
  using lse_scalar_t = float;
  using output_t = output_t_;
  static constexpr bool kIsAligned = isAligned_;
  static constexpr int32_t kAlignLSE = 32; // block size of backward

  static_assert(kQueriesPerBlock % 32 == 0);
  static_assert(kKeysPerBlock % 32 == 0);
  static constexpr int64_t kNumWarpsPerBlock =
      kQueriesPerBlock * kKeysPerBlock / (32 * 32);
  static constexpr int64_t kWarpSize = 32;

  // Launch bounds
  static constexpr int64_t kNumThreads = kWarpSize * kNumWarpsPerBlock;
  static constexpr int64_t kMinBlocksPerSm =
      getWarpsPerSm<scalar_t, ArchTag>() / kNumWarpsPerBlock;

  struct Params {
    // Input tensors
    scalar_t* query_ptr; // [num_queries, head_dim]
    scalar_t* key_ptr; // [num_keys, head_dim]
    scalar_t* value_ptr; // [num_keys, head_dim_value]

    // Output tensors
    output_t* output_ptr; // [num_queries, head_dim_value]
    lse_scalar_t* logsumexp_ptr; // [num_queries] - can be 0

    // Dimensions/strides
    int32_t head_dim;
    int32_t head_dim_value;
    int32_t num_queries;
    int32_t num_keys;
    int32_t num_batches;

    __device__ void advance_batches(int32_t batch_id) {
      auto lse_dim = ceil_div((int32_t)num_queries, kAlignLSE) * kAlignLSE;

      query_ptr += batch_id * head_dim * num_queries;
      key_ptr += batch_id * head_dim * num_keys;
      value_ptr += batch_id * head_dim_value * num_keys;
      output_ptr += batch_id * head_dim_value * num_queries;
      if (logsumexp_ptr != nullptr) {
        logsumexp_ptr += batch_id * lse_dim;
      }
    }

    __host__ dim3 getBlocksGrid() const {
      return dim3(
          1, ceil_div(num_queries, (int32_t)kQueriesPerBlock), num_batches);
    }
    __host__ dim3 getThreadsGrid() const {
      return dim3(kWarpSize, kNumWarpsPerBlock, 1);
    }
  };

  struct MM0 {
    /*
      In this first matmul, we compute a block of `Q @ K.T`.
      While the calculation result is still hot in registers, we update
      `mi`, `m_prime`, `s_prime` in shared-memory, and then store this value
      into a shared-memory ("AccumulatorSharedStorage") that is used later as
      operand A for the second matmul (see MM1)
    */
    using GemmType = DefaultGemmType<ArchTag, scalar_t>;

    using OpClass = typename GemmType::OpClass;
    using DefaultConfig =
        typename cutlass::gemm::device::DefaultGemmConfiguration<
            OpClass,
            ArchTag,
            scalar_t,
            scalar_t,
            scalar_t, // ElementC
            accum_t // ElementAccumulator
            >;
    static constexpr int64_t kAlignmentA =
        kIsAligned ? DefaultConfig::kAlignmentA : GemmType::kMinimumAlignment;
    static constexpr int64_t kAlignmentB =
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment;
    using ThreadblockShape = cutlass::gemm::
        GemmShape<kQueriesPerBlock, kKeysPerBlock, GemmType::ThreadK>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, GemmType::WarpK>;
    using DefaultMma = typename cutlass::gemm::threadblock::FindDefaultMma<
        scalar_t, // ElementA,
        cutlass::layout::RowMajor, // LayoutA,
        kAlignmentA,
        scalar_t, // ElementB,
        cutlass::layout::ColumnMajor, // LayoutB,
        kAlignmentB,
        accum_t,
        cutlass::layout::RowMajor, // LayoutC,
        OpClass,
        ArchTag, // ArchTag
        ThreadblockShape, // ThreadblockShape
        WarpShape, // WarpShape
        typename GemmType::InstructionShape, // InstructionShape
        DefaultConfig::kStages, // Should use `DefaultConfig::kStages`, but that
                                // uses too much smem
        typename GemmType::Operator // Operator
        >::DefaultMma;
    using MmaCore = typename DefaultMma::MmaCore;
    using IteratorA = typename DefaultMma::IteratorA;
    using IteratorB = typename DefaultMma::IteratorB;
    using Mma = typename DefaultMma::ThreadblockMma;
    using ScalingCoefsUpdater = typename DefaultAttentionScalingCoefsUpdater<
        typename Mma::Operator::IteratorC,
        accum_t,
        kWarpSize>::Updater;

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

  struct MM1 {
    /**
      Second matmul: perform `attn @ V` where `attn` is the attention (not
      normalized) and stored in shared memory
    */
    using GemmType = DefaultGemmType<ArchTag, scalar_t>;

    using OpClass = typename GemmType::OpClass;
    using DefaultConfig =
        typename cutlass::gemm::device::DefaultGemmConfiguration<
            OpClass,
            ArchTag,
            scalar_t,
            scalar_t,
            output_t, // ElementC
            accum_t // ElementAccumulator
            >;
    static constexpr int64_t kAlignmentA =
        DefaultConfig::kAlignmentA; // from smem
    static constexpr int64_t kAlignmentB =
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment;
    using ThreadblockShape = cutlass::gemm::
        GemmShape<kQueriesPerBlock, kKeysPerBlock, GemmType::ThreadK>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, GemmType::WarpK>;
    using InstructionShape = typename GemmType::InstructionShape;

    using LayoutB = cutlass::layout::RowMajor;
    using DefaultGemm = cutlass::gemm::kernel::DefaultGemm<
        scalar_t, // ElementA,
        cutlass::layout::RowMajor, // LayoutA,
        kAlignmentA,
        scalar_t, // ElementB,
        LayoutB, // LayoutB,
        kAlignmentB,
        output_t,
        cutlass::layout::RowMajor, // LayoutC,
        accum_t,
        OpClass,
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
            typename MM0::AccumulatorSharedStorage>;
    using Mma = typename DefaultMmaFromSmem::Mma;
    using IteratorB = typename Mma::IteratorB;
    using WarpCount = typename Mma::WarpCount;

    using DefaultEpilogue = typename DefaultGemm::Epilogue;
    using OutputTileIterator = typename DefaultEpilogue::OutputTileIterator;

    struct SharedStorageMM1 {
      union {
        // Storing parts of `V` during the matmul
        typename Mma::SharedStorage mm;
        // Used by the Epilogue (so we can reuse the same memory space)
        typename DefaultEpilogue::SharedStorage epilogue;
      };
    };

    static __device__ void compute_dot_product_att_value(
        Params const& p,
        SharedStorageMM1& shared_storage_mm,
        typename MM0::AccumulatorSharedStorage& shared_storage_si,
        int32_t const& iter_key_start,
        cutlass::Array<accum_t, kQueriesPerBlock> const& m_prime,
        cutlass::Array<accum_t, kQueriesPerBlock> const& s_prime,
        bool isLast) {
      cutlass::gemm::GemmCoord problem_size(
          std::min(
              (int32_t)kQueriesPerBlock, p.num_queries - query_start()), // M
          p.head_dim_value, // N
          std::min(
              int32_t(kKeysPerBlock),
              p.num_keys - iter_key_start) // K
      );

      typename IteratorB::Params params_B(LayoutB(p.head_dim_value));

      static_assert(
          WarpCount::kM * WarpCount::kN * WarpCount::kK == kNumWarpsPerBlock);

      const int64_t nBlockN =
          ceil_div((int64_t)problem_size.n(), int64_t(ThreadblockShape::kN));
      for (int blockN = 0; blockN < nBlockN; ++blockN) {
        /*
        Run the matmul `attn @ V` for a block of attn and V.
        `attn` is read from shared memory (in `shared_storage_si`)
        `V` is read from global memory (with iterator_B)
        */
        cutlass::gemm::GemmCoord tb_tile_offset = {0, blockN, 0};

        cutlass::MatrixCoord tb_offset_B{
            tb_tile_offset.k(), tb_tile_offset.n() * Mma::Shape::kN};

        typename Mma::IteratorB iterator_B(
            params_B,
            p.value_ptr + iter_key_start * p.head_dim_value,
            {problem_size.k(), problem_size.n()},
            thread_id(),
            tb_offset_B);

        typename Mma::FragmentC accum;
        accum.clear();

        Mma mma(
            shared_storage_mm.mm,
            shared_storage_si,
            thread_id(),
            warp_id(),
            lane_id(),
            problem_size.k());

        int gemm_k_iterations =
            (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

        // Compute threadblock-scoped matrix multiply-add and store it in accum
        // (in registers)
        __syncthreads(); // we share shmem between mma and epilogue
        mma(gemm_k_iterations, accum, iterator_B, accum);
        __syncthreads(); // we share shmem between mma and epilogue

        /*
          Epilogue: Store the following into global memory
          output <- alpha * accumulator + beta * source
            with:
              alpha = 1 / s_prime (to normalize when isLast=True, 1 otherwise)
              beta = alpha / m_prime (renormalize the output when the max
          changes) source is the current output
        */
        int col = blockN * Mma::Shape::kN;
        OutputTileIterator output_tile_it(
            typename OutputTileIterator::Params{(int32_t)p.head_dim_value},
            p.output_ptr + query_start() * p.head_dim_value + col,
            typename OutputTileIterator::TensorCoord{
                p.num_queries - query_start(), p.head_dim_value - col},
            thread_id());
        OutputTileIterator source_tile_it(
            typename OutputTileIterator::Params{(int32_t)p.head_dim_value},
            p.output_ptr + query_start() * p.head_dim_value + col,
            typename OutputTileIterator::TensorCoord{
                p.num_queries - query_start(), p.head_dim_value - col},
            thread_id());
        using ElementCompute =
            typename DefaultConfig::EpilogueOutputOp::ElementCompute;
        DISPATCH_BOOL(
            iter_key_start == 0, kIsFirst, ([&]() {
              DISPATCH_BOOL(
                  isLast, kIsLast, ([&]() {
                    using EpilogueOutputOp = typename cutlass::epilogue::
                        thread::MemoryEfficientAttentionNormalize<
                            output_t,
                            DefaultConfig::EpilogueOutputOp::kCount,
                            typename DefaultConfig::EpilogueOutputOp::
                                ElementAccumulator,
                            typename DefaultConfig::EpilogueOutputOp::
                                ElementCompute,
                            kIsFirst,
                            kIsLast,
                            cutlass::Array<ElementCompute, kQueriesPerBlock>>;
                    using Epilogue = typename cutlass::epilogue::threadblock::
                        EpilogueWithRowId<
                            typename DefaultEpilogue::Shape,
                            typename Mma::Operator,
                            DefaultEpilogue::kPartitionsK,
                            typename DefaultEpilogue::OutputTileIterator,
                            typename DefaultEpilogue::
                                AccumulatorFragmentIterator,
                            typename DefaultEpilogue::WarpTileIterator,
                            typename DefaultEpilogue::SharedLoadIterator,
                            EpilogueOutputOp,
                            typename DefaultEpilogue::Padding,
                            DefaultEpilogue::kFragmentsPerIteration,
                            true // IterationsUnroll
                            >;
                    EpilogueOutputOp rescale(s_prime, m_prime);
                    Epilogue epilogue(
                        shared_storage_mm.epilogue,
                        thread_id(),
                        warp_id(),
                        lane_id());
                    epilogue(rescale, output_tile_it, accum, source_tile_it);
                  }));
            }));
      }
    }
  };

  static constexpr int64_t kAlignmentQ = MM0::kAlignmentA;
  static constexpr int64_t kAlignmentK = MM0::kAlignmentB;
  static constexpr int64_t kAlignmentV = 1;

  struct SharedStorageAfterMM0 {
    // Everything here might be overwritten during MM0
    typename MM0::AccumulatorSharedStorage si;
    cutlass::Array<accum_t, kQueriesPerBlock> mi;
    typename MM1::SharedStorageMM1 mm1;
  };

  struct SharedStorage {
    cutlass::Array<accum_t, kQueriesPerBlock> m_prime;
    cutlass::Array<accum_t, kQueriesPerBlock> s_prime;
    union {
      typename MM0::Mma::SharedStorage mm0;
      SharedStorageAfterMM0 after_mm0;
    };
  };

  static void __device__ attention_kernel(Params const& p) {
    // In this block, we will only ever:
    // - read query[query_start:query_end, :]
    // - write to output[query_start:query_end, :]

    extern __shared__ char smem_buffer[];
    SharedStorage& shared_storage = *((SharedStorage*)smem_buffer);
    auto& m_prime = shared_storage.m_prime;
    auto& s_prime = shared_storage.s_prime;
    auto& si = shared_storage.after_mm0.si;
    auto& mi = shared_storage.after_mm0.mi;

    static_assert(kQueriesPerBlock < kNumWarpsPerBlock * kWarpSize);
    if (thread_id() < kQueriesPerBlock) {
      s_prime[thread_id()] = accum_t(0);
      m_prime[thread_id()] = -std::numeric_limits<accum_t>::infinity();
    }

    // Iterate through keys
    for (int32_t iter_key_start = 0; iter_key_start < p.num_keys;
         iter_key_start += kKeysPerBlock) {
      __syncthreads(); // Need to have shared memory initialized, and `m_prime`
                       // updated from end of prev iter
      // 1. Compute dot-product into shared memory for each query
      // also calculates `mi`, and updates `m_prime` / `s_prime`
      compute_dot_product_qk(
          p, iter_key_start, m_prime, s_prime, shared_storage);

      __syncthreads();
      bool isLast = (iter_key_start + kKeysPerBlock) >= p.num_keys;

      // 4. Partial matmul with the values we have and V
      // `v* <- v* . exp(m* - mi) + v_i . exp(si - mi)`
      MM1::compute_dot_product_att_value(
          p,
          shared_storage.after_mm0.mm1,
          shared_storage.after_mm0.si,
          iter_key_start,
          m_prime,
          s_prime,
          isLast // 6. Divide by s_prime all of the values on the last
                 // iteration
      );
      __syncthreads(); // we modify `m_prime` after

      // 5. `m_prime` <- `mi` (`mi` will be overwritten during MM0)
      static_assert(kQueriesPerBlock < kNumWarpsPerBlock * kWarpSize);
      if (thread_id() < kQueriesPerBlock) {
        m_prime[thread_id()] = mi[thread_id()];
      }
      __syncthreads();
    }

    // 7. Calculate logsumexp
    // To make the backward easier, we pad logsumexp with `inf`
    // this avoids a few bound checks, and is not more expensive during fwd
    static_assert(kQueriesPerBlock < kNumWarpsPerBlock * kWarpSize);
    if (p.logsumexp_ptr && thread_id() < kQueriesPerBlock) {
      if (query_start() + thread_id() < p.num_queries) {
        p.logsumexp_ptr[query_start() + thread_id()] =
            accum_t(m_prime[thread_id()]) +
            std::log(accum_t(s_prime[thread_id()]));
      } else if (thread_id() < kAlignLSE) {
        p.logsumexp_ptr[query_start() + thread_id()] =
            std::numeric_limits<accum_t>::infinity();
      }
    }
  }

  static __device__ void compute_dot_product_qk(
      Params const& p,
      int32_t const& iter_key_start,
      cutlass::Array<accum_t, kQueriesPerBlock>& m_prime,
      cutlass::Array<accum_t, kQueriesPerBlock>& s_prime,
      SharedStorage& shared_storage) {
    /*
    Computes the block-matrix product of:
    (a) query[query_start:query_end, :]
    with
    (b) key[iter_key_start:iter_key_start + kKeysPerBlock]
    and stores that into `shared_storage.si`
    */
    using MmaCore = typename MM0::MmaCore;
    using Mma = typename MM0::Mma;
    using IteratorA = typename MM0::IteratorA;
    using IteratorB = typename MM0::IteratorB;

    cutlass::gemm::GemmCoord problem_size(
        std::min((int32_t)kQueriesPerBlock, p.num_queries - query_start()),
        std::min(int32_t(kKeysPerBlock), p.num_keys - iter_key_start),
        p.head_dim);

    static_assert(
        MmaCore::WarpCount::kM * MmaCore::WarpCount::kN *
            MmaCore::WarpCount::kK ==
        kNumWarpsPerBlock);

    // Compute threadblock location
    cutlass::gemm::GemmCoord tb_tile_offset = {0, 0, 0};

    cutlass::MatrixCoord tb_offset_A{
        tb_tile_offset.m() * Mma::Shape::kM, tb_tile_offset.k()};

    cutlass::MatrixCoord tb_offset_B{
        tb_tile_offset.k(), tb_tile_offset.n() * Mma::Shape::kN};

    // Construct iterators to A and B operands
    typename Mma::IteratorA iterator_A(
        typename IteratorA::Params(typename MmaCore::LayoutA(p.head_dim)),
        p.query_ptr + query_start() * p.head_dim,
        {problem_size.m(), problem_size.k()},
        thread_id(),
        tb_offset_A);

    typename Mma::IteratorB iterator_B(
        typename IteratorB::Params(typename MmaCore::LayoutB(p.head_dim)),
        p.key_ptr + iter_key_start * p.head_dim,
        {problem_size.k(), problem_size.n()},
        thread_id(),
        tb_offset_B);

    auto my_warp_id = warp_id();
    auto my_lane_id = lane_id();

    // Construct thread-scoped matrix multiply
    Mma mma(shared_storage.mm0, thread_id(), my_warp_id, my_lane_id);

    typename Mma::FragmentC accum;

    accum.clear();

    auto gemm_k_iterations =
        (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

    // Compute threadblock-scoped matrix multiply-add
    mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);
    __syncthreads();
    auto& mi = shared_storage.after_mm0.mi;
    static_assert(kQueriesPerBlock < kNumWarpsPerBlock * kWarpSize);
    if (thread_id() < kQueriesPerBlock) {
      mi[thread_id()] = m_prime[thread_id()];
    }
    __syncthreads();

    // Scale
    accum_t scale = accum_t(1.0 / std::sqrt(float(p.head_dim)));
    accum = cutlass::multiplies<typename Mma::FragmentC>()(scale, accum);

    typename Mma::Operator::IteratorC::TensorCoord iteratorC_tile_offset = {
        (tb_tile_offset.m() * Mma::WarpCount::kM) +
            (my_warp_id % Mma::WarpCount::kM),
        (tb_tile_offset.n() * Mma::WarpCount::kN) +
            (my_warp_id / Mma::WarpCount::kM)};
    // Update `mi` from accum stored in registers
    MM0::ScalingCoefsUpdater::update<kQueriesPerBlock>(
        accum,
        mi,
        m_prime,
        s_prime,
        lane_id(),
        thread_id(),
        warp_id(),
        p.num_keys - iter_key_start,
        iteratorC_tile_offset);

    // Output results to shared-memory
    int warp_idx_mn_0 = my_warp_id %
        (MM0::Mma::Base::WarpCount::kM * MM0::Mma::Base::WarpCount::kN);
    auto output_tile_coords = cutlass::MatrixCoord{
        warp_idx_mn_0 % MM0::Mma::Base::WarpCount::kM,
        warp_idx_mn_0 / MM0::Mma::Base::WarpCount::kM};

    MM0::B2bGemm::accumToSmem(
        shared_storage.after_mm0.si, accum, my_lane_id, output_tile_coords);
  }

  static __device__ __forceinline__ int8_t lane_id() {
    return threadIdx.x;
  }
  static __device__ __forceinline__ int8_t warp_id() {
    return threadIdx.y;
  }
  static __device__ __forceinline__ int16_t thread_id() {
    return threadIdx.x + threadIdx.y * blockDim.x;
  }
  static __device__ __forceinline__ int32_t query_start() {
    return blockIdx.y * kQueriesPerBlock;
  }
};

template <typename AK>
__global__ void __launch_bounds__(AK::kNumThreads, AK::kMinBlocksPerSm)
    attention_kernel_batched(typename AK::Params params);

#define _ATTENTION_KERNEL_FORWARD_BEGIN(...)                                  \
  template <>                                                                 \
  __global__ void __launch_bounds__(                                          \
      __VA_ARGS__::kNumThreads, __VA_ARGS__::kMinBlocksPerSm)                 \
      attention_kernel_batched<__VA_ARGS__>(typename __VA_ARGS__::Params p) { \
    using Kernel = __VA_ARGS__;
#define _ATTENTION_KERNEL_FORWARD_END() }

#ifdef __CUDA_ARCH__
#define __CUDA_ARCH_OR_ZERO__ __CUDA_ARCH__
#else
#define __CUDA_ARCH_OR_ZERO__ 0
#endif

#define INSTANTIATE_ATTENTION_KERNEL_FORWARD(                      \
    ARCH, SCALAR_T, IS_ALIGNED, QUERIES_PER_BLOCK, KEYS_PER_BLOCK) \
  _ATTENTION_KERNEL_FORWARD_BEGIN(AttentionKernel<                 \
                                  SCALAR_T,                        \
                                  cutlass::arch::Sm##ARCH,         \
                                  IS_ALIGNED,                      \
                                  QUERIES_PER_BLOCK,               \
                                  KEYS_PER_BLOCK>)                 \
  auto batch_id = blockIdx.z;                                      \
  p.advance_batches(batch_id);                                     \
  Kernel::attention_kernel(p);                                     \
  _ATTENTION_KERNEL_FORWARD_END();

#define INSTANTIATE_ATTENTION_KERNEL_FORWARD_DISABLED(              \
    ARCH, SCALAR_T, IS_ALIGNED, QUERIES_PER_BLOCK, KEYS_PER_BLOCK)  \
  _ATTENTION_KERNEL_FORWARD_BEGIN(AttentionKernel<                  \
                                  SCALAR_T,                         \
                                  cutlass::arch::Sm##ARCH,          \
                                  IS_ALIGNED,                       \
                                  QUERIES_PER_BLOCK,                \
                                  KEYS_PER_BLOCK>)                  \
  printf(                                                           \
      "FATAL: this function is for sm%d, but was built for sm%d\n", \
      int(ARCH),                                                    \
      int(__CUDA_ARCH_OR_ZERO__));                                  \
  _ATTENTION_KERNEL_FORWARD_END();

// All kernels are disabled by default
#define INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM50(...) \
  INSTANTIATE_ATTENTION_KERNEL_FORWARD_DISABLED(50, __VA_ARGS__)
#define INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM70(...) \
  INSTANTIATE_ATTENTION_KERNEL_FORWARD_DISABLED(70, __VA_ARGS__)
#define INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM75(...) \
  INSTANTIATE_ATTENTION_KERNEL_FORWARD_DISABLED(75, __VA_ARGS__)
#define INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM80(...) \
  INSTANTIATE_ATTENTION_KERNEL_FORWARD_DISABLED(80, __VA_ARGS__)

// Enable the right one based on __CUDA_ARCH__
#ifndef __CUDA_ARCH__
#elif __CUDA_ARCH__ < 500
#error "Need cuda arch at least 5.0"
#elif __CUDA_ARCH__ < 700
#undef INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM50
#define INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM50(...) \
  INSTANTIATE_ATTENTION_KERNEL_FORWARD(50, __VA_ARGS__)
#elif __CUDA_ARCH__ < 750
#undef INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM70
#define INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM70(...) \
  INSTANTIATE_ATTENTION_KERNEL_FORWARD(70, __VA_ARGS__)
#elif __CUDA_ARCH__ < 800
#undef INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM75
#define INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM75(...) \
  INSTANTIATE_ATTENTION_KERNEL_FORWARD(75, __VA_ARGS__)
#elif __CUDA_ARCH__ >= 800
#undef INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM80
#define INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM80(...) \
  INSTANTIATE_ATTENTION_KERNEL_FORWARD(80, __VA_ARGS__)
#endif
