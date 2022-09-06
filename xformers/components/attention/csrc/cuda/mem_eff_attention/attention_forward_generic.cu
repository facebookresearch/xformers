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
template <
    // The datatype of Q/K/V
    typename scalar_t_,
    // Intermediate accumulation type (including softmax)
    typename accum_t_,
    // Output type (only float tested so far)
    typename output_t_,
    // If Q/K/V are correctly aligned in memory and we can run a fast kernel
    bool isAligned_>
struct AttentionKernelInfo {
  using scalar_t = scalar_t_;
  using accum_t = accum_t_;
  using output_t = output_t_;
  static constexpr bool kIsAligned = isAligned_;

  // Blocks
  // NOTE: Looks like 16 works better for K <= 64
  static constexpr int64_t kQueriesPerBlock = 32;
  static constexpr int64_t kNumWarpsPerBlock = 4;
  static constexpr int64_t kWarpSize = 32;
  static constexpr int64_t kNumBlocksX = 1;

  static int64_t getNumBlocksY(int64_t num_queries) {
    return ceil_div(num_queries, kQueriesPerBlock);
  }
};

template <typename KernelInfo, typename ArchTag>
struct AttentionKernel {
  using scalar_t = typename KernelInfo::scalar_t;
  using accum_t = typename KernelInfo::accum_t;
  using output_t = typename KernelInfo::output_t;
  static constexpr bool kIsAligned = KernelInfo::kIsAligned;
  static constexpr int64_t kQueriesPerBlock = KernelInfo::kQueriesPerBlock;
  static constexpr int64_t kNumWarpsPerBlock = KernelInfo::kNumWarpsPerBlock;
  static constexpr int64_t kWarpSize = KernelInfo::kWarpSize;

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
    using ThreadblockShape = cutlass::gemm::GemmShape<
        kQueriesPerBlock,
        kNumWarpsPerBlock * kWarpSize,
        GemmType::ThreadK>;
    using WarpShape =
        cutlass::gemm::GemmShape<kQueriesPerBlock, kWarpSize, GemmType::WarpK>;
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
    using ThreadblockShape = cutlass::gemm::GemmShape<
        kQueriesPerBlock,
        kNumWarpsPerBlock * kWarpSize,
        GemmType::ThreadK>;
    using WarpShape =
        cutlass::gemm::GemmShape<kQueriesPerBlock, kWarpSize, GemmType::WarpK>;
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
        SharedStorageMM1& shared_storage_mm,
        typename MM0::AccumulatorSharedStorage& shared_storage_si,
        int32_t const& iter_key_start,
        at::TensorAccessor<scalar_t, 2, at::DefaultPtrTraits, int32_t>& value,
        cutlass::Array<accum_t, kQueriesPerBlock> const& m_prime,
        cutlass::Array<accum_t, kQueriesPerBlock> const& s_prime,
        bool isLast,
        at::TensorAccessor<output_t, 2, at::DefaultPtrTraits, int32_t>&
            output) {
      cutlass::gemm::GemmCoord problem_size(
          std::min(
              (int32_t)kQueriesPerBlock, output.size(0) - query_start()), // M
          value.size(1), // N
          std::min(
              int32_t(kNumWarpsPerBlock * kWarpSize),
              value.size(0) - iter_key_start) // K
      );

      typename IteratorB::Params params_B(LayoutB(value.stride(0)));
      typename IteratorB::TensorRef ref_B(
          &value[iter_key_start][0], value.stride(0));

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
            ref_B.data(),
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
        mma(gemm_k_iterations, accum, iterator_B, accum);

        /*
          Epilogue: Store the following into global memory
          output <- alpha * accumulator + beta * source
            with:
              alpha = 1 / s_prime (to normalize when isLast=True, 1 otherwise)
              beta = alpha / m_prime (renormalize the output when the max
          changes) source is the current output
        */
        OutputTileIterator output_tile_it(
            typename OutputTileIterator::Params{output.stride(0)},
            &output[query_start()][0],
            {output.size(0) - query_start(), output.size(1)},
            thread_id());
        OutputTileIterator source_tile_it(
            typename OutputTileIterator::Params{output.stride(0)},
            &output[query_start()][0],
            {output.size(0) - query_start(), output.size(1)},
            thread_id());

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
                            kIsLast>;
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

  static void __device__ attention_kernel(
      at::TensorAccessor<output_t, 2, at::DefaultPtrTraits, int32_t> output,
      at::TensorAccessor<accum_t, 1, at::DefaultPtrTraits, int32_t> logsumexp,
      at::TensorAccessor<scalar_t, 2, at::DefaultPtrTraits, int32_t> query,
      at::TensorAccessor<scalar_t, 2, at::DefaultPtrTraits, int32_t> key,
      at::TensorAccessor<scalar_t, 2, at::DefaultPtrTraits, int32_t> value) {
    int8_t lane_id = threadIdx.x;
    int8_t warp_id = threadIdx.y;

    // In this block, we will only ever:
    // - read query[query_start:query_end, :]
    // - write to output[query_start:query_end, :]

    int32_t num_keys = key.size(0);
    int32_t num_values = value.size(0);
    int32_t num_queries = query.size(0);
    int32_t K = key.size(1);

    extern __shared__ char smem_buffer[];
    SharedStorage& shared_storage = *((SharedStorage*)smem_buffer);
    auto& m_prime = shared_storage.m_prime;
    auto& s_prime = shared_storage.s_prime;
    auto& si = shared_storage.after_mm0.si;
    auto& mi = shared_storage.after_mm0.mi;

    if (warp_id == 0) {
      static_assert(kQueriesPerBlock == kWarpSize);
      s_prime[lane_id] = accum_t(0);
      m_prime[lane_id] = -std::numeric_limits<accum_t>::infinity();
    }

    // Iterate through keys
    for (int32_t iter_key_start = 0; iter_key_start < num_keys;
         iter_key_start += kNumWarpsPerBlock * kWarpSize) {
      __syncthreads(); // Need to have shared memory initialized, and `m_prime`
                       // updated from end of prev iter
      // 1. Compute dot-product into shared memory for each query
      // also calculates `mi`, and updates `m_prime` / `s_prime`
      compute_dot_product_qk(
          iter_key_start, query, key, m_prime, s_prime, shared_storage);

      __syncthreads();
      bool isLast =
          (iter_key_start + kNumWarpsPerBlock * kWarpSize) >= num_keys;

      // 4. Partial matmul with the values we have and V
      // `v* <- v* . exp(m* - mi) + v_i . exp(si - mi)`
      MM1::compute_dot_product_att_value(
          shared_storage.after_mm0.mm1,
          shared_storage.after_mm0.si,
          iter_key_start,
          value,
          m_prime,
          s_prime,
          isLast, // 6. Divide by s_prime all of the values on the last
                  // iteration
          output);
      __syncthreads(); // we modify `m_prime` after

      // 5. `m_prime` <- `mi` (`mi` will be overwritten during MM0)
      if (warp_id == 0) {
        static_assert(kQueriesPerBlock == kWarpSize);
        m_prime[lane_id] = mi[lane_id];
      }
      __syncthreads();
    }

    // 7. Calculate logsumexp
    // To make the backward easier, we pad logsumexp with `inf`
    // this avoids a few bound checks, and is not more expensive during fwd
    if (logsumexp.size(0) && warp_id == 0) {
      static_assert(kQueriesPerBlock == kWarpSize);
      if (query_start() + lane_id < num_queries) {
        logsumexp[query_start() + lane_id] =
            accum_t(m_prime[lane_id]) + std::log(accum_t(s_prime[lane_id]));
      } else {
        logsumexp[query_start() + lane_id] =
            std::numeric_limits<accum_t>::infinity();
      }
    }
  }

  static __device__ void compute_dot_product_qk(
      int32_t const& iter_key_start,
      at::TensorAccessor<scalar_t, 2, at::DefaultPtrTraits, int32_t>& query,
      at::TensorAccessor<scalar_t, 2, at::DefaultPtrTraits, int32_t>& key,
      cutlass::Array<accum_t, kQueriesPerBlock>& m_prime,
      cutlass::Array<accum_t, kQueriesPerBlock>& s_prime,
      SharedStorage& shared_storage) {
    /*
    Computes the block-matrix product of:
    (a) query[query_start:query_end, :]
    with
    (b) key[iter_key_start:iter_key_start + kNumWarpsPerBlock * kWarpSize]
    and stores that into `shared_storage.si`
    */
    using MmaCore = typename MM0::MmaCore;
    using Mma = typename MM0::Mma;
    using IteratorA = typename MM0::IteratorA;
    using IteratorB = typename MM0::IteratorB;

    int32_t num_queries = query.size(0);
    int32_t K = key.size(1);

    cutlass::gemm::GemmCoord problem_size(
        std::min((int32_t)kQueriesPerBlock, num_queries - query_start()),
        std::min(
            int32_t(kNumWarpsPerBlock * kWarpSize),
            key.size(0) - iter_key_start),
        K);
    typename IteratorA::Params params_A(
        typename MmaCore::LayoutA(query.stride(0)));
    typename IteratorA::TensorRef ref_A(
        &query[query_start()][0], query.stride(0));

    typename IteratorB::Params params_B(
        typename MmaCore::LayoutB(key.stride(0)));
    typename IteratorB::TensorRef ref_B(&key[iter_key_start][0], key.stride(0));

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
        params_A,
        ref_A.data(),
        {problem_size.m(), problem_size.k()},
        thread_id(),
        tb_offset_A);

    typename Mma::IteratorB iterator_B(
        params_B,
        ref_B.data(),
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
    if (my_warp_id == 0) {
      static_assert(kQueriesPerBlock == kWarpSize);
      mi[my_lane_id] = m_prime[my_lane_id];
    }
    __syncthreads();

    // Scale
    accum_t scale = accum_t(1.0 / std::sqrt(float(K)));
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
        warp_id(),
        key.size(0) - iter_key_start,
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

template <typename AKInfo>
__global__ void __launch_bounds__(
    // maxThreadsPerBlock specifies the maximum number of threads per block with
    // which the application will ever launch
    AKInfo::kWarpSize* AKInfo::kNumWarpsPerBlock,
    // minBlocksPerMultiprocessor is optional and specifies the desired minimum
    // number of resident blocks per multiprocessor
    // TODO: We get slightly better performance by *removing* this on A100
    12 / AKInfo::kNumWarpsPerBlock)
    attention_kernel_batched(
        at::PackedTensorAccessor32<typename AKInfo::output_t, 3> output,
        at::PackedTensorAccessor32<typename AKInfo::accum_t, 2> logsumexp,
        at::PackedTensorAccessor32<typename AKInfo::scalar_t, 3> query,
        at::PackedTensorAccessor32<typename AKInfo::scalar_t, 3> key,
        at::PackedTensorAccessor32<typename AKInfo::scalar_t, 3> value) {
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

#ifdef __CUDA_ARCH__
  static_assert(CurrentArch::kMinComputeCapability * 10 <= __CUDA_ARCH__);
#endif

  auto batch_id = blockIdx.z;
  AttentionKernel<AKInfo, CurrentArch>::attention_kernel(
      output[batch_id],
      logsumexp[batch_id],
      query[batch_id],
      key[batch_id],
      value[batch_id]);
}

std::tuple<at::Tensor, at::Tensor, int64_t, int64_t>
efficient_attention_forward_generic(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    bool compute_logsumexp,
    const c10::optional<at::Tensor>& attn_bias_,
    double p) {
  TORCH_CHECK(p == 0.0, "Dropout is not supported at the moment");
  TORCH_CHECK(
      !attn_bias_.has_value(), "attn_bias is not supported at the moment");

  TORCH_CHECK(query.dim() == 3);
  TORCH_CHECK(key.dim() == 3);
  TORCH_CHECK(value.dim() == 3);

  TORCH_CHECK(query.size(2) == key.size(2));
  TORCH_CHECK(query.size(0) == key.size(0));

  TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
  TORCH_CHECK(key.is_cuda(), "key must be a CUDA tensor");
  TORCH_CHECK(value.is_cuda(), "value must be a CUDA tensor");

  TORCH_CHECK(!query.is_sparse(), "query must be a dense tensor");
  TORCH_CHECK(!key.is_sparse(), "key must be a dense tensor");
  TORCH_CHECK(!value.is_sparse(), "value must be a dense tensor");

  TORCH_CHECK(query.is_contiguous());
  TORCH_CHECK(key.is_contiguous());
  TORCH_CHECK(value.is_contiguous());

  at::Tensor attn_bias;
  if (attn_bias_.has_value()) {
    attn_bias = *attn_bias_;
    TORCH_CHECK(query.dim() == attn_bias.dim());
    TORCH_CHECK(query.size(0) == attn_bias.size(0));
    TORCH_CHECK(query.size(1) == attn_bias.size(1));
    TORCH_CHECK(key.size(1) == attn_bias.size(2));
    TORCH_CHECK(attn_bias.stride(1) == 0);
  }

  at::cuda::CUDAGuard device_guard(query.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);
  int64_t K = query.size(2);

  using accum_t = float;

  at::Tensor res;
  at::Tensor logsumexp;

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
// Dispatch to the right kernel
#define DISPATCH_TYPES(func)                                          \
  {                                                                   \
    if (query.scalar_type() == at::ScalarType::Float) {               \
      using scalar_t = float;                                         \
      using output_t = float;                                         \
      func();                                                         \
    } else if (query.scalar_type() == at::ScalarType::Half) {         \
      using scalar_t = cutlass::half_t;                               \
      using output_t = float;                                         \
      func();                                                         \
    } else {                                                          \
      TORCH_CHECK(false, "Only fp32 & half supported at the moment"); \
    }                                                                 \
  }

  DISPATCH_TYPES(([&]() {
    // Run a more efficient kernel (with `isAligned=True`) if memory is
    // correctly aligned
    using AlignedAKI = AttentionKernelInfo<scalar_t, accum_t, output_t, true>;
    bool isAligned;
    DISPATCH_ARCHTAG(([&]() {
      using AlignedAK = AttentionKernel<AlignedAKI, ArchTag>;
      isAligned =
          (query.stride(1) % AlignedAK::kAlignmentQ == 0 &&
           key.stride(1) % AlignedAK::kAlignmentK == 0 &&
           value.stride(1) % AlignedAK::kAlignmentV == 0);
      // TODO: Should we warn or log somewhere when we use a less efficient
      // kernel due to wrong alignment?
    }));
    DISPATCH_BOOL(
        isAligned, kIsAligned, ([&]() {
          using AKI =
              AttentionKernelInfo<scalar_t, accum_t, output_t, kIsAligned>;
          size_t smem_bytes = 0;
          DISPATCH_ARCHTAG(([&]() {
            using AK = AttentionKernel<AKI, ArchTag>;
            smem_bytes = sizeof(typename AK::SharedStorage);
            // Might happen on Sm80/half, where the minimum alignment is 32bits
            TORCH_CHECK(
                query.stride(1) % AK::kAlignmentQ == 0,
                "query is not correctly aligned");
            TORCH_CHECK(
                key.stride(1) % AK::kAlignmentK == 0,
                "key is not correctly aligned");
            TORCH_CHECK(
                value.stride(1) % AK::kAlignmentV == 0,
                "value is not correctly aligned");
          }));
          TORCH_INTERNAL_ASSERT(smem_bytes > 0, "No kernel found!?");

          res = at::zeros(
              {B, M, K},
              query.options().dtype(TypeTraits<output_t>::atScalarType()));
          // NOTE: Should be aligned (by padding) in case M is not a good number
          // for loading during backward
          constexpr decltype(M) kAlignLSE = 32; // block size of backward
          logsumexp = at::empty(
              {B, compute_logsumexp ? ceil_div(M, kAlignLSE) * kAlignLSE : 0},
              query.options().dtype(at::ScalarType::Float));

          dim3 grid(AKI::kNumBlocksX, AKI::getNumBlocksY(M), B);
          dim3 block(AKI::kWarpSize, AKI::kNumWarpsPerBlock, 1);

          constexpr auto kernel_fn = attention_kernel_batched<AKI>;
          if (smem_bytes > 0xc000) {
            TORCH_INTERNAL_ASSERT(
                computeCapability >= 70,
                "This kernel requires too much shared memory on this machine!");
            AT_CUDA_CHECK(cudaFuncSetAttribute(
                kernel_fn,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_bytes));
          }

          using m = TypeTraits<scalar_t>;
          kernel_fn<<<grid, block, smem_bytes>>>(
              TypeTraits<output_t>::packed_accessor<3>(res),
              logsumexp.packed_accessor32<accum_t, 2>(),
              m::packed_accessor<3>(query),
              m::packed_accessor<3>(key),
              m::packed_accessor<3>(value));
        }));
  }));

  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(res, logsumexp, int64_t(), int64_t());
}
} // namespace

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::efficient_attention_forward_generic"),
      TORCH_FN(efficient_attention_forward_generic));
}
