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
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/platform/platform.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "find_default_mma.h"

#include <inttypes.h>

// XXX: Maybe CUDA will wake up one day and provide this
template <typename scalar_t>
struct math;

template <>
struct math<cutlass::half_t> {
  using scalar_t = cutlass::half_t;
  using torch_dtype = half;
  static constexpr at::ScalarType kAtScalarType = at::ScalarType::Half;

  static __device__ __forceinline__ cutlass::half_t exp(
      cutlass::half_t const& h) {
    return cutlass::half_t(hexp(h.to_half()));
  }
  template <int nDim>
  static __host__ at::PackedTensorAccessor32<scalar_t, nDim> packed_accessor(
      at::Tensor const& tensor) {
    return at::PackedTensorAccessor32<scalar_t, nDim>(
        (scalar_t*)(tensor.data_ptr()),
        tensor.sizes().data(),
        tensor.strides().data());
  }
};
constexpr at::ScalarType math<cutlass::half_t>::kAtScalarType;

template <>
struct math<float> {
  using scalar_t = float;
  using torch_dtype = float;
  static constexpr at::ScalarType kAtScalarType = at::ScalarType::Float;

  static __device__ __forceinline__ float exp(float const& h) {
    return expf(h);
  }
  template <int nDim>
  static __host__ at::PackedTensorAccessor32<scalar_t, nDim> packed_accessor(
      at::Tensor const& tensor) {
    return tensor.packed_accessor32<scalar_t, nDim>();
  }
};
constexpr at::ScalarType math<float>::kAtScalarType;

namespace {
template <typename integer>
constexpr __host__ __device__ inline integer ceil_div(integer n, integer m) {
  return (n + m - 1) / m;
}

template <typename ArchTag, typename scalar_t_, typename Enable = void>
struct GemmTypeQK {
  // Default GEMM with simt
  static constexpr int ThreadK = 8;
  static constexpr int WarpK = 8;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  using OpClass = cutlass::arch::OpClassSimt;
  using Operator = cutlass::arch::OpMultiplyAdd;
};

// Using GEMM with TensorCores when available
template <typename ArchTag>
struct GemmTypeQK<
    ArchTag,
    float, // scalar_t_
    typename std::enable_if<ArchTag::kMinComputeCapability >= 80>::type> {
  static constexpr int ThreadK = 32;
  static constexpr int WarpK = 32;
  using OpClass = cutlass::arch::OpClassTensorOp;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using Operator = cutlass::arch::OpMultiplyAddFastF32;
};

template <typename ArchTag>
struct GemmTypeQK<
    ArchTag,
    cutlass::half_t, // scalar_t_
    typename std::enable_if<ArchTag::kMinComputeCapability >= 70>::type> {
  static constexpr int ThreadK = 32;
  static constexpr int WarpK = 32;
  using OpClass = cutlass::arch::OpClassTensorOp;
  using InstructionShape = typename std::conditional<
      ArchTag::kMinComputeCapability >= 75,
      cutlass::gemm::GemmShape<16, 8, 8>,
      cutlass::gemm::GemmShape<8, 8, 4>>::type;
  using Operator = cutlass::arch::OpMultiplyAdd;
};

template <
    typename scalar_t_,
    typename accum_t_,
    typename output_t_,
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
    using GemmType = GemmTypeQK<ArchTag, scalar_t>;

    using OpClass = typename GemmType::OpClass;
    using DefaultConfig =
        typename cutlass::gemm::device::DefaultGemmConfiguration<
            OpClass,
            ArchTag,
            scalar_t,
            scalar_t,
            accum_t, // ElementC
            accum_t // ElementAccumulator
            >;
    static constexpr int64_t kAlignmentA =
        kIsAligned ? DefaultConfig::kAlignmentA : 1;
    static constexpr int64_t kAlignmentB =
        kIsAligned ? DefaultConfig::kAlignmentB : 1;
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
        2, // Should use `DefaultConfig::kStages`, but that uses too much smem
        typename GemmType::Operator // Operator
        >::DefaultMma;
    using MmaCore = typename DefaultMma::MmaCore;
    using IteratorA = typename DefaultMma::IteratorA;
    using IteratorB = typename DefaultMma::IteratorB;
    using Mma = typename DefaultMma::ThreadblockMma;
    using ScalingCoefsUpdater = typename DefaultAttentionScalingCoefsUpdater<
        typename Mma::Operator::IteratorC,
        accum_t,
        kQueriesPerBlock,
        kWarpSize>::Updater;
  };

  struct MM1 {
    using ThreadblockShape = cutlass::gemm::
        GemmShape<kQueriesPerBlock, kNumWarpsPerBlock * kWarpSize, 8>;
    using WarpShape = cutlass::gemm::GemmShape<kQueriesPerBlock, kWarpSize, 8>;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

    // default_mma_core_simt.h
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        ThreadblockShape, // ThreadblockShape,
        WarpShape, // WarpShape,
        InstructionShape, // InstructionShape,
        accum_t, // ElementA,
        cutlass::layout::RowMajor, // LayoutA,
        scalar_t, // ElementB,
        cutlass::layout::RowMajor, // LayoutB,
        accum_t, // ElementC,
        cutlass::layout::RowMajor, // LayoutC,
        cutlass::arch::OpClassSimt,
        2, // Stages,
        cutlass::arch::OpMultiplyAdd // Operator,
        >;

    using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        typename MmaCore::ElementA,
        typename MmaCore::LayoutA,
        1,
        typename MmaCore::IteratorThreadMapA,
        MmaCore::IteratorThreadMapA::kElementsPerAccess, // AccessSize
        false, // Gather
        false // LoadFromGlobalMemoryOnly
        >;

    // Define iterators over tiles from the B operand
    using IteratorB = cutlass::transform::threadblock::PredicatedTileIterator<
        cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        typename MmaCore::ElementB,
        typename MmaCore::LayoutB,
        0,
        typename MmaCore::IteratorThreadMapB>;

    using Mma = cutlass::gemm::threadblock::MmaPipelined<
        typename MmaCore::Shape,
        IteratorA,
        typename MmaCore::SmemIteratorA,
        IteratorB,
        typename MmaCore::SmemIteratorB,
        typename MmaCore::ElementC,
        typename MmaCore::LayoutC,
        typename MmaCore::MmaPolicy>;

    static __device__ void compute_dot_product_att_value(
        typename Mma::SharedStorage& shared_storage,
        int32_t const& iter_key_start,
        at::TensorAccessor<scalar_t, 2, at::DefaultPtrTraits, int32_t>& value,
        cutlass::Array<accum_t, kQueriesPerBlock> const& m_prime,
        accum_t si[kQueriesPerBlock][kNumWarpsPerBlock * kWarpSize],
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
      typename IteratorA::Params params_A(kNumWarpsPerBlock * kWarpSize);
      typename IteratorA::TensorRef ref_A(
          &si[0][0], kNumWarpsPerBlock * kWarpSize);

      typename IteratorB::Params params_B(
          typename MmaCore::LayoutB(value.stride(0)));
      typename IteratorB::TensorRef ref_B(
          &value[iter_key_start][0], value.stride(0));

      static_assert(
          MmaCore::WarpCount::kM * MmaCore::WarpCount::kN *
              MmaCore::WarpCount::kK ==
          kNumWarpsPerBlock);

      const int64_t nBlockN =
          ceil_div((int64_t)problem_size.n(), int64_t(ThreadblockShape::kN));
      for (int blockN = 0; blockN < nBlockN; ++blockN) {
        // Compute threadblock location
        cutlass::gemm::GemmCoord tb_tile_offset = {0, blockN, 0};

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

        // Construct thread-scoped matrix multiply
        Mma mma(shared_storage, thread_id(), warp_id(), lane_id());

        auto iterator_C_offset_m = (tb_tile_offset.m() * Mma::WarpCount::kM) +
            (warp_id() % Mma::WarpCount::kM);
        auto iterator_C_offset_n = (tb_tile_offset.n() * Mma::WarpCount::kN) +
            (warp_id() / Mma::WarpCount::kM);
        using LaneMmaShape = typename Mma::Policy;
        typename Mma::Operator::IteratorC::Policy::LaneLayout lane_layout =
            Mma::Operator::IteratorC::Policy::get_lane_layout();
        cutlass::MatrixCoord lane_offset =
            lane_layout.inverse(lane_id()) *
            cutlass::MatrixCoord(
                Mma::Operator::IteratorC::Policy::LaneMmaShape::kM,
                Mma::Operator::IteratorC::Policy::LaneMmaShape::kN);

        typename Mma::FragmentC accum,
            accum2; // cutlass::Array<float, 16, true>
        // TODO: We could avoid all this mess using cutlass's Epilogue concept I
        // think but I got lost in templates and reimplemented everything

        const int32_t thread_offset_m =
            Mma::WarpGemm::kM * iterator_C_offset_m + lane_offset.row();
        const int32_t thread_offset_n =
            Mma::WarpGemm::kN * iterator_C_offset_n + lane_offset.column();
        output_t* output_ptr = &output[query_start()][0];
        const int32_t output_s0 = output.stride(0);
        const int32_t max_m = output.size(0) - query_start();
        const int32_t max_n = output.size(1);

        // Load data already calculated, and rescale it (as the max value for
        // the softmax might have changed) Technically, we could do that on
        // `accum`, but then we would have to wait for load to finish to start
        // the gemm calculations. Let's rather load it in parallel (software
        // pipelining) on another register `accum2`
        accum.clear();
        accum2.clear();
        iterate_on_frag<Mma::Operator::IteratorC>(
            accum2,
            thread_offset_m,
            thread_offset_n,
            [&](typename Mma::FragmentC::reference accum_v,
                int32_t m,
                int32_t n) {
              if (m < max_m && n < max_n) {
                accum_v = accum_t(output_ptr[m * output_s0 + n]) * m_prime[m];
              }
            });
        int gemm_k_iterations =
            (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

        // Compute threadblock-scoped matrix multiply-add
        mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);

        // Add discounted `v_prime` (stored in `accum2`) to `accum` (which will
        // be stored to `output`)
        accum = cutlass::plus<decltype(accum)>()(accum, accum2);
        iterate_on_frag<Mma::Operator::IteratorC>(
            accum,
            thread_offset_m,
            thread_offset_n,
            [&](typename Mma::FragmentC::reference accum_v,
                int32_t const& m,
                int32_t const& n) {
              if (m < max_m && n < max_n) {
                output_ptr[m * output_s0 + n] = output_t(accum_v);
              }
            });
      }
    }
  };

  static constexpr int64_t kAlignmentQ = MM0::kAlignmentA;
  static constexpr int64_t kAlignmentK = MM0::kAlignmentB;
  static constexpr int64_t kAlignmentV = 1;

  struct SharedStorageGlobal {
    accum_t si[kQueriesPerBlock][kNumWarpsPerBlock * kWarpSize];
    cutlass::Array<accum_t, kQueriesPerBlock> mi;
    typename MM1::Mma::SharedStorage mm1;
  };

  union SharedStorage {
    // Shared storage needed by threadblock-scoped matrix multiply-accumulate
    typename MM0::Mma::SharedStorage mm0;
    SharedStorageGlobal after_mm0;
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

    __shared__ cutlass::Array<accum_t, kQueriesPerBlock> m_prime;
    __shared__ cutlass::Array<accum_t, kQueriesPerBlock> s_prime;
    __shared__ SharedStorage shared_storage;
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

      // 4. Partial matmull with the values we have and V
      // `v* <- v* . exp(m* - mi) + v_i . exp(si - mi)`
      MM1::compute_dot_product_att_value(
          shared_storage.after_mm0.mm1,
          iter_key_start,
          value,
          m_prime,
          si,
          output);
      __syncthreads(); // we modify `m_prime` after

      // 5. `m_prime` <- `mi`
      if (warp_id == 0) {
        static_assert(kQueriesPerBlock == kWarpSize);
        m_prime[lane_id] = mi[lane_id];
      }
      __syncthreads();
    }

    // 6. Divide by s_prime all of the values
    const int32_t output_stride0 = output.stride(0);
    const int32_t iter_col_last = output.size(1) - lane_id;
    int32_t iter_query_last = std::min<int32_t>(
        (int32_t)kQueriesPerBlock,
        int32_t(num_queries - warp_id - query_start()));
    if (iter_col_last > 0 && iter_query_last > 0) {
      // &output[query_start()][thread_id]
      output_t* output_line_ptr =
          output.data() + (query_start() + warp_id) * output_stride0 + lane_id;
      for (int32_t q = 0; q < iter_query_last;
           q += kNumWarpsPerBlock) { // parallel warps
        auto line_s_prime = s_prime[q + warp_id];
        for (int32_t value_col = 0; value_col < iter_col_last;
             value_col += kWarpSize) { // parallel lanes
          output_line_ptr[value_col] =
              output_t(accum_t(output_line_ptr[value_col]) / line_s_prime);
        }
        output_line_ptr += output_stride0 * kNumWarpsPerBlock;
      }
    }

    // 7. Calculate logsumexp
    if (logsumexp.size(0) && warp_id == 0) {
      static_assert(kQueriesPerBlock == kWarpSize);
      if (query_start() + lane_id < num_queries) {
        logsumexp[query_start() + lane_id] =
            accum_t(m_prime[lane_id]) + std::log(accum_t(s_prime[lane_id]));
      }
    }
  }

  template <typename Iterator, typename Fragment, typename FN>
  static void __device__ __forceinline__ iterate_on_frag(
      Fragment& frag,
      int32_t const& offset_m,
      int32_t const& offset_n,
      FN callback) {
    // TODO: This is quite hacky, and only needed for Simt. For other Mmas, we
    // can use epilogue.
    using Policy = typename Iterator::Policy;
    using Delta = typename Iterator::Delta;
    using Iterations = typename Iterator::Iterations;
    using Element = typename Iterator::Element;

    CUTLASS_PRAGMA_UNROLL
    for (int mma_m = 0; mma_m < Iterations::kRow; ++mma_m) { // 0
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < Policy::LaneMmaShape::kM; ++m) {
        CUTLASS_PRAGMA_UNROLL
        for (int mma_n = 0; mma_n < Iterations::kColumn; ++mma_n) {
          CUTLASS_PRAGMA_UNROLL
          for (int n = 0; n < Policy::LaneMmaShape::kN; ++n) {
            callback(
                frag.at(
                    n +
                    Policy::LaneMmaShape::kN *
                        (mma_n +
                         Iterations::kColumn *
                             (m + mma_m * Policy::LaneMmaShape::kM))),
                offset_m + m + mma_m * Delta::kRow,
                offset_n + n +
                    mma_n * Policy::WarpShape::kColumn *
                        Policy::LaneMmaShape::kN);
          }
        }
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
    and stores that into `si`
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
    auto& si = shared_storage.after_mm0.si;
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
    typename MM0::ScalingCoefsUpdater updater;
    updater.update(
        accum,
        mi,
        m_prime,
        s_prime,
        my_lane_id,
        my_warp_id,
        key.size(0) - iter_key_start,
        iteratorC_tile_offset);

    // Output results
    typename Mma::Operator::IteratorC iterator_C(
        {&si[0][0], kNumWarpsPerBlock * kWarpSize}, my_lane_id);

    iterator_C.add_tile_offset(iteratorC_tile_offset);
    iterator_C.store(accum);
  }

  static __device__ __forceinline__ accum_t warpMax(accum_t val) {
    for (int stride = kWarpSize / 2; stride > 0; stride >>= 1) {
      accum_t tmp =
          accum_t(__shfl_xor_sync(0xffffffff, val, stride, kWarpSize));
      val = tmp > val ? tmp : val;
    }
    return val;
  }

  static __device__ __forceinline__ accum_t warpSum(accum_t val) {
    for (int stride = kWarpSize / 2; stride > 0; stride >>= 1) {
      accum_t tmp =
          accum_t(__shfl_xor_sync(0xffffffff, val, stride, kWarpSize));
      val += tmp;
    }
    return val;
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
  at::Tensor logsumexp = at::empty(
      {B, compute_logsumexp ? M : 0},
      query.options().dtype(at::ScalarType::Float));

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
      using ArchTag = cutlass::arch::Sm75;                                \
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

#define DISPATCH_BOOL(BOOL_V, BOOL_NAME, F) \
  {                                         \
    if (BOOL_V) {                           \
      constexpr bool BOOL_NAME = true;      \
      F();                                  \
    } else {                                \
      constexpr bool BOOL_NAME = false;     \
      F();                                  \
    }                                       \
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
    }));
    DISPATCH_BOOL(
        isAligned, IsAligned, ([&]() {
          using AKI =
              AttentionKernelInfo<scalar_t, accum_t, output_t, IsAligned>;
          DISPATCH_ARCHTAG(([&]() {
            using AK = AttentionKernel<AKI, ArchTag>;
            TORCH_INTERNAL_ASSERT(
                query.stride(1) % AK::kAlignmentQ == 0,
                "query is not correctly aligned");
            TORCH_INTERNAL_ASSERT(
                key.stride(1) % AK::kAlignmentK == 0,
                "key is not correctly aligned");
            TORCH_INTERNAL_ASSERT(
                value.stride(1) % AK::kAlignmentV == 0,
                "value is not correctly aligned");
          }));
          using m = math<scalar_t>;

          res = at::zeros(
              {B, M, K}, query.options().dtype(math<output_t>::kAtScalarType));

          dim3 grid(AKI::kNumBlocksX, AKI::getNumBlocksY(M), B);
          dim3 block(AKI::kWarpSize, AKI::kNumWarpsPerBlock, 1);

          attention_kernel_batched<AKI><<<grid, block>>>(
              math<output_t>::packed_accessor<3>(res),
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
