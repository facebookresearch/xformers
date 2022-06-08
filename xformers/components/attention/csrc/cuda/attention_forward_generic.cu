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

#include "cutlass/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/platform/platform.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"

#include <inttypes.h>

// #define FP16_ONLY_USE_TENSORCORES

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
  static __host__ at::PackedTensorAccessor<scalar_t, nDim> packed_accessor(
      at::Tensor const& tensor) {
    return at::PackedTensorAccessor<scalar_t, nDim>(
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
  static __host__ at::PackedTensorAccessor<scalar_t, nDim> packed_accessor(
      at::Tensor const& tensor) {
    return tensor.packed_accessor<scalar_t, nDim>();
  }
};
constexpr at::ScalarType math<float>::kAtScalarType;

namespace {
template <typename integer>
constexpr __host__ __device__ inline integer ceil_div(integer n, integer m) {
  return (n + m - 1) / m;
}

template <
    typename scalar_t_,
    typename accum_t_ = float,
    typename output_t_ = float>
struct AttentionKernel {
  using scalar_t = scalar_t_;
  using accum_t = accum_t_;
  using output_t = output_t_;

// Blocks
// NOTE: Looks like 16 works better for K <= 64
#ifdef FP16_ONLY_USE_TENSORCORES
  static constexpr int64_t kQueriesPerBlock = 64;
  static constexpr int64_t kWarpSize = 32;
  static constexpr int64_t kNumWarpsPerBlock = 2;
#else
  static constexpr int64_t kQueriesPerBlock = 32;
  static constexpr int64_t kWarpSize = 32;
  static constexpr int64_t kNumWarpsPerBlock = 4;
#endif
  static constexpr int64_t kNumBlocksX = 1;
  static int64_t getNumBlocksY(int64_t num_queries) {
    return ceil_div(num_queries, kQueriesPerBlock);
  }

  static constexpr int64_t kSiDim1 = kNumWarpsPerBlock * kWarpSize;

  static void __device__ attention_kernel(
      at::TensorAccessor<output_t, 2> output,
      at::TensorAccessor<accum_t, 1> logsumexp,
      at::TensorAccessor<scalar_t, 2> query,
      at::TensorAccessor<scalar_t, 2> key,
      at::TensorAccessor<scalar_t, 2> value) {
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
    __shared__ cutlass::Array<accum_t, kQueriesPerBlock> mi;
    __shared__ cutlass::Array<accum_t, kQueriesPerBlock> s_prime;
    accum_t __shared__ si[kQueriesPerBlock][kSiDim1];

    for (int32_t q = 0; q + lane_id < kQueriesPerBlock; q += kWarpSize) {
      mi[q + lane_id] = -std::numeric_limits<accum_t>::infinity();
    }
    if (warp_id == 0) {
      for (int32_t q = 0; q + lane_id < kQueriesPerBlock; q += kWarpSize) {
        s_prime[q + lane_id] = accum_t(0);
        m_prime[q + lane_id] = -std::numeric_limits<accum_t>::infinity();
      }
    }

    // Iterate through keys
    for (int32_t iter_key_start = 0; iter_key_start < num_keys;
         iter_key_start += kNumWarpsPerBlock * kWarpSize) {
      __syncthreads(); // Need to have shared memory initialized, and `m_prime`
                       // updated from end of prev iter

      // 1. Compute dot-product into shared memory for each query
      compute_dot_product_qk(iter_key_start, query, key, m_prime, si, mi);

      __syncthreads(); // `mi` calculation done based on block data. `mi[a][i]
                       // == mi[a][j]` for all (a, i, j)

      // WARNING: This modifies `si` and `m_prime` to store the precalculated
      // exp version so we can reuse it later in `compute_dot_product_att_value`
      static_assert(
          kQueriesPerBlock % kNumWarpsPerBlock == 0,
          ".. or add a condition to loop below");
      for (int32_t q = warp_id; q < kQueriesPerBlock;
           q += kNumWarpsPerBlock) { // parallel warps
        // 3. Update s_prime
        accum_t sp = accum_t(0);
        accum_t my_mi = mi[q];
        static_assert(
            kNumWarpsPerBlock * kWarpSize % kWarpSize == 0,
            ".. or add a condition to loop below");
        for (int32_t key_id = lane_id; key_id < kNumWarpsPerBlock * kWarpSize;
             key_id += kWarpSize) { // parallel lanes
          accum_t si_exp = math<accum_t>::exp(si[q][key_id] - my_mi);
          si_exp *= accum_t(key_id + iter_key_start < num_keys);
          sp += si_exp;
          si[q][key_id] = si_exp;
        }
        accum_t m_prime_exp = math<accum_t>::exp(m_prime[q] - my_mi);
        sp = warpSum(sp) + s_prime[q] * m_prime_exp;

        m_prime[q] = m_prime_exp;
        s_prime[q] = sp;
      }
      __syncthreads(); // `s_prime` done

      // 4. Partial matmull with the values we have and V
      // `v* <- v* . exp(m* - mi) + v_i . exp(si - mi)`
      compute_dot_product_att_value(iter_key_start, value, m_prime, si, output);
      __syncthreads(); // we modify `m_prime` after

      // 5. `m_prime` <- `mi`
      for (int64_t q = thread_id(); q < kQueriesPerBlock;
           q += kWarpSize * kNumWarpsPerBlock) { // parallel lanes
        m_prime[q] = mi[q];
      }
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
    if (logsumexp.size(0)) {
      iter_query_last = std::min<int32_t>(
          (int32_t)kQueriesPerBlock, int32_t(num_queries - query_start()));
      for (int64_t q = thread_id(); q < iter_query_last;
           q += kNumWarpsPerBlock * kWarpSize) {
        *(logsumexp.data() + query_start() + q) =
            accum_t(m_prime[q]) + std::log(accum_t(s_prime[q]));
      }
    }
  }

  // cutlass version
  static __device__ void compute_dot_product_att_value(
      int32_t const& iter_key_start,
      at::TensorAccessor<scalar_t, 2>& value,
      cutlass::Array<accum_t, kQueriesPerBlock> const& m_prime,
      accum_t si[kQueriesPerBlock][kSiDim1],
      at::TensorAccessor<output_t, 2>& output) {
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
        // Just use `cutlass::arch::OpClassTensorOp` for TensorCores (requires
        // sm>7.0)
        cutlass::arch::
            OpClassSimt, // OpClass:
                         // OpClassSimt/OpClassWmmaTensorOp/OpClassTensorOp
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

    cutlass::gemm::GemmCoord problem_size(
        std::min(
            (int64_t)kQueriesPerBlock, output.size(0) - query_start()), // M
        value.size(1), // N
        std::min(
            kNumWarpsPerBlock * kWarpSize, value.size(0) - iter_key_start) // K
    );
    typename IteratorA::Params params_A(kSiDim1);
    typename IteratorA::TensorRef ref_A(&si[0][0], kSiDim1);

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
      // Shared storage needed by threadblock-scoped matrix multiply-accumulate
      __shared__ typename Mma::SharedStorage shared_storage;

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

      typename Mma::FragmentC accum, accum2; // cutlass::Array<float, 16, true>
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

      // Load data already calculated, and rescale it (as the max value for the
      // softmax might have changed) Technically, we could do that on `accum`,
      // but then we would have to wait for load to finish to start the gemm
      // calculations. Let's rather load it in parallel (software pipelining) on
      // another register `accum2`
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

      // Add discounted `v_prime` (stored in `accum2`) to `accum` (which will be
      // stored to `output`)
      {
        auto it1 = accum.begin();
        auto it2 = accum2.begin();
        while (it1 != accum.end()) {
          *it1 = *it1 + *it2;
          ++it1;
          ++it2;
        }
      }
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
      at::TensorAccessor<scalar_t, 2>& query,
      at::TensorAccessor<scalar_t, 2>& key,
      cutlass::Array<accum_t, kQueriesPerBlock>& m_prime,
      accum_t si[kQueriesPerBlock][kSiDim1],
      cutlass::Array<accum_t, kQueriesPerBlock>& mi) {
    /*
    Computes the block-matrix product of:
    (a) query[query_start:query_end, :]
    with
    (b) key[iter_key_start:iter_key_start + kNumWarpsPerBlock * kWarpSize]
    and stores that into `si`
    */
#ifdef FP16_ONLY_USE_TENSORCORES
    using ThreadblockShape = cutlass::gemm::
        GemmShape<kQueriesPerBlock, kNumWarpsPerBlock * kWarpSize, 32>;
    using WarpShape = cutlass::gemm::GemmShape<kQueriesPerBlock, kWarpSize, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
    using OpClass = cutlass::arch::OpClassTensorOp; // OpClassWmmaTensorOp?
#else
    using ThreadblockShape = cutlass::gemm::
        GemmShape<kQueriesPerBlock, kNumWarpsPerBlock * kWarpSize, 8>;
    using WarpShape = cutlass::gemm::GemmShape<kQueriesPerBlock, kWarpSize, 8>;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    using OpClass = cutlass::arch::OpClassSimt;
#endif
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        ThreadblockShape, // ThreadblockShape,
        WarpShape, // WarpShape,
        InstructionShape, // InstructionShape,
        scalar_t, // ElementA,
        cutlass::layout::RowMajor, // LayoutA,
        scalar_t, // ElementB,
        cutlass::layout::ColumnMajor, // LayoutB,
        accum_t, // ElementC,
        cutlass::layout::RowMajor, // LayoutC,
        OpClass,
        2, // Stages,
        cutlass::arch::OpMultiplyAdd // Operator,
        >;

    using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        typename MmaCore::ElementA,
        typename MmaCore::LayoutA,
        1,
        typename MmaCore::IteratorThreadMapA>;

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

    int64_t num_queries = query.size(0);
    int64_t K = key.size(1);

    cutlass::gemm::GemmCoord problem_size(
        std::min((int64_t)kQueriesPerBlock, num_queries - query_start()),
        std::min(kNumWarpsPerBlock * kWarpSize, key.size(0) - iter_key_start),
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

    // Shared storage needed by threadblock-scoped matrix multiply-accumulate
    __shared__ typename Mma::SharedStorage shared_storage;

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
    Mma mma(shared_storage, thread_id(), my_warp_id, my_lane_id);

    typename Mma::FragmentC accum;

    accum.clear();

    auto gemm_k_iterations =
        (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

    // Compute threadblock-scoped matrix multiply-add
    mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);

    // Output results
    typename Mma::Operator::IteratorC iterator_C(
        {&si[0][0], kSiDim1}, my_lane_id);

    iterator_C.add_tile_offset(
        {(tb_tile_offset.m() * Mma::WarpCount::kM) +
             (my_warp_id % Mma::WarpCount::kM),
         (tb_tile_offset.n() * Mma::WarpCount::kN) +
             (my_warp_id / Mma::WarpCount::kM)});

    iterator_C.store(accum);
    __syncthreads();

    // 2. Update `mi`
    int64_t num_keys = key.size(0);
    accum_t scale = accum_t(1.0 / std::sqrt(float(K)));
    static_assert(kQueriesPerBlock % kNumWarpsPerBlock == 0);
    for (int16_t q = 0; q < kQueriesPerBlock;
         q += kNumWarpsPerBlock) { // parallel warps
      if (query_start() + q + warp_id() >= num_queries) {
        continue;
      }
      accum_t currentMax = m_prime[q + warp_id()];
      CUTLASS_PRAGMA_UNROLL
      for (int64_t key_id = 0; key_id < kSiDim1;
           key_id += kWarpSize) { // parallel lanes
        if (iter_key_start + key_id + lane_id() >= num_keys) {
          break;
        }
        // TODO: Scaling could be done as part of an epilogue
        // in the cutlass calculation above
        accum_t dot_product = si[q + warp_id()][key_id + lane_id()];
        dot_product *= scale;
        si[q + warp_id()][key_id + lane_id()] = dot_product;
        currentMax = std::max(currentMax, dot_product);
      }

      currentMax = warpMax(currentMax);
      mi[q + warp_id()] = currentMax;
    }
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

template <typename AK>
__global__ void __launch_bounds__(
    // maxThreadsPerBlock specifies the maximum number of threads per block with
    // which the application will ever launch
    AK::kWarpSize* AK::kNumWarpsPerBlock,
    // minBlocksPerMultiprocessor is optional and specifies the desired minimum
    // number of resident blocks per multiprocessor
    12 / AK::kNumWarpsPerBlock)
    attention_kernel_batched(
        at::PackedTensorAccessor<typename AK::output_t, 3> output,
        at::PackedTensorAccessor<typename AK::accum_t, 2> logsumexp,
        at::PackedTensorAccessor<typename AK::scalar_t, 3> query,
        at::PackedTensorAccessor<typename AK::scalar_t, 3> key,
        at::PackedTensorAccessor<typename AK::scalar_t, 3> value) {
  auto batch_id = blockIdx.z;
  AK::attention_kernel(
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

  if (query.scalar_type() == at::ScalarType::Float) {
#ifdef FP16_ONLY_USE_TENSORCORES
    TORCH_CHECK(
        false, "Only support f32 with FP16_ONLY_USE_TENSORCORES defined");
#else
    using scalar_t = float;
    using output_t = float;
    using AK = AttentionKernel<scalar_t, accum_t, output_t>;
    using m = math<scalar_t>;

    res = at::zeros(
        {B, M, K}, query.options().dtype(math<output_t>::kAtScalarType));

    dim3 grid(AK::kNumBlocksX, AK::getNumBlocksY(M), B);
    dim3 block(AK::kWarpSize, AK::kNumWarpsPerBlock, 1);

    attention_kernel_batched<AK><<<grid, block>>>(
        math<output_t>::packed_accessor<3>(res),
        logsumexp.packed_accessor<accum_t, 2>(),
        m::packed_accessor<3>(query),
        m::packed_accessor<3>(key),
        m::packed_accessor<3>(value));
#endif
  } else if (query.scalar_type() == at::ScalarType::Half) {
    using scalar_t = cutlass::half_t;
    using output_t = float;
    using AK = AttentionKernel<scalar_t, accum_t, output_t>;
    using m = math<scalar_t>;

    res = at::zeros(
        {B, M, K}, query.options().dtype(math<output_t>::kAtScalarType));

    dim3 grid(AK::kNumBlocksX, AK::getNumBlocksY(M), B);
    dim3 block(AK::kWarpSize, AK::kNumWarpsPerBlock, 1);

    attention_kernel_batched<AK><<<grid, block>>>(
        math<output_t>::packed_accessor<3>(res),
        logsumexp.packed_accessor<accum_t, 2>(),
        m::packed_accessor<3>(query),
        m::packed_accessor<3>(key),
        m::packed_accessor<3>(value));
  } else {
    TORCH_CHECK(false, "Only fp32 & half supported at the moment");
  }

  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(res, logsumexp, int64_t(), int64_t());
}
} // namespace

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::efficient_attention_forward_generic"),
      TORCH_FN(efficient_attention_forward_generic));
}
