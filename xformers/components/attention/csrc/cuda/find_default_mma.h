/*! \file
    \brief Cutlass provides helper template functions to figure out the right
   datastructures to instanciate to run a GEMM with various parameters (see
   `cutlass/gemm/threadblock/default_mma.h`). However, due to template
   instanciation priority rules, it will only create an MmaMultiStage with
   kStages=3 (otherwise creates an MmePipelined - which is not compatible with
   FastF32). kStages=3 uses too much shared memory and we want to use kStages=2,
   so we just copy-pasted some code from `default_mma.h` and
   `default_mma_core.h` files and wrapped this template to allow our usecase.

    This is really only for the FastF32 case - aka using TensorCores with fp32.
*/

#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"

namespace cutlass {
namespace gemm {
namespace threadblock {

template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Layout type for C and D matrix operand
    typename LayoutC,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Operation perfomed by GEMM
    typename Operator,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor = false,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone>
struct FindDefaultMma {
  using DefaultMma = cutlass::gemm::threadblock::DefaultMma<
      ElementA,
      LayoutA,
      kAlignmentA,
      ElementB,
      LayoutB,
      kAlignmentB,
      ElementAccumulator,
      LayoutC,
      OperatorClass,
      ArchTag,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      Stages,
      Operator,
      AccumulatorsInRowMajor,
      SharedMemoryClear>;
};

/// Specialization for sm80 / FastF32 / multistage with kStages=2
template <
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape>
struct FindDefaultMma<
    float,
    LayoutA_,
    kAlignmentA,
    float,
    LayoutB_,
    kAlignmentB,
    float,
    layout::RowMajor,
    arch::OpClassTensorOp,
    arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    2,
    arch::OpMultiplyAddFastF32> {
  struct DefaultMma {
    static constexpr int kStages = 2;
    static SharedMemoryClearOption constexpr SharedMemoryClear =
        SharedMemoryClearOption::kNone;
    using ElementA = float;
    using ElementB = float;
    using ElementAccumulator = float;
    using LayoutC = layout::RowMajor;
    using Operator = arch::OpMultiplyAddFastF32;
    static constexpr bool GatherA = false;
    static constexpr bool GatherB = false;

    static_assert(
        std::is_same<LayoutC, layout::RowMajor>::value ||
            std::is_same<LayoutC, layout::AffineRankN<2>>::value,
        "simt epilogue must be row major");

    static cutlass::arch::CacheOperation::Kind const CacheOpA =
        ((sizeof_bits<ElementA>::value * kAlignmentA) == 128)
        ? cutlass::arch::CacheOperation::Global
        : cutlass::arch::CacheOperation::Always;

    static cutlass::arch::CacheOperation::Kind const CacheOpB =
        ((sizeof_bits<ElementB>::value * kAlignmentB) == 128)
        ? cutlass::arch::CacheOperation::Global
        : cutlass::arch::CacheOperation::Always;

    // In theory we should do the following, but it would match the template for
    // MmaPipelined - and we want MmaMultiStage!
    // using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
    //     ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
    //     ElementB, LayoutB, ElementAccumulator, LayoutC,
    //     arch::OpClassTensorOp, Stages, Operator, false,
    //     CacheOpA, CacheOpB>;
    struct MmaCore {
      using LayoutA = LayoutA_;
      using LayoutB = LayoutB_;
      using Shape = ThreadblockShape;
      using ElementC = ElementAccumulator;
      static cutlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
      static cutlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;

      /// Number of warps present
      using WarpCount = GemmShape<
          Shape::kM / WarpShape::kM,
          Shape::kN / WarpShape::kN,
          Shape::kK / WarpShape::kK>;

      // Divisility requirements
      static_assert(
          !(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
          "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size.");

      /// Number of threads per warp
      static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

      /// Number of threads total
      static int const kThreads = WarpCount::kCount * kWarpSize;

      /// Size of a threadblock-scoped access
      static int const kAccessSizeInBits = 128;

      // Warp thread arrangement
      static int const kWarpThreadArrangementContiguousA =
          Shape::kK / (kAccessSizeInBits / sizeof_bits<ElementA>::value);

      static int const kWarpThreadArrangementStridedA =
          kWarpSize / kWarpThreadArrangementContiguousA;

      static int const kWarpThreadArrangementContiguousB =
          Shape::kK / (kAccessSizeInBits / sizeof_bits<ElementA>::value);

      static int const kWarpThreadArrangementStridedB =
          kWarpSize / kWarpThreadArrangementContiguousB;

      //
      // Shared memory layouts
      //

      using SmemLayoutA = layout::RowMajorTensorOpMultiplicandCrosswise<
          sizeof_bits<ElementA>::value,
          Shape::kK>;

      // Shared memory layout
      using SmemLayoutB = layout::ColumnMajorTensorOpMultiplicandCrosswise<
          sizeof_bits<ElementB>::value,
          Shape::kK>;

      //
      // Iterators to write to shared memory
      //

      /// ThreadMap of iterator A
      using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
          layout::PitchLinearShape<Shape::kK, Shape::kM>,
          kThreads,
          layout::PitchLinearShape<
              kWarpThreadArrangementContiguousA,
              kWarpThreadArrangementStridedA>,
          kAccessSizeInBits / sizeof_bits<ElementA>::value>;

      /// Shared memory iterator to A operand
      using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
          MatrixShape<Shape::kM, Shape::kK>,
          ElementA,
          SmemLayoutA,
          0,
          IteratorThreadMapA>;

      /// ThreadMap of iterator B
      using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
          layout::PitchLinearShape<Shape::kK, Shape::kN>,
          kThreads,
          layout::PitchLinearShape<
              kWarpThreadArrangementContiguousB,
              kWarpThreadArrangementStridedB>,
          kAccessSizeInBits / sizeof_bits<ElementB>::value>;

      /// Shared memory iterator to B operand
      using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
          MatrixShape<Shape::kK, Shape::kN>,
          ElementB,
          SmemLayoutB,
          1,
          IteratorThreadMapB>;

      //
      // Warp-level matrix multiply operator
      //

      // Define the warp-level tensor op
      using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
          WarpShape,
          InstructionShape,
          ElementA,
          SmemLayoutA,
          ElementB,
          SmemLayoutB,
          ElementC,
          LayoutC,
          Operator,
          WarpCount::kK>::Type;

      /// Policy used to define MmaPipelined
      using MmaPolicy = MmaPolicy<
          MmaTensorOp,
          MatrixShape<0, 0>,
          MatrixShape<0, 0>,
          WarpCount::kK>;
    };

    // Define iterators over tiles from the A operand
    using ThreadMapA = typename MmaCore::IteratorThreadMapA;
    using AccessTypeA = cutlass::Array<ElementA, kAlignmentA>;
    using IteratorA =
        cutlass::transform::threadblock::PredicatedTileAccessIterator<
            cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
            ElementA,
            LayoutA_,
            1,
            ThreadMapA,
            AccessTypeA,
            GatherA>;

    // Define iterators over tiles from the B operand
    using ThreadMapB = typename MmaCore::IteratorThreadMapB;
    using AccessTypeB = cutlass::Array<ElementB, kAlignmentB>;
    using IteratorB =
        cutlass::transform::threadblock::PredicatedTileAccessIterator<
            cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
            ElementB,
            LayoutB_,
            0,
            ThreadMapB,
            AccessTypeB,
            GatherB>;

    // Define the threadblock-scoped multistage matrix multiply
    using ThreadblockMma = cutlass::gemm::threadblock::MmaMultistage<
        typename MmaCore::Shape,
        IteratorA,
        typename MmaCore::SmemIteratorA,
        MmaCore::kCacheOpA,
        IteratorB,
        typename MmaCore::SmemIteratorB,
        MmaCore::kCacheOpB,
        ElementAccumulator,
        LayoutC,
        typename MmaCore::MmaPolicy,
        kStages,
        SharedMemoryClear>;
  };
};

} // namespace threadblock
} // namespace gemm
} // namespace cutlass
