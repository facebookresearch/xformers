#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cutlass/array.h>
#include <cutlass/bfloat16.h>
#include <cutlass/half.h>

#include "pt_stable_utils.h"
#include "static_sort.h"

template <typename Element, bool kHasRandom>
struct Sp24MaskKernel {
  // thread-level tiling
  static constexpr int kNumRows = 4;
  static constexpr int kNumCols = 4;
  static constexpr int kMaxPerRow = 2;
  static constexpr int kMaxPerCol = 2;
  static constexpr int kTakeMaximumCount = kNumRows * kMaxPerRow;
  // block-level tiling
  static constexpr int kRowsPerBlock = 32;
  static constexpr int kColsPerBlock = 16;
  static constexpr int kNumThreadsPerBlock =
      (kRowsPerBlock / kNumRows) * (kRowsPerBlock / kNumRows);
  static_assert(kNumThreadsPerBlock % 32 == 0, "invalid warp size");

  struct PseudoRandomOrder {
    uint8_t row;
    uint8_t col;
    uint16_t random_score;

    CUTLASS_DEVICE bool operator<(PseudoRandomOrder const& other) const {
      return random_score < other.random_score;
    }
  };

  struct ElementWithPos {
    Element e;
    uint8_t row;
    uint8_t col;

    CUTLASS_DEVICE bool operator<(ElementWithPos const& other) const {
      return e < other.e;
    }
  };

  struct Params {
    // Inputs assumed in RowMajor
    Element const* input;
    Element* output;
    int64_t stride0;
    int64_t size0;
    int64_t size1;
    int8_t numRandom;

    CUTLASS_HOST_DEVICE int _getNumBlocksRows() const {
      return (size0 + kRowsPerBlock - 1) / kRowsPerBlock;
    }
    CUTLASS_HOST_DEVICE int _getNumBlocksCols() const {
      return (size1 + kColsPerBlock - 1) / kColsPerBlock;
    }
    CUTLASS_HOST_DEVICE int getBlocksGrid() const {
      return _getNumBlocksRows() * _getNumBlocksCols();
    }
    CUTLASS_HOST_DEVICE int getThreadsGrid() const {
      return kNumThreadsPerBlock;
    }
  };

  CUTLASS_DEVICE static void run(Params p) {
    // Block-level position
    int block_id = blockIdx.x;
    int thread_row = (block_id % p._getNumBlocksRows()) * kRowsPerBlock;
    int thread_col = (block_id / p._getNumBlocksRows()) * kColsPerBlock;

    // Thread-level position
    int thread_id = threadIdx.x;
    constexpr int kTilingRows = kRowsPerBlock / kNumRows;
    thread_row += (thread_id % kTilingRows) * kNumRows;
    thread_col += (thread_id / kTilingRows) * kNumCols;

    bool enabled = thread_row < p.size0 && thread_col < p.size1;
    if (!enabled) {
      return;
    }

    // We operate on a small 4x4 patch per thread
    using FragmentRow = cutlass::Array<Element, kNumCols>;
    cutlass::Array<ElementWithPos, kNumCols * kNumRows> allValues;
    cutlass::Array<cutlass::uint1b_t, kNumCols * kNumRows> allOutputs;
    allOutputs.clear();
    uint8_t numOutputsPerRow[kNumRows] = {0};
    uint8_t numOutputsPerCol[kNumCols] = {0};
    uint8_t totalAdded = 0;

    CUTLASS_PRAGMA_UNROLL
    for (int row = 0; row < kNumRows; ++row) {
      FragmentRow l = *reinterpret_cast<FragmentRow const*>(
          p.input + thread_col + (thread_row + row) * p.stride0);
      CUTLASS_PRAGMA_UNROLL
      for (int col = 0; col < kNumCols; ++col) {
        allValues[col + row * kNumCols].e = l[col];
        allValues[col + row * kNumCols].row = row;
        allValues[col + row * kNumCols].col = col;
      }
    }

    // Sort - ascending order
    StaticSort<kNumCols * kNumRows> sorter;
    sorter(allValues);

    // Take all the values we can starting with the largest
    int i = allValues.size() - 1;
    CUTLASS_PRAGMA_UNROLL
    for (; i >= 0; --i) {
      if (kHasRandom && totalAdded + p.numRandom == kTakeMaximumCount) {
        break;
      }

      auto const& v = allValues[i];
      if (numOutputsPerRow[v.row] == kMaxPerRow ||
          numOutputsPerCol[v.col] == kMaxPerCol) {
        continue;
      }
      numOutputsPerRow[v.row] += 1;
      numOutputsPerCol[v.col] += 1;
      totalAdded += 1;
      allOutputs[v.col + v.row * kNumCols] = cutlass::uint1b_t(1);
    }

    // Add random elements now
    if (kHasRandom) {
      cutlass::Array<PseudoRandomOrder, kNumCols * kNumRows> randomSorting;
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < randomSorting.size(); ++j) {
        randomSorting[j].row = allValues[j].row;
        randomSorting[j].col = allValues[j].col;
        randomSorting[j].random_score = uint16_t(-1);
        if (j <= i) { // not already considered
          uint16_t seed = allValues[j].e.storage;
          // Assume the lower bits of the significant are more random
          seed = seed & 0x2f;
          randomSorting[j].random_score = (seed ^ (seed << 1)) & 0x7fff;
        }
      }
      sorter(randomSorting);
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < randomSorting.size(); ++j) {
        auto const& v = randomSorting[j];
        if (numOutputsPerRow[v.row] == kMaxPerRow ||
            numOutputsPerCol[v.col] == kMaxPerCol ||
            allOutputs[v.col + v.row * kNumCols].get()) {
          continue;
        }
        numOutputsPerRow[v.row] += 1;
        numOutputsPerCol[v.col] += 1;
        allOutputs[v.col + v.row * kNumCols] = cutlass::uint1b_t(1);
      }
    }

    // Write output
    CUTLASS_PRAGMA_UNROLL
    for (int row = 0; row < kNumRows; ++row) {
      FragmentRow fragmentOut;
      fragmentOut.clear();
      CUTLASS_PRAGMA_UNROLL
      for (int col = 0; col < kNumCols; ++col) {
        fragmentOut[col] =
            allOutputs[col + row * kNumCols].get() ? Element(1) : Element(0);
      }
      *reinterpret_cast<FragmentRow*>(
          p.output + thread_col + (thread_row + row) * p.stride0) = fragmentOut;
    }
  }
};

template <typename Kernel>
__global__ void sparse24_largest_mask_2d_cu(typename Kernel::Params p) {
  Kernel::run(p);
}

template <bool kHasRandom>
torch::stable::Tensor sparse24_largest_with_random_mask_2d_impl(
    const torch::stable::Tensor input,
    int64_t numRandom) {
  STD_TORCH_CHECK(input.is_cuda(), "must be a CUDA tensor");
  STD_TORCH_CHECK(!xf_is_sparse(input), "must be a dense tensor");
  STD_TORCH_CHECK(input.is_contiguous(), "must be contiguous");
  STD_TORCH_CHECK(input.dim() == 2, "only works on 2d tensors");
  STD_TORCH_CHECK(numRandom <= 8, "There are at most 4x2 elements")

  torch::stable::accelerator::DeviceGuard device_guard(input.device().index());
  cudaStream_t stream = xf_getCurrentCUDAStream();
  torch::stable::Tensor output = torch::stable::empty_like(input);

  auto runKernel = [&](auto _) {
    using Element = decltype(_);
    using Kernel = Sp24MaskKernel<Element, kHasRandom>;
    typename Kernel::Params p;
    p.input = (Element*)input.data_ptr();
    p.output = (Element*)output.data_ptr();
    p.stride0 = input.stride(0);
    p.size0 = input.size(0);
    p.size1 = input.size(1);
    p.numRandom = numRandom;
    STD_TORCH_CHECK((input.size(-1) % Kernel::kNumRows) == 0, "Wrong shape");
    STD_TORCH_CHECK((input.size(-2) % Kernel::kNumCols) == 0, "Wrong shape");

    sparse24_largest_mask_2d_cu<Kernel>
        <<<p.getBlocksGrid(), p.getThreadsGrid(), 0, stream>>>(p);
  };
  if (input.scalar_type() == torch::headeronly::ScalarType::Half) {
    runKernel(cutlass::half_t(0));
  } else {
    STD_TORCH_CHECK(
        input.scalar_type() == torch::headeronly::ScalarType::BFloat16,
        "only f16/bf16 supported");
    runKernel(cutlass::bfloat16_t(0));
  }
  return output;
}

torch::stable::Tensor sparse24_largest_mask_2d(
    const torch::stable::Tensor input) {
  return sparse24_largest_with_random_mask_2d_impl<false>(input, 0);
}

torch::stable::Tensor sparse24_largest_with_Krandom_mask_2d(
    const torch::stable::Tensor input,
    int64_t numRandom) {
  return sparse24_largest_with_random_mask_2d_impl<true>(input, numRandom);
}

STABLE_TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl("sparse24_largest_mask_2d", TORCH_BOX(sparse24_largest_mask_2d));
  m.impl(
      "sparse24_largest_with_Krandom_mask_2d",
      TORCH_BOX(sparse24_largest_with_Krandom_mask_2d));
}
