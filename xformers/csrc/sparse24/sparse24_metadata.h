#pragma once
#include <torch/library.h>
#include <torch/types.h>
#include "compute_sparse_tile.h"

namespace xformers {
namespace sp24 {
template <typename ElementCutlass>
struct CutlassToAt;

template <>
struct CutlassToAt<cutlass::half_t> {
  static auto constexpr value = at::ScalarType::Half;
};
template <>
struct CutlassToAt<cutlass::bfloat16_t> {
  static auto constexpr value = at::ScalarType::BFloat16;
};
template <>
struct CutlassToAt<cutlass::float_e4m3_t> {
  static auto constexpr value = at::ScalarType::Float8_e4m3fn;
};
template <>
struct CutlassToAt<uint16_t> {
  static auto constexpr value = at::ScalarType::UInt16;
};
template <>
struct CutlassToAt<int32_t> {
  static auto constexpr value = at::ScalarType::Int;
};
template <>
struct CutlassToAt<uint8_t> {
  static auto constexpr value = at::ScalarType::Byte;
};
template <>
struct CutlassToAt<float> {
  static auto constexpr value = at::ScalarType::Float;
};

struct MetadataCuSparseLtSm80 {
  // Format used by cuSparseLt
  // This is based on reverse-engineering, for a visual illustration:
  // https://docs.google.com/presentation/d/1DtmKThv8S5QAyBktuLRYzZhRzCvS1qSkBbrqNCjMPeA/edit#slide=id.g29afe95bda8_0_0
  static constexpr int kStrideBlock32x32 =
      (32 * 32) / (sizeof(ElementInputE) * 8);

  ElementInputE* _meta;
  ElementInputE* _meta_trans;
  int64_t _rows;
  int64_t _cols;

  static int64_t getMetadataSize(int rows, int cols) {
    TORCH_CHECK(
        rows % 128 == 0 && cols % 128 == 0,
        "Only supports rows/cols multiples of 128");
    // 1 bit per dense value
    return (rows * cols) / (8 * sizeof(ElementInputE));
  }
  static std::tuple<
      at::Tensor, // return value of the function
      at::Tensor, // packed
      at::Tensor // packed_meta
      >
  create_compressed_representation(
      int rows,
      int cols,
      at::Tensor const& like,
      bool needs_metadata) {
    TORCH_CHECK(
        like.scalar_type() == at::ScalarType::Half ||
        like.scalar_type() == at::ScalarType::BFloat16);
    constexpr int kBytesPerScalar = 2;
    int64_t data_scalars = rows * cutlass::ceil_div(cols, 2);
    int64_t meta_scalars = getMetadataSize(rows, cols);

    at::Tensor storage = at::empty(
        {(data_scalars + meta_scalars)},
        at::TensorOptions().device(like.device()).dtype(like.dtype()));
    using namespace torch::indexing;
    at::Tensor packed = storage.index({Slice(None, data_scalars)})
                            .view({rows, cutlass::ceil_div(cols, 2)});
    at::Tensor metadata = storage.index({Slice(data_scalars, None)});
    // TODO: Cast metadata to Short
    static_assert(kBytesPerScalar == 2, "or modify the last dim below");
    metadata = metadata.view({rows / 128, cols / 32, 256});
    return std::make_tuple(storage, packed, metadata);
  }
  MetadataCuSparseLtSm80(
      at::Tensor metaN,
      at::Tensor metaT,
      int rows,
      int cols) {
    _meta = (ElementInputE*)metaN.data_ptr();
    _meta_trans = (ElementInputE*)metaT.data_ptr();
    _rows = rows;
    _cols = cols;
  }
  CUTLASS_HOST_DEVICE
  static int64_t _get_meta_offset(
      int warp_row,
      int thread_row,
      int warp_col,
      int thread_col,
      int totalRows) {
    int64_t offset = 0;
    // warp-level: Find the 128x64 tile
    offset += (warp_row / 128) * (kStrideBlock32x32 * 8);
    offset += (warp_col / 64) * (kStrideBlock32x32 * 8) * (totalRows / 128);
    // Find the 32x32 tile inside
    offset += (((warp_row + thread_row) % 128) / 32) * kStrideBlock32x32;
    offset += (((warp_col + thread_col) % 64) / 32) * (kStrideBlock32x32 * 4);
    // Inside the 32x32 tile
    offset += (warp_row % 32) * 2;
    // Top/bottom 16x16 tile
    offset += ((thread_row % 32) / 16) * 4;
    // Left/right 16x16 tile
    offset += ((thread_col % 32) / 16) * 2;
    return offset;
  }
  CUTLASS_HOST_DEVICE
  ElementInputE* get_metaN(
      int warp_row,
      int thread_row,
      int warp_col,
      int thread_col) const {
    return _meta +
        _get_meta_offset(warp_row, thread_row, warp_col, thread_col, _rows);
  }
  CUTLASS_HOST_DEVICE
  ElementInputE* get_metaT(
      int warp_row,
      int thread_row,
      int warp_col,
      int thread_col) const {
    return _meta_trans +
        _get_meta_offset(warp_col, thread_col, warp_row, thread_row, _cols);
  }
};

struct MetadataCutlassSm80 {
  // Layout needed to run 2:4 gemms in CUTLASS
  // There is basically a hardware specific value for every
  // 32x32 dense tile (1024 bits). Then these tiles are
  // stored in a Column-Major fashion
  ElementInputE* _meta;
  ElementInputE* _meta_trans;
  int64_t _meta_reordered_sy;
  int64_t _meta_trans_reordered_sx;

  static std::tuple<
      at::Tensor, // return value of the function
      at::Tensor, // packed
      at::Tensor // packed_meta
      >
  create_compressed_representation(
      int rows,
      int cols,
      at::Tensor const& like,
      bool needs_metadata) {
    TORCH_CHECK(
        like.scalar_type() == at::ScalarType::Half ||
        like.scalar_type() == at::ScalarType::BFloat16);
    auto roundedx = cutlass::round_up(rows, kWarpX);
    auto roundedy = cutlass::round_up(cols, kWarpY);

    // NB: Writing to `packed` tensors in transposed manner
    at::Tensor packed =
        at::empty({roundedx, cutlass::ceil_div(roundedy, 2)}, like.options());
    at::Tensor packed_meta;
    if (needs_metadata) {
      packed_meta = at::empty(
                        {roundedx * roundedy / 16},
                        like.options().dtype(at::ScalarType::Short))
                        .view({roundedy / 32, roundedx, 2})
                        .permute({1, 2, 0});
    }
    return std::make_tuple(packed, packed, packed_meta);
  }
  MetadataCutlassSm80(at::Tensor metaN, at::Tensor metaT, int rows, int cols) {
    _meta = (ElementInputE*)metaN.data_ptr();
    _meta_reordered_sy = metaN.stride(2);
    _meta_trans = (ElementInputE*)metaT.data_ptr();
    _meta_trans_reordered_sx = metaT.stride(2);
  }
  CUTLASS_HOST_DEVICE
  int64_t _get_meta_offset(
      int warp_row,
      int thread_row,
      int warp_col,
      int thread_col,
      int64_t stride) const {
    int64_t offset = 0;
    offset += warp_row * 2 + (warp_col / 32) * stride;
    // A single warp is 32x64. The right 32x32 tile is at a different position
    offset += 64 * (thread_row / 32);
    offset += (thread_col / 32) * stride;
    // Top/bottom 16x16 tile
    offset += ((thread_row % 32) / 16) * 4;
    // Left/right 16x16 tile
    offset += ((thread_col % 32) / 16) * 2;
    return offset;
  }
  CUTLASS_HOST_DEVICE
  ElementInputE* get_metaN(
      int warp_row,
      int thread_row,
      int warp_col,
      int thread_col) const {
    return _meta +
        _get_meta_offset(
               warp_row, thread_row, warp_col, thread_col, _meta_reordered_sy);
  }
  CUTLASS_HOST_DEVICE
  ElementInputE* get_metaT(
      int warp_row,
      int thread_row,
      int warp_col,
      int thread_col) const {
    return _meta_trans +
        _get_meta_offset(
               warp_col,
               thread_col,
               warp_row,
               thread_row,
               _meta_trans_reordered_sx);
  }
};

struct MetadataCutlass8bitsSm90 {
  template <typename ElementOut>
  static std::tuple<at::Tensor, at::Tensor> createTensors(at::Tensor input) {
    auto n_rows = input.size(0);
    auto n_cols = input.size(1);
    TORCH_CHECK(n_cols % 128 == 0); // aligned metadata
    TORCH_CHECK(n_rows % 64 == 0); // aligned metadata
    int mdata_bytes = n_rows * n_cols / 8;

    at::Tensor packed = at::empty(
        {n_rows, n_cols / 2},
        input.options().dtype(CutlassToAt<ElementOut>::value));
    at::Tensor mdata =
        at::empty({mdata_bytes}, input.options().dtype(at::ScalarType::Byte));
    return std::make_tuple(packed, mdata);
  }
  static CUTLASS_HOST_DEVICE int64_t
  mdataBlockPtrOffset(int row, int col, int64_t n_rows) {
    constexpr int kStrideRow = 16;
    return row * kStrideRow + (col / 128 * n_rows * 16) + (col % 128) / 8;
  }
};

struct MetadataCusparseLt16bitsSm90 {
  template <typename ElementOut>
  static std::tuple<at::Tensor, at::Tensor> createTensors(at::Tensor input) {
    auto n_rows = input.size(0);
    auto n_cols = input.size(1);
    int packed_elements = n_rows * n_cols / 2;
    int mdata_bytes = n_rows * n_cols / 8;

    // We assume 2 bytes per element
    at::Tensor sparse_packed = at::empty(
        {int64_t(packed_elements + mdata_bytes / sizeof(ElementOut))},
        input.options().dtype(CutlassToAt<ElementOut>::value));
    using namespace torch::indexing;
    return std::make_tuple(
        sparse_packed,
        sparse_packed.index({Slice(packed_elements, None)})
            .view(at::ScalarType::Byte));
  }
};

} // namespace sp24
} // namespace xformers
