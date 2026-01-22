#include <cutlass/array.h>
#include <cutlass/bfloat16.h>
#include <cutlass/coord.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/half.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/tensor_ref.h>
#include <cutlass/tensor_view.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/core/TensorAccessor.h>

#include "pt_stable_utils.h"
#include "static_sort.h"

namespace {
// This is for 2:4 f16
using ElementInputE = uint16_t;
using LayoutInputE = cutlass::layout::RowMajor;
using ReorderedLayoutInputE = cutlass::layout::ColumnMajorInterleaved<2>;

using RefInp = typename cutlass::TensorRef<ElementInputE, LayoutInputE>;
using RefReordered =
    typename cutlass::TensorRef<ElementInputE, ReorderedLayoutInputE>;

torch::stable::Tensor _sparse24_pack_mask(const torch::stable::Tensor input) {
  STD_TORCH_CHECK(input.is_contiguous(), "Expected contiguous tensor");
  STD_TORCH_CHECK(input.dim() == 2, "Expected 2d tensor");
  STD_TORCH_CHECK(
      input.size(0) % 32 == 0 && input.size(1) % 32 == 0,
      "Wrong dim, should be dividable by 32");
  STD_TORCH_CHECK(
      input.scalar_type() == torch::headeronly::ScalarType::Bool,
      "Expected bool Tensor");

  torch::stable::Tensor packed = torch::stable::new_empty(
      input,
      {input.size(0), input.size(1) / 16},
      torch::headeronly::ScalarType::Short);
  auto input_a = torch::headeronly::HeaderOnlyTensorAccessor<bool, 2>(
      input.mutable_data_ptr<bool>(),
      input.sizes().data(),
      input.strides().data());
  auto packed_a = torch::headeronly::HeaderOnlyTensorAccessor<int16_t, 2>(
      packed.mutable_data_ptr<int16_t>(),
      packed.sizes().data(),
      packed.strides().data());
  for (int row = 0; row < input.size(0); ++row) {
    for (int col_s = 0; col_s < input.size(1); col_s += 16) {
      ElementInputE out = 0;
      for (int bit_shifts = 0; bit_shifts < 16; bit_shifts += 4) {
        int first_pos = -1;
        int second_pos = -1;
        for (int i = 0; i < 4; ++i) {
          if (input_a[row][col_s + bit_shifts + i]) {
            if (first_pos == -1) {
              first_pos = i;
            } else if (second_pos == -1) {
              second_pos = i;
            } else {
              STD_TORCH_CHECK(
                  second_pos != -1,
                  "Invalid mask at (",
                  row,
                  ", ",
                  col_s + bit_shifts,
                  "): too many values");
            }
          }
        }
        STD_TORCH_CHECK(
            second_pos != -1,
            "Invalid mask at (",
            row,
            ", ",
            col_s + bit_shifts,
            "): not enough values");
        out |= (first_pos | (second_pos * 4)) << bit_shifts;
      }
      packed_a[row][col_s / 16] = out;
    }
  }
  return packed;
}

// Taken from <cutlass/tools/util/include/cutlass/util/host_reorder.h>
// Can't include it directly as we have compilation errors...
template <typename Element, typename LayoutDest, typename LayoutSrc>
void reorder_meta(
    cutlass::TensorRef<Element, LayoutDest> dest,
    cutlass::TensorRef<Element, LayoutSrc> src,
    cutlass::gemm::GemmCoord problem_size) {
  for (int m = 0; m < problem_size.m(); m++) {
    for (int k = 0; k < problem_size.k(); k++) {
      // First reorder the rows.
      int group = (sizeof(Element) == 2) ? 32 : 16;
      int interweave = (sizeof(Element) == 2) ? 4 : 2;

      int dest_row = m / group * group + (m % 8) * interweave + (m % group) / 8;
      int dest_col = k;

      // Next swizzle the 2x2 blocks from Z to N.
      if (((dest_row % 2) == 0) && ((dest_col % 2) == 1)) {
        ++dest_row;
        --dest_col;
      } else if (((dest_row % 2) == 1) && ((dest_col % 2) == 0)) {
        --dest_row;
        ++dest_col;
      }

      dest.at({dest_row, dest_col}) = src.at({m, k});
    }
  }
}

torch::stable::Tensor _sparse24_reorder_meta(torch::stable::Tensor input) {
  STD_TORCH_CHECK(input.dim() == 2, "Expected 2d tensor");
  STD_TORCH_CHECK(input.size(0) % 32 == 0, "Wrong dim0");
  STD_TORCH_CHECK(input.size(1) % 2 == 0, "Wrong dim1");
  STD_TORCH_CHECK(
      input.scalar_type() == torch::headeronly::ScalarType::Short,
      "Expected int16 tensor");
  input = xf_contiguous(input);
  cutlass::gemm::GemmCoord problem_size(input.size(0), 0, input.size(1));

  cutlass::MatrixCoord meta_dim{input.size(0), input.size(1)};
  auto reordered_layout = ReorderedLayoutInputE::packed(meta_dim);
  torch::stable::Tensor reordered =
      torch::stable::new_empty(input, {reordered_layout.capacity(meta_dim)});

  RefInp ref_inp{(uint16_t*)input.data_ptr(), LayoutInputE(input.stride(0))};
  RefReordered ref_reordered{(uint16_t*)reordered.data_ptr(), reordered_layout};

  reorder_meta(ref_reordered, ref_inp, problem_size);
  return xf_permute(
      torch::stable::view(reordered, {input.size(1) / 2, input.size(0), 2}),
      {1, 2, 0});
}

torch::stable::Tensor _sparse24_pack_tensor_according_to_mask(
    torch::stable::Tensor a,
    torch::stable::Tensor meta_reordered) {
  STD_TORCH_CHECK(a.dim() == 2, "Expected 2d tensor");
  STD_TORCH_CHECK(a.size(0) % 32 == 0, "Wrong dim0");
  STD_TORCH_CHECK(a.size(1) % 4 == 0, "Wrong dim1");
  STD_TORCH_CHECK(
      meta_reordered.dim() == 3, "Expected meta to be reordered already");

  torch::stable::Tensor a_packed =
      torch::stable::new_empty(a, {a.size(0), a.size(1) / 2});
  cutlass::MatrixCoord meta_dim{
      meta_reordered.size(0), meta_reordered.size(1) * meta_reordered.size(2)};
  auto reordered_layout = ReorderedLayoutInputE::packed(meta_dim);
  torch::stable::Tensor reordered =
      torch::stable::new_empty(a, {reordered_layout.capacity(meta_dim)});
  RefReordered ref_meta_reordered{
      (uint16_t*)meta_reordered.data_ptr(), reordered_layout};
  RefInp ref_a{(uint16_t*)a.data_ptr(), LayoutInputE(a.stride(0))};
  RefInp ref_a_packed{
      (uint16_t*)a_packed.data_ptr(), LayoutInputE(a_packed.stride(0))};

  for (int m = 0; m < a.size(0); m++) {
    for (int k = 0; k < a.size(1) / 16; k++) {
      // First reorder the rows.
      int group = (sizeof(ElementInputE) == 2) ? 32 : 16;
      int interweave = (sizeof(ElementInputE) == 2) ? 4 : 2;

      int dest_row = m / group * group + (m % 8) * interweave + (m % group) / 8;
      int dest_col = k;

      // Next swizzle the 2x2 blocks from Z to N.
      if (((dest_row % 2) == 0) && ((dest_col % 2) == 1)) {
        ++dest_row;
        --dest_col;
      } else if (((dest_row % 2) == 1) && ((dest_col % 2) == 0)) {
        --dest_row;
        ++dest_col;
      }

      uint16_t pack_info = ref_meta_reordered.at({dest_row, dest_col});
      // For each group of 4, read the only 2 that are selected in the mask
      for (int group_shift = 0; group_shift < 16; group_shift += 4) {
        int pos0 = (pack_info >> group_shift) & 3;
        int pos1 = (pack_info >> (group_shift + 2)) & 3;
        ref_a_packed.at({m, 8 * k + group_shift / 2}) =
            ref_a.at({m, 16 * k + group_shift + pos0});
        ref_a_packed.at({m, 8 * k + group_shift / 2 + 1}) =
            ref_a.at({m, 16 * k + group_shift + pos1});
      }
    }
  }
  return a_packed;
}
} // namespace

STABLE_TORCH_LIBRARY_IMPL(xformers, CPU, m) {
  m.impl("_sparse24_pack_mask", TORCH_BOX(_sparse24_pack_mask));
  m.impl("_sparse24_reorder_meta", TORCH_BOX(_sparse24_reorder_meta));
  m.impl(
      "_sparse24_pack_tensor_according_to_mask",
      TORCH_BOX(_sparse24_pack_tensor_according_to_mask));
}
