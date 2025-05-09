#pragma once

#include <cutlass/arch/memory.h>
#include <cutlass/array.h>
#include <cutlass/numeric_conversion.h>

#include "static_sort.h"

namespace xformers {
namespace sp24 {

template <typename Element, int kRows_, int kCols_>
struct WarpTensor {
  // This class represents a Tensor sharded across an entire warp,
  // on registers. The sharding is row-major, eg looks like this
  // for a `WarpTensor<T, 8, 32>`:
  // [row 0]   [thread0][thread1]...
  // [row 1]   [thread4][thread5]...
  //...
  // [row 8]   [thread28][thread29]...
  // Each thread would hold 8 values. This format is optimized to
  // load from gmem efficiently (coalescing)

  static constexpr int kRows = kRows_;
  static constexpr int kCols = kCols_;
  // NOTE: Stored in Row-Major
  static constexpr int kElementsPerThread = (kRows * kCols / 32);
  static constexpr int kThreadsPerRow = 32 / kRows;
  static_assert(32 % kRows == 0);

  cutlass::Array<Element, kElementsPerThread> data; // < current thread data
  int lane = threadIdx.x % 32;

  CUTLASS_DEVICE int thread_row() const {
    return lane / kThreadsPerRow;
  }
  CUTLASS_DEVICE int thread_col() const {
    return kElementsPerThread * (lane % kThreadsPerRow);
  }
  // load/store in gmem
  template <typename RowMod>
  CUTLASS_DEVICE void load(
      Element const* ptr,
      int64_t stride0,
      RowMod row_mod) {
    cutlass::arch::global_load<decltype(data), sizeof(data)>(
        data, ptr + stride0 * row_mod(thread_row()) + thread_col(), true);
  }
  CUTLASS_DEVICE void load(Element const* ptr, int64_t stride0) {
    load(ptr, stride0, [](int i) { return i; });
  }
  CUTLASS_DEVICE void store_line(Element* ptr) const {
    cutlass::arch::global_store<decltype(data), sizeof(data)>(
        data, ptr + thread_col(), true);
  }
  CUTLASS_DEVICE void store(Element* ptr, int64_t stride0) const {
    cutlass::arch::global_store<decltype(data), sizeof(data)>(
        data, ptr + stride0 * thread_row() + thread_col(), true);
  }

  // load/store in smem
  template <int kStride0, int kStride1, typename ElementSmem>
  CUTLASS_DEVICE void load_32bits(ElementSmem const* ptr) {
    if constexpr (kStride1 == 1 && std::is_same<Element, ElementSmem>::value) {
      cutlass::Array<ElementSmem, 4 / sizeof(ElementSmem)> frag32;
      static_assert(sizeof(frag32) == 4);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kElementsPerThread / frag32.size(); ++i) {
        frag32 =
            *((decltype(frag32) const*)(ptr + thread_col() + frag32.size() * i +
                                        kStride0 * thread_row()));
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < frag32.size(); ++j) {
          data[frag32.size() * i + j] = Element(frag32[j]);
        }
      }
    } else {
      CUTLASS_PRAGMA_UNROLL
      for (int col = 0; col < data.size(); ++col) {
        data[col] = Element(
            ptr[kStride0 * thread_row() + kStride1 * (thread_col() + col)]);
      }
    }
  }
  template <int kStride0, int kStride1, typename ElementSmem = Element>
  CUTLASS_DEVICE void store_32bits(ElementSmem* ptr) const {
    if constexpr (
        kStride1 == 1 && sizeof(Element) == 2 &&
        std::is_same<Element, ElementSmem>::value) {
      // store packed as 32bits - Row-Major
      uint32_t const* pack_ptr = reinterpret_cast<uint32_t const*>(&data);
      uint32_t* smem_ptr =
          (uint32_t*)(ptr + kStride0 * thread_row() + kStride1 * thread_col());
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < data.size() / 2; ++c) {
        smem_ptr[c] = pack_ptr[c];
      }
    } else if constexpr (
        kStride0 == 1 && sizeof(Element) == 2 && kRows == 2 &&
        kElementsPerThread % 2 == 0 &&
        std::is_same<Element, ElementSmem>::value) {
      // store packed as 32bits - Col-Major
      uint32_t const* pack_ptr = reinterpret_cast<uint32_t const*>(&data);
      bool is_low_thread = threadIdx.x & kThreadsPerRow;
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < data.size() / 2; ++c) {
        uint32_t my_val = pack_ptr[c];
        uint32_t other_val =
            __shfl_xor_sync(0xffffffff, my_val, kThreadsPerRow);
        if (is_low_thread) {
          my_val = (my_val & 0x0000FFFF) | (other_val << 16);
        } else {
          my_val = (my_val & 0xFFFF0000) | (other_val & 0xFFFF);
        }
        uint32_t* smem_ptr =
            (uint32_t*)(ptr +
                        kStride1 * (thread_col() + is_low_thread + 2 * c));
        *smem_ptr = my_val;
      }
    } else {
      // not optimized path
      CUTLASS_PRAGMA_UNROLL
      for (int col = 0; col < data.size(); ++col) {
        ptr[kStride0 * thread_row() + kStride1 * (thread_col() + col)] =
            ElementSmem(data[col]);
      }
    }
  }
  template <typename ElementOut>
  CUTLASS_DEVICE WarpTensor<ElementOut, kRows, kCols> to() const {
    cutlass::NumericArrayConverter<
        ElementOut,
        Element,
        kElementsPerThread,
        cutlass::FloatRoundStyle::round_to_nearest>
        converter;

    WarpTensor<ElementOut, kRows, kCols> out;
    out.data = converter(data);
    return out;
  }

  CUTLASS_DEVICE void print(int offs_row = 0) const {
    for (int i = 0; i < 32; ++i) {
      if (lane == i) {
        printf(
            "[lane=%d][%d, %d:%d] = ",
            int(lane),
            int(thread_row() + offs_row),
            int(thread_col()),
            int(thread_col() + kElementsPerThread));
        for (int j = 0; j < data.size(); ++j) {
          // printf("0x%x ", uint32_t(data[j]));
          printf("%f ", float(data[j]));
        }
        printf("\n");
      }
      __syncthreads();
    }
  }

  template <typename Algo>
  CUTLASS_DEVICE std::tuple<
      WarpTensor<Element, kRows, kCols / 2>,
      WarpTensor<uint8_t, kRows, kCols / 8>>
  sparsify_pack(Algo algo) {
    constexpr int kCount = kElementsPerThread;
    auto dense_values = data;

    WarpTensor<Element, kRows, kCols / 2> tensor_packed;
    WarpTensor<uint8_t, kRows, kCols / 8> tensor_mdata;
    uint8_t metadata = 0;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount / 4; ++i) {
      cutlass::Array<Element, 4> to_sparsify;
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < 4; ++j) {
        to_sparsify[j] = dense_values[4 * i + j].get();
      }
      cutlass::Array<Element, 2> packed2;
      int m = algo(to_sparsify, packed2);
      metadata |= (m << (4 * i));
      tensor_packed.data[2 * i] = packed2[0].get();
      tensor_packed.data[2 * i + 1] = packed2[1].get();
    }
    tensor_mdata.data[0] = metadata;
    return std::make_tuple(tensor_packed, tensor_mdata);
  }

  CUTLASS_DEVICE WarpTensor<Element, kRows, kCols / 2> sparsify_as(
      WarpTensor<uint8_t, kRows, kCols / 8> mdata) const {
    static_assert(sizeof(Element) == 2);
    auto* ptr = reinterpret_cast<uint32_t const*>(&data);

    WarpTensor<Element, kRows, kCols / 2> packed;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerThread / 4; ++i) {
      auto a = ptr[2 * i];
      auto b = ptr[2 * i + 1];
      auto sparseSelect = [&](uint8_t mdata_element) {
        int m0 = mdata_element & 0x1;
        int m1 = (mdata_element >> 1) & 0x1;
        int out = ((a >> (16 * m0)) * (1 - m1) + (b >> (16 * m0)) * m1);
        return reinterpret_cast<Element&>(out);
      };
      uint8_t mdata_i = mdata.data[i / 2].get() >> (4 * (i % 2));
      packed.data[2 * i] = sparseSelect(mdata_i);
      packed.data[2 * i + 1] = sparseSelect(mdata_i >> 2);
    }
    return packed;
  }

  CUTLASS_DEVICE WarpTensor<Element, kRows, kCols * 2> unpack(
      WarpTensor<uint8_t, kRows, kCols * 2 / 8> mdata) const {
    static_assert(sizeof(Element) == 2);

    WarpTensor<Element, kRows, kCols * 2> unpacked;
    auto* ptr_p = reinterpret_cast<uint32_t const*>(&data);
    auto* ptr_unp = reinterpret_cast<uint32_t*>(&unpacked.data);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerThread / 2; ++i) {
      auto packed = ptr_p[i];
      uint32_t p0 = packed & 0xFFFF;
      uint32_t p1 = packed >> 16;
      uint8_t mdata_i = mdata.data[i / 2].get() >> (4 * (i % 2));
      uint32_t m0 = mdata_i & 0x3;
      uint32_t m1 = (mdata_i >> 2) & 0x3;
      p0 = p0 << ((m0 & 1) * 16);
      p1 = p1 << ((m1 & 1) * 16);

      uint32_t unp0 = 0;
      uint32_t unp1 = 0;
      if (m0 & 0x1) {
        unp1 = p0;
      } else {
        unp0 = p0;
      }
      if (m1 & 0x1) {
        unp1 += p1;
      } else {
        unp0 += p1;
      }
      ptr_unp[2 * i] = unp0;
      ptr_unp[2 * i + 1] = unp1;
    }
    return unpacked;
  }

  template <typename BinaryOp>
  CUTLASS_DEVICE std::tuple<
      cutlass::Array<Element, kCols / 32>, // reduce elements
      uint32_t // thread offset
      >
  all_reduce(BinaryOp binary_op) const {
    // reduces across the first dimension (eg `out[i,k]=out[j,k]`)
    WarpTensor<Element, kRows, kCols> red;
    red.data = data;

    CUTLASS_PRAGMA_UNROLL
    for (int xor_lane = kThreadsPerRow; xor_lane < 32; xor_lane *= 2) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < red.data.size(); ++i) {
        Element other_val = Element(
            __shfl_xor_sync(0xffffffff, Element(red.data[i]), xor_lane));
        red.data[i] = binary_op(red.data[i], other_val);
      }
    }

    uint32_t offset = thread_col();
    cutlass::Array<Element, kCols / 32> out;
    if constexpr (kThreadsPerRow == 16) {
      static constexpr int kOffset = kElementsPerThread / 2;
      if (thread_row() == 1) {
        offset += kOffset;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kOffset; ++i) {
          out[i] = red.data[i + kOffset];
        }
      } else {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kOffset; ++i) {
          out[i] = red.data[i];
        }
      }
    } else {
      static_assert(kThreadsPerRow == 16); // Only supported in that case
    }

    return std::make_tuple(out, offset);
  }

  template <typename BinaryOp>
  CUTLASS_DEVICE Element reduce_line(BinaryOp binary_op) const {
    Element reduced = data[0];
    // local reduction
    CUTLASS_PRAGMA_UNROLL
    for (int i = 1; i < data.size(); ++i) {
      reduced = binary_op(reduced, Element(data[i]));
    }

    // reduce with other lanes
    CUTLASS_PRAGMA_UNROLL
    for (int xor_lane = 1; xor_lane < kThreadsPerRow; xor_lane *= 2) {
      Element other_val =
          Element(__shfl_xor_sync(0xffffffff, reduced, xor_lane));
      reduced = binary_op(reduced, other_val);
    }
    return reduced;
  }

  struct TileValueOrdered1d {
    union {
      struct {
        Element value;
        uint16_t pos;
      } parts;
      uint32_t raw;
    };
    CUTLASS_DEVICE bool operator<(TileValueOrdered1d const& other) const {
      return parts.value < other.parts.value;
    }
    CUTLASS_DEVICE TileValueOrdered1d() {}
  };

  template <int N, int M, typename SortPreproc>
  CUTLASS_DEVICE WarpTensor<Element, kRows, kCols> sparsify_dense(
      SortPreproc sort_preproc) const {
    static_assert(M == kElementsPerThread);

    WarpTensor<Element, kRows, kCols> out;

    cutlass::Array<TileValueOrdered1d, M> values_ordered;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < data.size(); ++i) {
      auto& v = values_ordered[i];
      v.parts.value = sort_preproc(data[i].get());
      v.parts.pos = i;
    }
    StaticSort<M> sorter;
    sorter(values_ordered);

    // mask out smallest elements
    uint32_t kept_mask = 0;
    CUTLASS_PRAGMA_UNROLL
    for (int i = M - N; i < M; ++i) {
      kept_mask |= (1 << values_ordered[i].parts.pos);
    }
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < M; ++i) {
      if (kept_mask & 0x1) {
        out.data[i] = data[i].get();
      } else {
        out.data[i] = Element(0);
      }
      kept_mask = kept_mask >> 1;
    }
    return out;
  }
};

CUTLASS_DEVICE void store_metadata_reordered(
    WarpTensor<uint8_t, 16, 32 / 8> mdata_tensor,
    uint8_t* mdata_ptr) {
  // This function is explained in
  // https://docs.google.com/spreadsheets/d/1JvEsw9QnoIvXctOnED3Gk0LFnIe8XJTnbCRamxvfFBw/edit?gid=1603247130#gid=1603247130
  auto lane = mdata_tensor.lane;
  static_assert(mdata_tensor.kElementsPerThread == 2);

  uint16_t mdata_step0 = reinterpret_cast<uint16_t const&>(mdata_tensor.data);
  uint16_t other_step0 = __shfl_xor_sync(0xffffffff, mdata_step0, 16);

  // step1
  uint16_t mdata_step1 = 0;
  if (lane & 16) { // T16-T31
    mdata_step1 = ((mdata_step0 & 0xF0F0) | ((other_step0 >> 4) & 0x0F0F));
  } else { // T0-T15
    mdata_step1 = ((mdata_step0 & 0x0F0F) | ((other_step0 << 4) & 0xF0F0));
  }

  // step2
  uint16_t other_step1 = __shfl_xor_sync(0xffffffff, mdata_step1, 1);
  uint16_t mdata_gmem = 0;
  if (lane & 1) { // T1
    mdata_gmem = ((mdata_step1 & 0xFF00) | ((other_step1 >> 8) & 0x00FF));
  } else { // T0
    mdata_gmem = ((mdata_step1 & 0x00FF) | ((other_step1 << 8) & 0xFF00));
  }

  // read to store to gmem
  cutlass::arch::global_store<decltype(mdata_gmem), sizeof(mdata_gmem)>(
      mdata_gmem,
      mdata_ptr + (lane % 2) * 4 + ((lane % 16) / 2) * 8 + (lane / 16) * 2,
      true);
}

struct Identity {
  template <typename T>
  T CUTLASS_DEVICE operator()(T x) const {
    return x;
  }
};

template <
    int kSmemStride0,
    int kSmemStride1,
    int kNumRows,
    int kWarpsPerCTA,
    typename Algo,
    typename Element,
    typename PointwiseFn>
CUTLASS_DEVICE void warp_dump_sparse_and_dense_from_smem_32cols(
    Element const* smem,
    Algo algo,
    int32_t const* destination_idx_ptr,
    // sparse part
    uint8_t* sparse_bitmask_ptr,
    int64_t sparse_bitmask_s0,
    int64_t sparse_bitmask_s1,
    Element* sparse_packed_ptr,
    int64_t sparse_packed_s0,
    // dense part
    Element* dense_ptr,
    int64_t dense_s0,
    PointwiseFn pointwise_fn = Identity()) {
  // 64x32 data is layed out like:
  // row 0: [T0 (128 bits)][T1 (128 bits)][T2 (128 bits)]...
  // row 1: [T4 (128 bits)][T5 (128 bits)]...
  // ...
  // row 8: [T0 (128 bits)][T1 (128 bits)]...
  // ..
  WarpTensor<Element, 8, 32> tensor;

  cutlass::Array<int32_t, kNumRows / 8 / kWarpsPerCTA> destination_idx_array;
  int warp_row = (threadIdx.x / 32) * tensor.kRows;
  CUTLASS_PRAGMA_UNROLL
  for (int row = 0; row < kNumRows; row += kWarpsPerCTA * tensor.kRows) {
    cutlass::arch::global_load<int32_t, sizeof(destination_idx_array[0])>(
        destination_idx_array[row / (kWarpsPerCTA * tensor.kRows)],
        destination_idx_ptr + tensor.thread_row() + row + warp_row,
        true);
  }

  CUTLASS_PRAGMA_UNROLL
  for (int row = 0; row < kNumRows; row += kWarpsPerCTA * tensor.kRows) {
    tensor.template load_32bits<kSmemStride0, kSmemStride1>(
        smem + kSmemStride0 * (row + warp_row));
    tensor.data = pointwise_fn(tensor.data);
    // RF -> RF (sparsify)
    auto [packed, bitmask] = tensor.sparsify_pack(algo);
    int32_t destination_idx =
        destination_idx_array[row / (kWarpsPerCTA * tensor.kRows)];
    if (destination_idx >= 0) {
      // shape: [cols/32, rows, 32/8] (b8)
      int64_t coord0 = (tensor.thread_col()) / 32;
      int64_t coord1 = destination_idx;
      int64_t coord2 = (tensor.thread_col() % 32) / 8;
      sparse_bitmask_ptr
          [coord0 * sparse_bitmask_s0 + coord1 * sparse_bitmask_s1 + coord2] =
              bitmask.data[0];
      packed.store_line(sparse_packed_ptr + sparse_packed_s0 * destination_idx);
    } else {
      destination_idx = -(destination_idx + 1);
      tensor.store_line(dense_ptr + dense_s0 * destination_idx);
    }
  }
}
} // namespace sp24
} // namespace xformers
