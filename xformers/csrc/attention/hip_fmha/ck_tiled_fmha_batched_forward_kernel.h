#pragma once

#include "ck/tensor/tensor_view.hpp"
#include "ck/tile_program/tile/tile_window.hpp"
#include "ck/utility/common_header.hpp"

// S[seqlen_q, seqlen_k] = Q[seqlen_q, hdim_q] * K[seqlen_k, hdim_q]
// P[seqlen_q, seqlen_k] = Softmax(S[seqlen_q, seqlen_k])
// O[seqlen_q, hdim_v] = P[seqlen_q, seqlen_k] * V[hdim_v, seqlen_k]

#define C_LOG2E 1.44269504088896340736 // log2(e)

template <
    typename TilePartitioner_,
    typename FmhaPipeline_,
    typename EpiloguePipeline_>
struct FmhaFwdKernel {
  using TilePartitioner = ck::remove_cvref_t<TilePartitioner_>;
  using FmhaPipeline = ck::remove_cvref_t<FmhaPipeline_>;
  using EpiloguePipeline = ck::remove_cvref_t<EpiloguePipeline_>;
  static constexpr ck::index_t kBlockSize = FmhaPipeline::kBlockSize;

  using QDataType = ck::remove_cvref_t<typename FmhaPipeline::QDataType>;
  using KDataType = ck::remove_cvref_t<typename FmhaPipeline::KDataType>;
  using VDataType = ck::remove_cvref_t<typename FmhaPipeline::VDataType>;
  using ODataType = ck::remove_cvref_t<typename FmhaPipeline::ODataType>;

  using VLayout = ck::remove_cvref_t<typename FmhaPipeline::VLayout>;

  struct Kargs {
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    void* o_ptr;
    ck::index_t seqlen_q;
    ck::index_t seqlen_k;
    ck::index_t hdim_q;
    ck::index_t hdim_v;

    float scale;

    ck::index_t stride_q;
    ck::index_t stride_k;
    ck::index_t stride_v;
    ck::index_t stride_o;

    ck::index_t nhead_stride_q;
    ck::index_t nhead_stride_k;
    ck::index_t nhead_stride_v;
    ck::index_t nhead_stride_o;

    ck::index_t batch_stride_q;
    ck::index_t batch_stride_k;
    ck::index_t batch_stride_v;
    ck::index_t batch_stride_o;
  };

  __host__ static constexpr Kargs MakeKargs(
      const void* q_ptr,
      const void* k_ptr,
      const void* v_ptr,
      void* o_ptr,
      ck::index_t seqlen_q,
      ck::index_t seqlen_k,
      ck::index_t hdim_q,
      ck::index_t hdim_v,
      float scale,
      ck::index_t stride_q,
      ck::index_t stride_k,
      ck::index_t stride_v,
      ck::index_t stride_o,
      ck::index_t nhead_stride_q,
      ck::index_t nhead_stride_k,
      ck::index_t nhead_stride_v,
      ck::index_t nhead_stride_o,
      ck::index_t batch_stride_q,
      ck::index_t batch_stride_k,
      ck::index_t batch_stride_v,
      ck::index_t batch_stride_o) {
    return Kargs{q_ptr,          k_ptr,          v_ptr,          o_ptr,
                 seqlen_q,       seqlen_k,       hdim_q,         hdim_v,
                 scale,          stride_q,       stride_k,       stride_v,
                 stride_o,       nhead_stride_q, nhead_stride_k, nhead_stride_v,
                 nhead_stride_o, batch_stride_q, batch_stride_k, batch_stride_v,
                 batch_stride_o};
  }

  __host__ static constexpr auto GridSize(
      ck::index_t batch_size_,
      ck::index_t nhead_,
      ck::index_t seqlen_q_,
      ck::index_t hdim_v_) {
    return TilePartitioner::GridSize(batch_size_, nhead_, seqlen_q_, hdim_v_);
  }

  __host__ static constexpr auto BlockSize() {
    return dim3(kBlockSize);
  }

  __host__ __device__ static constexpr ck::index_t GetSmemSize() {
    return ck::math::max(
        FmhaPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
  }

  __device__ void operator()(Kargs kargs) const {
    using namespace ck;
    using namespace ck::tile_program;
    using namespace ck::tile_program::block;

    // allocate LDS
    __shared__ char smem_ptr[GetSmemSize()];

    // divide problem
    const auto [i_tile_m, i_tile_n, i_nhead, i_batch] =
        TilePartitioner{}(kargs.seqlen_q, kargs.hdim_v);

    const index_t i_m0 =
        __builtin_amdgcn_readfirstlane(i_tile_m * FmhaPipeline::kM0);
    const index_t i_n1 =
        __builtin_amdgcn_readfirstlane(i_tile_n * FmhaPipeline::kN1);

    // for simplicity, batch stride we just modify the pointer
    const QDataType* q_ptr = reinterpret_cast<const QDataType*>(kargs.q_ptr) +
        i_nhead * kargs.nhead_stride_q + i_batch * kargs.batch_stride_q;
    const KDataType* k_ptr = reinterpret_cast<const KDataType*>(kargs.k_ptr) +
        i_nhead * kargs.nhead_stride_k + i_batch * kargs.batch_stride_k;
    const VDataType* v_ptr = reinterpret_cast<const VDataType*>(kargs.v_ptr) +
        i_nhead * kargs.nhead_stride_v + i_batch * kargs.batch_stride_v;
    ODataType* o_ptr = reinterpret_cast<ODataType*>(kargs.o_ptr) +
        i_nhead * kargs.nhead_stride_o + i_batch * kargs.batch_stride_o;

    // Q/K/V DRAM and DRAM window
    const auto q_dram = make_naive_tensor_view<AddressSpaceEnum::Global>(
        q_ptr,
        make_tuple(kargs.seqlen_q, kargs.hdim_q),
        make_tuple(kargs.stride_q, 1),
        Number<32>{},
        Number<1>{});

    const auto k_dram = make_naive_tensor_view<AddressSpaceEnum::Global>(
        k_ptr,
        make_tuple(kargs.seqlen_k, kargs.hdim_q),
        make_tuple(kargs.stride_k, 1),
        Number<32>{},
        Number<1>{});

    const auto v_dram = [&]() {
      if constexpr (ck::is_same_v<VLayout, ck::tensor_layout::gemm::RowMajor>) {
        const auto v_dram_tmp =
            make_naive_tensor_view<AddressSpaceEnum::Global>(
                v_ptr,
                make_tuple(kargs.seqlen_k, kargs.hdim_v),
                make_tuple(kargs.stride_v, 1),
                Number<32>{},
                Number<1>{});
        return transform_tensor_view(
            v_dram_tmp,
            make_tuple(
                make_pass_through_transform(kargs.hdim_v),
                make_pass_through_transform(kargs.seqlen_k)),
            make_tuple(Sequence<1>{}, Sequence<0>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));
      } else {
        return make_naive_tensor_view<AddressSpaceEnum::Global>(
            v_ptr,
            make_tuple(kargs.hdim_v, kargs.seqlen_k),
            make_tuple(kargs.stride_v, 1),
            Number<32>{},
            Number<1>{});
      }
    }();

    auto q_dram_window = make_tile_window(
        q_dram,
        [&]() {
          if constexpr (FmhaPipeline::kQLoadOnce)
            return make_tuple(
                Number<FmhaPipeline::kM0>{},
                Number<FmhaPipeline::kK0BlockLength>{});
          else
            return make_tuple(
                Number<FmhaPipeline::kM0>{}, Number<FmhaPipeline::kK0>{});
        }(),
        {i_m0, 0});

    auto k_dram_window = make_tile_window(
        k_dram,
        make_tuple(Number<FmhaPipeline::kN0>{}, Number<FmhaPipeline::kK0>{}),
        {0, 0});

    auto v_dram_window = make_tile_window(
        v_dram,
        make_tuple(Number<FmhaPipeline::kN1>{}, Number<FmhaPipeline::kK1>{}),
        {i_n1, 0});

    auto o_acc_tile = FmhaPipeline{}(
        q_dram_window,
        k_dram_window,
        v_dram_window,
        kargs.scale,
        kargs.seqlen_k / FmhaPipeline::kN0,
        kargs.hdim_q / FmhaPipeline::kK0,
        smem_ptr);

    // O DRAM and O DRAM window
    auto o_dram = make_naive_tensor_view<AddressSpaceEnum::Global>(
        o_ptr,
        make_tuple(kargs.seqlen_q, kargs.hdim_v),
        make_tuple(kargs.stride_o, 1),
        Number<32>{},
        Number<1>{});

    auto o_dram_window = make_tile_window(
        o_dram,
        make_tuple(Number<FmhaPipeline::kM0>{}, Number<FmhaPipeline::kN1>{}),
        {i_m0, i_n1});

    EpiloguePipeline{}(o_dram_window, o_acc_tile);
  }
};
