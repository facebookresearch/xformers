/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <optional>
#include <type_traits>

#include "ck/utility/common_header.hpp"
#include "ck/tensor/tensor_view.hpp"
#include "ck/tile_program/tile/tile_window.hpp"

// S[seqlen_q, seqlen_k] = Q[seqlen_q, hdim_q] * K[seqlen_k, hdim_q]
// S'[seqlen_q, seqlen_k] = S[seqlen_q, seqlen_k] * Scale[1]
// S''[seqlen_q, seqlen_k] = S'[seqlen_q, seqlen_k] + Bias[seqlen_q, seqlen_k]
// P[seqlen_q, seqlen_k] = Softmax(S[seqlen_q, seqlen_k])
// O[seqlen_q, hdim_v] = P[seqlen_q, seqlen_k] * V[hdim_v, seqlen_k]

#define C_LOG2E 1.44269504088896340736 // log2(e)

template <typename TilePartitioner_, typename FmhaPipeline_, typename EpiloguePipeline_>
struct FmhaFwdKernel
{
    using TilePartitioner                   = ck::remove_cvref_t<TilePartitioner_>;
    using FmhaPipeline                      = ck::remove_cvref_t<FmhaPipeline_>;
    using EpiloguePipeline                  = ck::remove_cvref_t<EpiloguePipeline_>;
    static constexpr ck::index_t kBlockSize = FmhaPipeline::kBlockSize;

    using QDataType    = ck::remove_cvref_t<typename FmhaPipeline::QDataType>;
    using KDataType    = ck::remove_cvref_t<typename FmhaPipeline::KDataType>;
    using VDataType    = ck::remove_cvref_t<typename FmhaPipeline::VDataType>;
    using BiasDataType = ck::remove_cvref_t<typename FmhaPipeline::BiasDataType>;
    using ODataType    = ck::remove_cvref_t<typename FmhaPipeline::ODataType>;

    using VLayout = ck::remove_cvref_t<typename FmhaPipeline::VLayout>;

    static constexpr bool kIsGroupMode     = FmhaPipeline::kIsGroupMode;
    static constexpr bool kM0NeedPadding   = FmhaPipeline::kM0NeedPadding;
    static constexpr bool kN0K1NeedPadding = FmhaPipeline::kN0K1NeedPadding;
    static constexpr bool kSupportsBias    = FmhaPipeline::kSupportsBias;

    using C0MatrixMask = ck::tile_program::block::C0MatrixMask_impl<
        ck::remove_cvref_t<typename FmhaPipeline::BlockFmhaMask>>;

    private:
    struct EmptyKargs
    {
    };

    struct CommonKargs
    {
        __host__ constexpr CommonKargs(const void* q_ptr_,
                                       const void* k_ptr_,
                                       const void* v_ptr_,
                                       void* o_ptr_,
                                       ck::index_t seqlen_q_,
                                       ck::index_t seqlen_k_,
                                       ck::index_t hdim_q_,
                                       ck::index_t hdim_v_,
                                       float scale_,
                                       ck::index_t stride_q_,
                                       ck::index_t stride_k_,
                                       ck::index_t stride_v_,
                                       ck::index_t stride_o_,
                                       ck::index_t nhead_stride_q_,
                                       ck::index_t nhead_stride_k_,
                                       ck::index_t nhead_stride_v_,
                                       ck::index_t nhead_stride_o_)
            : q_ptr{reinterpret_cast<const QDataType*>(q_ptr_)},
              k_ptr{reinterpret_cast<const KDataType*>(k_ptr_)},
              v_ptr{reinterpret_cast<const VDataType*>(v_ptr_)},
              o_ptr{reinterpret_cast<ODataType*>(o_ptr_)},
              seqlen_q{seqlen_q_},
              seqlen_k{seqlen_k_},
              hdim_q{hdim_q_},
              hdim_v{hdim_v_},
              scale{scale_},
              stride_q{stride_q_},
              stride_k{stride_k_},
              stride_v{stride_v_},
              stride_o{stride_o_},
              nhead_stride_q{nhead_stride_q_},
              nhead_stride_k{nhead_stride_k_},
              nhead_stride_v{nhead_stride_v_},
              nhead_stride_o{nhead_stride_o_}
        {
        }

        const QDataType* q_ptr;
        const KDataType* k_ptr;
        const VDataType* v_ptr;
        ODataType* o_ptr;

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
    };

    struct CommonBiasKargs
    {
        const BiasDataType* bias_ptr  = nullptr;
        ck::index_t stride_bias       = 0;
        ck::index_t nhead_stride_bias = 0;
    };

    struct BatchModeBiasKargs : CommonBiasKargs
    {
        ck::index_t batch_stride_bias = 0;
    };

    struct BatchModeKargs : CommonKargs,
                            std::conditional_t<kSupportsBias, BatchModeBiasKargs, EmptyKargs>
    {
        __host__ constexpr BatchModeKargs(const void* q_ptr_,
                                          const void* k_ptr_,
                                          const void* v_ptr_,
                                          void* o_ptr_,
                                          ck::index_t seqlen_q_,
                                          ck::index_t seqlen_k_,
                                          ck::index_t hdim_q_,
                                          ck::index_t hdim_v_,
                                          float scale_,
                                          ck::index_t stride_q_,
                                          ck::index_t stride_k_,
                                          ck::index_t stride_v_,
                                          ck::index_t stride_o_,
                                          ck::index_t nhead_stride_q_,
                                          ck::index_t nhead_stride_k_,
                                          ck::index_t nhead_stride_v_,
                                          ck::index_t nhead_stride_o_,
                                          ck::index_t batch_stride_q_,
                                          ck::index_t batch_stride_k_,
                                          ck::index_t batch_stride_v_,
                                          ck::index_t batch_stride_o_)
            : CommonKargs{q_ptr_,
                          k_ptr_,
                          v_ptr_,
                          o_ptr_,
                          seqlen_q_,
                          seqlen_k_,
                          hdim_q_,
                          hdim_v_,
                          scale_,
                          stride_q_,
                          stride_k_,
                          stride_v_,
                          stride_o_,
                          nhead_stride_q_,
                          nhead_stride_k_,
                          nhead_stride_v_,
                          nhead_stride_o_},
              batch_stride_q{batch_stride_q_},
              batch_stride_k{batch_stride_k_},
              batch_stride_v{batch_stride_v_},
              batch_stride_o{batch_stride_o_}
        {
        }

        ck::index_t batch_stride_q;
        ck::index_t batch_stride_k;
        ck::index_t batch_stride_v;
        ck::index_t batch_stride_o;
    };

    struct GroupModeKargs : CommonKargs,
                            std::conditional_t<kSupportsBias, CommonBiasKargs, EmptyKargs>
    {
        __host__ constexpr GroupModeKargs(const void* q_ptr_,
                                          const void* k_ptr_,
                                          const void* v_ptr_,
                                          void* o_ptr_,
                                          const void* seqstart_q_ptr_,
                                          const void* seqstart_k_ptr_,
                                          const void* seqlen_k_ptr_,
                                          ck::index_t hdim_q_,
                                          ck::index_t hdim_v_,
                                          float scale_,
                                          ck::index_t stride_q_,
                                          ck::index_t stride_k_,
                                          ck::index_t stride_v_,
                                          ck::index_t stride_o_,
                                          ck::index_t nhead_stride_q_,
                                          ck::index_t nhead_stride_k_,
                                          ck::index_t nhead_stride_v_,
                                          ck::index_t nhead_stride_o_)
            : CommonKargs{q_ptr_,
                          k_ptr_,
                          v_ptr_,
                          o_ptr_,
                          -1 /* will be updated inside the kernel */,
                          -1 /* will be updated inside the kernel */,
                          hdim_q_,
                          hdim_v_,
                          scale_,
                          stride_q_,
                          stride_k_,
                          stride_v_,
                          stride_o_,
                          nhead_stride_q_,
                          nhead_stride_k_,
                          nhead_stride_v_,
                          nhead_stride_o_},
              seqstart_q_ptr{reinterpret_cast<const int32_t*>(seqstart_q_ptr_)},
              seqstart_k_ptr{reinterpret_cast<const int32_t*>(seqstart_k_ptr_)},
              seqlen_k_ptr{reinterpret_cast<const int32_t*>(seqlen_k_ptr_)}
        {
        }

        const int32_t* seqstart_q_ptr;
        const int32_t* seqstart_k_ptr;
        const int32_t* seqlen_k_ptr;
    };

    public:
    using Kargs = std::conditional_t<kIsGroupMode, GroupModeKargs, BatchModeKargs>;

    template <bool Cond = !kIsGroupMode && !kSupportsBias>
    __host__ static constexpr std::enable_if_t<Cond, Kargs> MakeKargs(const void* q_ptr,
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
                                                                      ck::index_t batch_stride_o)
    {
        return Kargs{q_ptr,          k_ptr,          v_ptr,          o_ptr,          seqlen_q,
                     seqlen_k,       hdim_q,         hdim_v,         scale,          stride_q,
                     stride_k,       stride_v,       stride_o,       nhead_stride_q, nhead_stride_k,
                     nhead_stride_v, nhead_stride_o, batch_stride_q, batch_stride_k, batch_stride_v,
                     batch_stride_o};
    }

    template <bool Cond = !kIsGroupMode && kSupportsBias>
    __host__ static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* q_ptr,
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
              ck::index_t batch_stride_o,
              std::optional<std::tuple<const void*, ck::index_t, ck::index_t, ck::index_t>> bias =
                  std::nullopt)
    {
        Kargs kargs{q_ptr,          k_ptr,          v_ptr,          o_ptr,          seqlen_q,
                    seqlen_k,       hdim_q,         hdim_v,         scale,          stride_q,
                    stride_k,       stride_v,       stride_o,       nhead_stride_q, nhead_stride_k,
                    nhead_stride_v, nhead_stride_o, batch_stride_q, batch_stride_k, batch_stride_v,
                    batch_stride_o};

        if(bias.has_value())
        {
            kargs.bias_ptr          = reinterpret_cast<const BiasDataType*>(std::get<0>(*bias));
            kargs.stride_bias       = std::get<1>(*bias);
            kargs.nhead_stride_bias = std::get<2>(*bias);
            kargs.batch_stride_bias = std::get<3>(*bias);
        }

        return kargs;
    }

    template <bool Cond = kIsGroupMode && !kSupportsBias>
    __host__ static constexpr std::enable_if_t<Cond, Kargs> MakeKargs(const void* q_ptr,
                                                                      const void* k_ptr,
                                                                      const void* v_ptr,
                                                                      void* o_ptr,
                                                                      const void* seqstart_q_ptr,
                                                                      const void* seqstart_k_ptr,
                                                                      const void* seqlen_k_ptr,
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
                                                                      ck::index_t nhead_stride_o)
    {
        return Kargs{q_ptr,
                     k_ptr,
                     v_ptr,
                     o_ptr,
                     seqstart_q_ptr,
                     seqstart_k_ptr,
                     seqlen_k_ptr,
                     hdim_q,
                     hdim_v,
                     scale,
                     stride_q,
                     stride_k,
                     stride_v,
                     stride_o,
                     nhead_stride_q,
                     nhead_stride_k,
                     nhead_stride_v,
                     nhead_stride_o};
    }

    template <bool Cond = kIsGroupMode&& kSupportsBias>
    __host__ static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* q_ptr,
              const void* k_ptr,
              const void* v_ptr,
              void* o_ptr,
              const void* seqstart_q_ptr,
              const void* seqstart_k_ptr,
              const void* seqlen_k_ptr,
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
              std::optional<std::tuple<const void*, ck::index_t, ck::index_t>> bias = std::nullopt)
    {
        Kargs kargs{q_ptr,
                    k_ptr,
                    v_ptr,
                    o_ptr,
                    seqstart_q_ptr,
                    seqstart_k_ptr,
                    seqlen_k_ptr,
                    hdim_q,
                    hdim_v,
                    scale,
                    stride_q,
                    stride_k,
                    stride_v,
                    stride_o,
                    nhead_stride_q,
                    nhead_stride_k,
                    nhead_stride_v,
                    nhead_stride_o};

        if(bias.has_value())
        {
            kargs.bias_ptr          = reinterpret_cast<const BiasDataType*>(std::get<0>(*bias));
            kargs.stride_bias       = std::get<1>(*bias);
            kargs.nhead_stride_bias = std::get<2>(*bias);
        }

        return kargs;
    }

    __host__ static constexpr auto GridSize(ck::index_t batch_size_,
                                            ck::index_t nhead_,
                                            ck::index_t seqlen_q_,
                                            ck::index_t hdim_v_)
    {
        return TilePartitioner::GridSize(batch_size_, nhead_, seqlen_q_, hdim_v_);
    }

    __host__ static constexpr auto BlockSize() { return dim3(kBlockSize); }

    __host__ __device__ static constexpr ck::index_t GetSmemSize()
    {
        return ck::math::max(FmhaPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    __device__ void operator()(Kargs kargs) const
    {
        using namespace ck;
        using namespace ck::tile_program;
        using namespace ck::tile_program::block;

        // allocate LDS
        __shared__ char smem_ptr[GetSmemSize()];

        // divide problem
        const auto [i_tile_m, i_tile_n, i_nhead, i_batch] =
            TilePartitioner{}(kargs.seqlen_q, kargs.hdim_v);

        const index_t i_m0 = __builtin_amdgcn_readfirstlane(i_tile_m * FmhaPipeline::kM0);
        const index_t i_n1 = __builtin_amdgcn_readfirstlane(i_tile_n * FmhaPipeline::kN1);

        index_t batch_offset_q    = 0;
        index_t batch_offset_k    = 0;
        index_t batch_offset_v    = 0;
        index_t batch_offset_bias = 0;
        index_t batch_offset_o    = 0;

        if constexpr(kIsGroupMode)
        {
            // get starting offset for each batch
            const index_t query_start = kargs.seqstart_q_ptr[i_batch];
            const index_t key_start   = kargs.seqstart_k_ptr[i_batch];

            batch_offset_q = query_start * kargs.stride_q;
            batch_offset_k = key_start * kargs.stride_k;
            if constexpr(ck::is_same_v<VLayout, ck::tensor_layout::gemm::RowMajor>)
            {
                batch_offset_v = key_start * kargs.stride_v;
            }
            else
            {
                batch_offset_v = key_start;
            }
            if constexpr(kSupportsBias)
            {
                batch_offset_bias = query_start * kargs.stride_bias + key_start;
            }
            else
            {
                batch_offset_bias = key_start;
            }
            batch_offset_o = query_start * kargs.stride_o;

            // get real # queries & # keys under group mode
            const auto adjusted_seqstart_q_ptr = kargs.seqstart_q_ptr + i_batch;
            kargs.seqlen_q = adjusted_seqstart_q_ptr[1] - adjusted_seqstart_q_ptr[0];

            if(kargs.seqlen_k_ptr != nullptr)
            {
                kargs.seqlen_k = kargs.seqlen_k_ptr[i_batch];
            }
            else
            {
                const auto adjusted_seqstart_k_ptr = kargs.seqstart_k_ptr + i_batch;
                kargs.seqlen_k = adjusted_seqstart_k_ptr[1] - adjusted_seqstart_k_ptr[0];
            }

            if(i_m0 >= kargs.seqlen_q)
                return;
        }
        else
        {
            batch_offset_q = i_batch * kargs.batch_stride_q;
            batch_offset_k = i_batch * kargs.batch_stride_k;
            batch_offset_v = i_batch * kargs.batch_stride_v;
            if constexpr(kSupportsBias)
            {
                batch_offset_bias = i_batch * kargs.batch_stride_bias;
            }
            batch_offset_o = i_batch * kargs.batch_stride_o;
        }

        // for simplicity, batch stride we just modify the pointer
        const QDataType* q_ptr = kargs.q_ptr + i_nhead * kargs.nhead_stride_q + batch_offset_q;
        const KDataType* k_ptr = kargs.k_ptr + i_nhead * kargs.nhead_stride_k + batch_offset_k;
        const VDataType* v_ptr = kargs.v_ptr + i_nhead * kargs.nhead_stride_v + batch_offset_v;
        ODataType* o_ptr       = kargs.o_ptr + i_nhead * kargs.nhead_stride_o + batch_offset_o;

        // Q/K/V DRAM and DRAM window
        const auto q_dram = [&]() {
            const auto q_dram_naive = make_naive_tensor_view<AddressSpaceEnum::Global>(
                q_ptr,
                make_tuple(kargs.seqlen_q, kargs.hdim_q),
                make_tuple(kargs.stride_q, 1),
                Number<32>{},
                Number<1>{});

            return pad_tensor_view(q_dram_naive,
                                   make_tuple(Number<FmhaPipeline::kM0>{}, Number<1>{}),
                                   Sequence<kM0NeedPadding, false>{});
        }();
        const auto k_dram = [&]() {
            const auto k_dram_naive = make_naive_tensor_view<AddressSpaceEnum::Global>(
                k_ptr,
                make_tuple(kargs.seqlen_k, kargs.hdim_q),
                make_tuple(kargs.stride_k, 1),
                Number<32>{},
                Number<1>{});

            return pad_tensor_view(k_dram_naive,
                                   make_tuple(Number<FmhaPipeline::kN0>{}, Number<1>{}),
                                   Sequence<kN0K1NeedPadding, false>{});
        }();
        const auto v_dram = [&]() {
            if constexpr(ck::is_same_v<VLayout, ck::tensor_layout::gemm::RowMajor>)
            {
                const auto v_dram_naive = make_naive_tensor_view<AddressSpaceEnum::Global>(
                    v_ptr,
                    make_tuple(kargs.seqlen_k, kargs.hdim_v),
                    make_tuple(kargs.stride_v, 1),
                    Number<32>{},
                    Number<1>{});

                const auto v_dram_transposed =
                    transform_tensor_view(v_dram_naive,
                                          make_tuple(make_pass_through_transform(kargs.seqlen_k),
                                                     make_pass_through_transform(kargs.hdim_v)),
                                          make_tuple(Sequence<1>{}, Sequence<0>{}),
                                          make_tuple(Sequence<0>{}, Sequence<1>{}));

                /// FIXME: The return value of v_dram_naive.GetTensorDescriptor().GetLength() is
                /// same as
                ///   v_dram_transposed.GetTensorDescriptor().GetLength(). Replace following
                ///   if-clause by pad_tensor_view() call after fixing this issue.
                if constexpr(kN0K1NeedPadding)
                {
                    const index_t pad_length =
                        FmhaPipeline::kK1 *
                            ck::math::integer_divide_ceil(kargs.seqlen_k, FmhaPipeline::kK1) -
                        kargs.seqlen_k;

                    return transform_tensor_view(
                        v_dram_transposed,
                        make_tuple(make_pass_through_transform(kargs.hdim_v),
                                   make_right_pad_transform(kargs.seqlen_k, pad_length)),
                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                        make_tuple(Sequence<0>{}, Sequence<1>{}));
                }
                else
                {
                    return v_dram_transposed;
                }
            }
            else
            {
                const auto v_dram_naive = make_naive_tensor_view<AddressSpaceEnum::Global>(
                    v_ptr,
                    make_tuple(kargs.hdim_v, kargs.seqlen_k),
                    make_tuple(kargs.stride_v, 1),
                    Number<32>{},
                    Number<1>{});

                return pad_tensor_view(v_dram_naive,
                                       make_tuple(Number<1>{}, Number<FmhaPipeline::kK1>{}),
                                       Sequence<false, kN0K1NeedPadding>{});
            }
        }();

        auto q_dram_window = make_tile_window(
            q_dram,
            [&]() {
                if constexpr(FmhaPipeline::kQLoadOnce)
                    return make_tuple(Number<FmhaPipeline::kM0>{},
                                      Number<FmhaPipeline::kK0BlockLength>{});
                else
                    return make_tuple(Number<FmhaPipeline::kM0>{}, Number<FmhaPipeline::kK0>{});
            }(),
            {i_m0, 0});

        auto k_dram_window = make_tile_window(
            k_dram, make_tuple(Number<FmhaPipeline::kN0>{}, Number<FmhaPipeline::kK0>{}), {0, 0});

        auto v_dram_window =
            make_tile_window(v_dram,
                             make_tuple(Number<FmhaPipeline::kN1>{}, Number<FmhaPipeline::kK1>{}),
                             {i_n1, 0});

        const auto run_pipeline_with = [&](auto bias_dram_window) {
            C0MatrixMask casual_mask{kargs.seqlen_q, kargs.seqlen_k};

            return FmhaPipeline{}(q_dram_window,
                                  k_dram_window,
                                  v_dram_window,
                                  bias_dram_window,
                                  casual_mask,
                                  kargs.scale,
                                  ck::math::integer_divide_ceil(kargs.seqlen_k, FmhaPipeline::kN0),
                                  ck::math::integer_divide_ceil(kargs.hdim_q, FmhaPipeline::kK0),
                                  smem_ptr);
        };

        /// FIXME: Before C++20, capturing structured binding variables is not supported. Remove
        /// following copy capture of the 'i_nhead'
        ///        if compiled in C++20
        auto o_acc_tile = [&, i_nhead_ = i_nhead]() {
            constexpr auto bias_dram_window_lengths =
                make_tuple(Number<FmhaPipeline::kM0>{}, Number<FmhaPipeline::kN0>{});

            if constexpr(kSupportsBias)
            {
                if(kargs.bias_ptr != nullptr)
                {
                    const BiasDataType* bias_ptr =
                        kargs.bias_ptr + i_nhead_ * kargs.nhead_stride_bias + batch_offset_bias;

                    const auto bias_dram = [&]() {
                        const auto bias_dram_naive =
                            make_naive_tensor_view<AddressSpaceEnum::Global>(
                                bias_ptr,
                                make_tuple(kargs.seqlen_q, kargs.seqlen_k),
                                make_tuple(kargs.stride_bias, 1),
                                Number<32>{},
                                Number<1>{});

                        return pad_tensor_view(bias_dram_naive,
                                               bias_dram_window_lengths,
                                               Sequence<kM0NeedPadding, kN0K1NeedPadding>{});
                    }();

                    const auto bias_dram_window =
                        make_tile_window(bias_dram, bias_dram_window_lengths, {i_m0, 0});

                    return run_pipeline_with(bias_dram_window);
                }
                else
                {
                    const auto dummy_bias_dram_window =
                        make_null_tile_window(bias_dram_window_lengths);

                    return run_pipeline_with(dummy_bias_dram_window);
                }
            }
            else
            {
                const auto dummy_bias_dram_window = make_null_tile_window(bias_dram_window_lengths);

                return run_pipeline_with(dummy_bias_dram_window);
            }
        }();

        // O DRAM and O DRAM window
        auto o_dram = [&]() {
            const auto o_dram_naive = make_naive_tensor_view<AddressSpaceEnum::Global>(
                o_ptr,
                make_tuple(kargs.seqlen_q, kargs.hdim_v),
                make_tuple(kargs.stride_o, 1),
                Number<32>{},
                Number<1>{});

            return pad_tensor_view(o_dram_naive,
                                   make_tuple(Number<FmhaPipeline::kM0>{}, Number<1>{}),
                                   Sequence<kM0NeedPadding, false>{});
        }();

        auto o_dram_window =
            make_tile_window(o_dram,
                             make_tuple(Number<FmhaPipeline::kM0>{}, Number<FmhaPipeline::kN1>{}),
                             {i_m0, i_n1});

        EpiloguePipeline{}(o_dram_window, o_acc_tile);
    }
};
