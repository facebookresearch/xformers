#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tile_program/tile/store_tile.hpp"
#include "ck/tile_program/tile/tile_elementwise.hpp"

template <typename BlockFmhaShape_>
struct FmhaFwdTilePartitioner
{
    using BlockFmhaShape = ck::remove_cvref_t<BlockFmhaShape_>;

    static constexpr ck::index_t kM0 = BlockFmhaShape::kM0;
    static constexpr ck::index_t kN0 = BlockFmhaShape::kN0;
    static constexpr ck::index_t kK0 = BlockFmhaShape::kK0;
    static constexpr ck::index_t kN1 = BlockFmhaShape::kN1;
    static constexpr ck::index_t kK1 = BlockFmhaShape::kK1;

    __host__ static constexpr auto GridSize(ck::index_t batch_size_,
                                            ck::index_t nhead_,
                                            ck::index_t seqlen_q_,
                                            ck::index_t hdim_v_)
    {
        // TODO: this may need tuning
        return dim3(ck::math::integer_divide_ceil(seqlen_q_, kM0) *
                        ck::math::integer_divide_ceil(hdim_v_, kN1),
                    batch_size_,
                    nhead_);
    }

    __device__ auto operator()(ck::index_t /*seqlen_q*/, ck::index_t hdim_v)
    {
        using namespace ck;

        // const index_t num_tile_m0 = seqlen_q / kM0;
        const index_t num_tile_n1 = hdim_v / kN1;

        const index_t i_block = blockIdx.x;
        const index_t i_batch = blockIdx.y;
        const index_t i_nhead = blockIdx.z;

        const auto f = [](index_t dividend, index_t divisor) {
            index_t quotient = dividend / divisor;
            index_t modulus  = dividend - quotient * divisor;
            return ck::make_tuple(quotient, modulus);
        };

        const auto [i_tile_m, i_tile_n] = f(i_block, num_tile_n1);

        return ck::make_tuple(i_tile_m, i_tile_n, i_nhead, i_batch);
    }
};
