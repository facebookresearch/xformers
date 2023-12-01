#pragma once

#include <ck/tile_program/block_tile/block_masking_specialization.hpp>

enum struct CausalMaskType
{
    MaskDisabled,
    MaskUpperTriangleFromTopLeft,
    MaskUpperTriangleFromBottomRight
};

template <CausalMaskType type>
struct CausalMaskPredicate;

template <>
struct CausalMaskPredicate<CausalMaskType::MaskDisabled>
{
    using predicate = ck::tile_program::block::MaskDisabledPredicate;
};

template <>
struct CausalMaskPredicate<CausalMaskType::MaskUpperTriangleFromTopLeft>
{
    using predicate = ck::tile_program::block::MaskUpperTriangleFromTopLeftPredicate;
};

template <>
struct CausalMaskPredicate<CausalMaskType::MaskUpperTriangleFromBottomRight>
{
    using predicate = ck::tile_program::block::MaskUpperTriangleFromBottomRightPredicate;
};
