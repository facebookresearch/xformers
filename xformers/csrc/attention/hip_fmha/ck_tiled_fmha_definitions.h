/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

//#include <ck/tile_program/block_tile/block_masking_specialization.hpp>

enum struct CausalMaskType
{
    MaskDisabled,
    MaskUpperTriangleFromTopLeft,
    MaskUpperTriangleFromBottomRight
};

/*
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
*/
