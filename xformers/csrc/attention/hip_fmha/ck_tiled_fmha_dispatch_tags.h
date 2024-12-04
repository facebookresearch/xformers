/*
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "ck_tile/core/numeric/integral_constant.hpp"

template <bool v>
struct has_mask_t : ck_tile::bool_constant<v> {};

template <bool v>
struct has_bias_t : ck_tile::bool_constant<v> {};

template <bool v>
struct has_dropout_t : ck_tile::bool_constant<v> {};

template <ck_tile::index_t v>
struct max_head_dimension_t : ck_tile::integral_constant<ck_tile::index_t, v> {
};

template <ck_tile::index_t v>
struct max_query_seqlen_t : ck_tile::integral_constant<ck_tile::index_t, v> {};
