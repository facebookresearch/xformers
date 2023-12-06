/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ck/ck.hpp>
#include "ck_fmha_op_helper.h"

// list the template parameters that is commonly used
struct GemmOpConstantsCommon
{
    static constexpr ck::index_t NumDimG = 2;
    static constexpr ck::index_t NumDimM = 1;
    static constexpr ck::index_t NumDimN = 1;
    static constexpr ck::index_t NumDimK = 1;
    static constexpr ck::index_t NumDimO = 1;

    static constexpr auto TensorSpecA = ck::tensor_operation::device::TensorSpecialization::Default;
    static constexpr auto TensorSpecB0 =
        ck::tensor_operation::device::TensorSpecialization::Default;
    static constexpr auto TensorSpecB1 =
        ck::tensor_operation::device::TensorSpecialization::Default;
    static constexpr auto TensorSpecC = ck::tensor_operation::device::TensorSpecialization::Default;
};
