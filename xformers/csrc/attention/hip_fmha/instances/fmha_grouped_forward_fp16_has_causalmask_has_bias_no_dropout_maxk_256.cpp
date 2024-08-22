
/*
  Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * The file is automatically generated, don't modify!
 */

#include <ck_tile/core/numeric/half.hpp>
#include "ck_tiled_fmha_grouped_forward.h"

template void run_grouped_forward_causalmask_bias_dropout_dispatch<
    ck_tile::fp16_t,
    true,
    true,
    false,
    256>(GroupedForwardParams& param, hipStream_t stream);
