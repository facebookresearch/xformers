
/*
  Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * The file is automatically generated, don't modify!
 * See the generator script `/home/mpodkory/xformers/xformers/csrc/attention/hip_fmha/generate_instances.py`
 */

#include <ck_tile/core/numeric/bfloat16.hpp>
#include "ck_tiled_fmha_grouped_backward.h"

template void run_grouped_backward_mask_bias_dropout_dispatch<
    ck_tile::bf16_t,
    false,
    false,
    false,
    false,
    64>(GroupedBackwardParams& param, hipStream_t stream);
