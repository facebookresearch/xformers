/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ck/ck.hpp>
#include "ck_fmha_grouped_backward.h"

template void run_grouped_backward_masktype_attnbias_dispatched<ck::half_t, 2, false, true>(
    GroupedBackwardParams& param, hipStream_t stream);
