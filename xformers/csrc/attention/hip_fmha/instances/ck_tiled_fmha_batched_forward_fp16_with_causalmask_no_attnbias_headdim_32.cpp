/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ck/ck.hpp>

#include "ck_tiled_fmha_batched_forward.h"

template void run_batched_forward_causalmask_attnbias_dispatched<
    ck::half_t,
    true,
    false,
    32>(BatchedForwardParams& param, hipStream_t stream);
