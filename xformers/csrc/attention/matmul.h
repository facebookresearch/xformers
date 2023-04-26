/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ATen/ATen.h>
#include "macros.h"

XFORMERS_API at::Tensor matmul_with_mask(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& mask);
