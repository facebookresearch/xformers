#pragma once

#include <ATen/ATen.h>
#include "macros.h"

XFORMERS_API at::Tensor matmul_with_mask(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& mask);
