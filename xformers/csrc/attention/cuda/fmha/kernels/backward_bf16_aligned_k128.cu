// This file is auto-generated. See "generate_kernels.sh"
#include "backward.h"
INSTANTIATE_ATTENTION_KERNEL_BACKWARD_SM50(
    cutlass::bfloat16_t,
    true,
    false,
    128);
INSTANTIATE_ATTENTION_KERNEL_BACKWARD_SM70(
    cutlass::bfloat16_t,
    true,
    false,
    128);
INSTANTIATE_ATTENTION_KERNEL_BACKWARD_SM75(
    cutlass::bfloat16_t,
    true,
    false,
    128);
INSTANTIATE_ATTENTION_KERNEL_BACKWARD_SM80(
    cutlass::bfloat16_t,
    true,
    false,
    128);
