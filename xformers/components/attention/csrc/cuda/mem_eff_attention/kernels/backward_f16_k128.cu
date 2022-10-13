// This file is auto-generated. See "generate_kernels.sh"
#ifndef XFORMERS_MEM_EFF_ATTENTION_DISABLE_BACKWARD
#include "../kernel_backward.h"
INSTANTIATE_ATTENTION_KERNEL_BACKWARD_SM50(cutlass::half_t, false, 128);
INSTANTIATE_ATTENTION_KERNEL_BACKWARD_SM70(cutlass::half_t, false, 128);
INSTANTIATE_ATTENTION_KERNEL_BACKWARD_SM75(cutlass::half_t, false, 128);
INSTANTIATE_ATTENTION_KERNEL_BACKWARD_SM80(cutlass::half_t, false, 128);
#endif
