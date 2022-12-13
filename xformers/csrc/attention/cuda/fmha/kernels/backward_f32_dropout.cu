// This file is auto-generated. See "generate_kernels.sh"
#ifndef XFORMERS_MEM_EFF_ATTENTION_DISABLE_BACKWARD
#include "../kernel_backward.h"
INSTANTIATE_ATTENTION_KERNEL_BACKWARD_SM50(float, false, true);
INSTANTIATE_ATTENTION_KERNEL_BACKWARD_SM70(float, false, true);
INSTANTIATE_ATTENTION_KERNEL_BACKWARD_SM75(float, false, true);
INSTANTIATE_ATTENTION_KERNEL_BACKWARD_SM80(float, false, true);
#endif
