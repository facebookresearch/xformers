#include <ck/ck.hpp>
#include "ck_fmha_grouped_backward.h"

template void run_grouped_backward_masktype_attnbias_dispatched<
    ck::bhalf_t,
    0,
    true,
    true>(GroupedBackwardParams& param, hipStream_t stream);
