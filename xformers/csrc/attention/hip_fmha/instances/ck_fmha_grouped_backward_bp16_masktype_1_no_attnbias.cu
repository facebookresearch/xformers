#include <ck/ck.hpp>
#include "ck_fmha_grouped_backward.h"

template void run_grouped_backward_masktype_attnbias_dispatched<
    ck::bhalf_t,
    1,
    false,
    true>(GroupedBackwardParams& param, hipStream_t stream);

template void run_grouped_backward_masktype_attnbias_dispatched<
    ck::bhalf_t,
    1,
    false,
    false>(GroupedBackwardParams& param, hipStream_t stream);
