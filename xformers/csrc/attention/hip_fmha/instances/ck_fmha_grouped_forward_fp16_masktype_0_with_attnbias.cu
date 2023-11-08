#include <ck/ck.hpp>
#include "ck_fmha_grouped_forward.h"

template void run_grouped_forward_masktype_attnbias_dispatched<
    ck::half_t,
    0,
    true>(GroupedForwardParams& param, hipStream_t stream);
