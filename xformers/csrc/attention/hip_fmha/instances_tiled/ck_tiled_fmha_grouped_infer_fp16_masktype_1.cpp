#include <ck/ck.hpp>

#include "ck_tiled_fmha_grouped_infer.h"

template void run_grouped_infer_masktype_attnbias_dispatched<ck::half_t, 1>(
    GroupedForwardParams& param,
    hipStream_t stream);
