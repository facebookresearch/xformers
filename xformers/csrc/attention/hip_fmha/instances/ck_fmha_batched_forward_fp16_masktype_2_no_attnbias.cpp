#include <ck/ck.hpp>
#include "ck_fmha_batched_forward.h"

template void run_batched_forward_masktype_attnbias_dispatched<
    ck::half_t,
    2,
    false>(BatchedForwardParams& param, hipStream_t stream);
