#include <ck/ck.hpp>
#include <stdexcept>

#include "ck_fmha_batched_backward.h"

template void run_batched_backward_masktype_attnbias_dispatched<
    ck::half_t,
    1,
    true,
    true>(BatchedBackwardParams& param, hipStream_t stream);
