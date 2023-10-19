#include <ck.hpp>
#include <stdexcept>

#include "ck_fmha_batched_backward.h"

template struct batched_backward_masktype_attnbias_dispatched<
    ck::bhalf_t,
    1,
    true,
    true>;

template struct batched_backward_masktype_attnbias_dispatched<
    ck::bhalf_t,
    1,
    true,
    false>;

template struct batched_backward_masktype_attnbias_dispatched<
    ck::bhalf_t,
    1,
    false,
    true>;

template struct batched_backward_masktype_attnbias_dispatched<
    ck::bhalf_t,
    1,
    false,
    false>;
