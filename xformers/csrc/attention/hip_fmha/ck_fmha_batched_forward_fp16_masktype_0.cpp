#include <ck/ck.hpp>
#include <stdexcept>

#include "ck_fmha_batched_forward.h"

template struct batched_forward_masktype_attnbias_dispatched<
    ck::half_t,
    0,
    true>;

template struct batched_forward_masktype_attnbias_dispatched<
    ck::half_t,
    0,
    false>;
