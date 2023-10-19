#include <ck/ck.hpp>
#include <stdexcept>

#include "ck_fmha_grouped_backward.h"

template struct grouped_backward_masktype_attnbias_dispatched<
    ck::half_t,
    2,
    true,
    true>;

template struct grouped_backward_masktype_attnbias_dispatched<
    ck::half_t,
    2,
    true,
    false>;

template struct grouped_backward_masktype_attnbias_dispatched<
    ck::half_t,
    2,
    false,
    true>;

template struct grouped_backward_masktype_attnbias_dispatched<
    ck::half_t,
    2,
    false,
    false>;
