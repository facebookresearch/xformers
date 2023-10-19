#include <ck/ck.hpp>
#include <stdexcept>

#include "ck_fmha_grouped_backward.h"

template struct grouped_backward_masktype_attnbias_dispatched<
    ck::bhalf_t,
    0,
    true,
    true>;

template struct grouped_backward_masktype_attnbias_dispatched<
    ck::bhalf_t,
    0,
    true,
    false>;

template struct grouped_backward_masktype_attnbias_dispatched<
    ck::bhalf_t,
    0,
    false,
    true>;

template struct grouped_backward_masktype_attnbias_dispatched<
    ck::bhalf_t,
    0,
    false,
    false>;
