#include <ck/ck.hpp>
#include "ck_fmha_grouped_backward.h"

template struct grouped_backward_masktype_attnbias_dispatched<
    ck::bhalf_t,
    1,
    false,
    true>;

template struct grouped_backward_masktype_attnbias_dispatched<
    ck::bhalf_t,
    1,
    false,
    false>;
