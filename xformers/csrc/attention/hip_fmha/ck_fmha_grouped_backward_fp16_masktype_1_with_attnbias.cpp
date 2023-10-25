#include <ck/ck.hpp>
#include "ck_fmha_grouped_backward.h"

template struct grouped_backward_masktype_attnbias_dispatched<
    ck::half_t,
    1,
    true,
    true>;

template struct grouped_backward_masktype_attnbias_dispatched<
    ck::half_t,
    1,
    true,
    false>;
