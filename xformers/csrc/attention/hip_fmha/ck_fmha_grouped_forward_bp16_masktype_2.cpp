#include <ck/ck.hpp>
#include <stdexcept>

#include "ck_fmha_grouped_forward.h"

template struct grouped_forward_masktype_attnbias_dispatched<
    ck::bhalf_t,
    2,
    true>;

template struct grouped_forward_masktype_attnbias_dispatched<
    ck::bhalf_t,
    2,
    false>;
