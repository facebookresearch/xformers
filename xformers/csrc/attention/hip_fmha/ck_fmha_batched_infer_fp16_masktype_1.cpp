#include <ck/ck.hpp>
#include <stdexcept>

#include "ck_fmha_batched_infer.h"

template struct batched_infer_masktype_attnbias_dispatched<ck::half_t, 1, true>;

template struct batched_infer_masktype_attnbias_dispatched<
    ck::half_t,
    1,
    false>;
