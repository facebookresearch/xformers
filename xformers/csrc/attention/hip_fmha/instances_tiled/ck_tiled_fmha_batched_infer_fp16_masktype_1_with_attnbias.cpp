#include <ck/ck.hpp>

#include "ck_tiled_fmha_batched_infer.h"

template void
run_batched_infer_masktype_attnbias_dispatched<ck::half_t, 1, true>(BatchedForwardParams& param,
                                                                    hipStream_t stream);
