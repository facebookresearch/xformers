# noqa: C801
# Copyright (c) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from pathlib import Path

FMHA_INSTANCE_HEADER = """
/*
  Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * The file is automatically generated, don't modify!
 */
"""

FMHA_INFER_INSTANCE_TEMPLATE = """
#include <ck_tile/core/numeric/{dtype_file}.hpp>
#include \"ck_tiled_fmha_{mode}_infer.h\"

template void run_{mode}_infer_causalmask_bias_dropout_dispatch<
    {dtype},
    {has_causalmask},
    {has_bias},
    {has_dropout},
    {max_k}>({cap_mode}ForwardParams& param, hipStream_t stream);
"""

FMHA_INFER_INSTANCE_FNAME = (
    "fmha_{mode}_infer_{dtype_str}_{has_or_no_causalmask_str}_"
    "{has_or_no_bias_str}_{has_or_no_dropout_str}_{max_k_str}.cpp"
)

FMHA_FORWARD_INSTANCE_TEMPLATE = """
#include <ck_tile/core/numeric/{dtype_file}.hpp>
#include \"ck_tiled_fmha_{mode}_forward.h\"

template void run_{mode}_forward_causalmask_bias_dropout_dispatch<
    {dtype},
    {has_causalmask},
    {has_bias},
    {has_dropout},
    {max_k}>({cap_mode}ForwardParams& param, hipStream_t stream);
"""

FMHA_FORWARD_INSTANCE_FNAME = (
    "fmha_{mode}_forward_{dtype_str}_{has_or_no_causalmask_str}_"
    "{has_or_no_bias_str}_{has_or_no_dropout_str}_{max_k_str}.cpp"
)

FMHA_BACKWARD_INSTANCE_TEMPLATE = """
#include <ck_tile/core/numeric/{dtype_file}.hpp>
#include \"ck_tiled_fmha_{mode}_backward.h\"

template void run_{mode}_backward_causalmask_bias_dropout_dispatch<
    {dtype},
    {has_causalmask},
    {has_bias},
    {has_bias_grad},
    {has_dropout},
    {max_k}>({cap_mode}BackwardParams& param, hipStream_t stream);
"""

FMHA_BACKWARD_INSTANCE_FNAME = (
    "fmha_{mode}_backward_{dtype_str}_{has_or_no_causalmask_str}_"
    "{has_or_no_bias_str}_{has_or_no_biasgrad_str}_{has_or_no_dropout_str}_{max_k_str}.cpp"
)

BOOL_MAP = {True: "true", False: "false"}

BOOL_MAP_CAUSALMASK = {
    True: "has_causalmask",
    False: "no_causalmask",
}

BOOL_MAP_BIAS = {
    True: "has_bias",
    False: "no_bias",
}

BOOL_MAP_BIASGRAD = {
    True: "has_biasgrad",
    False: "no_biasgrad",
}

BOOL_MAP_DROPOUT = {
    True: "has_dropout",
    False: "no_dropout",
}

INT_MAP_MAX_K = {
    32: "maxk_32",
    64: "maxk_64",
    128: "maxk_128",
    256: "maxk_256",
}

TYPE_CTYPE_MAP = {
    "fp16": "ck_tile::fp16_t",
    "bf16": "ck_tile::bf16_t",
}

TYPE_FNAME_MAP = {
    "fp16": "half",
    "bf16": "bfloat16",
}

MODE_NAME_MAP = {
    "batched": "Batched",
    "grouped": "Grouped",
}


def create_infer_instances(instance_dir: Path) -> None:
    for mode in ["batched", "grouped"]:
        for dtype in ["fp16", "bf16"]:
            for has_causalmask in [True, False]:
                for has_bias in [True, False]:
                    for has_dropout in [True, False]:
                        for max_k in [32, 64, 128, 256]:
                            fname = FMHA_INFER_INSTANCE_FNAME.format(
                                mode=mode,
                                dtype_str=dtype,
                                has_or_no_causalmask_str=BOOL_MAP_CAUSALMASK[
                                    has_causalmask
                                ],
                                has_or_no_bias_str=BOOL_MAP_BIAS[has_bias],
                                has_or_no_dropout_str=BOOL_MAP_DROPOUT[has_dropout],
                                max_k_str=INT_MAP_MAX_K[max_k],
                            )
                            infer_instance = FMHA_INFER_INSTANCE_TEMPLATE.format(
                                mode=mode,
                                dtype_file=TYPE_FNAME_MAP[dtype],
                                dtype=TYPE_CTYPE_MAP[dtype],
                                has_causalmask=BOOL_MAP[has_causalmask],
                                has_bias=BOOL_MAP[has_bias],
                                has_dropout=BOOL_MAP[has_dropout],
                                max_k=max_k,
                                cap_mode=MODE_NAME_MAP[mode],
                            )
                            (instance_dir / fname).write_text(
                                FMHA_INSTANCE_HEADER + infer_instance
                            )


def create_forward_instances(instance_dir: Path) -> None:
    for mode in ["batched", "grouped"]:
        for dtype in ["fp16", "bf16"]:
            for has_causalmask in [True, False]:
                for has_bias in [True, False]:
                    for has_dropout in [True, False]:
                        for max_k in [32, 64, 128, 256]:
                            fname = FMHA_FORWARD_INSTANCE_FNAME.format(
                                mode=mode,
                                dtype_str=dtype,
                                has_or_no_causalmask_str=BOOL_MAP_CAUSALMASK[
                                    has_causalmask
                                ],
                                has_or_no_bias_str=BOOL_MAP_BIAS[has_bias],
                                has_or_no_dropout_str=BOOL_MAP_DROPOUT[has_dropout],
                                max_k_str=INT_MAP_MAX_K[max_k],
                            )
                            infer_instance = FMHA_FORWARD_INSTANCE_TEMPLATE.format(
                                mode=mode,
                                dtype_file=TYPE_FNAME_MAP[dtype],
                                dtype=TYPE_CTYPE_MAP[dtype],
                                has_causalmask=BOOL_MAP[has_causalmask],
                                has_bias=BOOL_MAP[has_bias],
                                has_dropout=BOOL_MAP[has_dropout],
                                max_k=max_k,
                                cap_mode=MODE_NAME_MAP[mode],
                            )
                            (instance_dir / fname).write_text(
                                FMHA_INSTANCE_HEADER + infer_instance
                            )


def create_backward_instances(instance_dir: Path) -> None:
    for mode in ["batched", "grouped"]:
        for dtype in ["fp16", "bf16"]:
            for has_causalmask in [True, False]:
                for has_bias, has_bias_grad in [
                    [True, False],
                    [True, True],
                    [False, False],
                ]:
                    for has_dropout in [True, False]:
                        for max_k in [32, 64, 128]:
                            fname = FMHA_BACKWARD_INSTANCE_FNAME.format(
                                mode=mode,
                                dtype_str=dtype,
                                has_or_no_causalmask_str=BOOL_MAP_CAUSALMASK[
                                    has_causalmask
                                ],
                                has_or_no_bias_str=BOOL_MAP_BIAS[has_bias],
                                has_or_no_biasgrad_str=BOOL_MAP_BIASGRAD[has_bias_grad],
                                has_or_no_dropout_str=BOOL_MAP_DROPOUT[has_dropout],
                                max_k_str=INT_MAP_MAX_K[max_k],
                            )
                            infer_instance = FMHA_BACKWARD_INSTANCE_TEMPLATE.format(
                                mode=mode,
                                dtype_file=TYPE_FNAME_MAP[dtype],
                                dtype=TYPE_CTYPE_MAP[dtype],
                                has_causalmask=BOOL_MAP[has_causalmask],
                                has_bias=BOOL_MAP[has_bias],
                                has_bias_grad=BOOL_MAP[has_bias_grad],
                                has_dropout=BOOL_MAP[has_dropout],
                                max_k=max_k,
                                cap_mode=MODE_NAME_MAP[mode],
                            )
                            (instance_dir / fname).write_text(
                                FMHA_INSTANCE_HEADER + infer_instance
                            )


if __name__ == "__main__":
    this_dir = os.path.dirname(__file__)
    output_dir = Path(this_dir) / "instances"
    output_dir.mkdir(parents=True, exist_ok=True)
    create_infer_instances(output_dir)
    create_forward_instances(output_dir)
    create_backward_instances(output_dir)
