# noqa: C801
# Copyright (c) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from pathlib import Path
from typing import List

FMHA_COPYRIGHT_HEADER = """
/*
  Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * The file is automatically generated, don't modify!
 * See the generator script
 * `{file}`
 */
""".format(
    file=os.path.relpath(os.path.realpath(__file__), start=Path(__file__).parents[4])
)

FMHA_INFER_INSTANCE_TEMPLATE_INC = """
#include <ck_tile/core/numeric/{dtype_file}.hpp>
#include \"ck_tiled_fmha_{mode}_infer.h\"
"""

FMHA_INFER_INSTANCE_TEMPLATE = """
{extern}template void run_{mode}_infer_mask_bias_dropout_dispatch<
    {dtype},
    {has_mask},
    {has_bias},
    {has_dropout},
    {max_k}>({cap_mode}ForwardParams& param, hipStream_t stream);
"""

FMHA_INFER_INSTANCE_FNAME = (
    "fmha_{mode}_infer_{dtype_str}_{has_or_no_mask_str}_"
    "{has_or_no_bias_str}_{has_or_no_dropout_str}_{max_k_str}.cpp"
)

FMHA_FORWARD_INSTANCE_TEMPLATE_INC = """
#include <ck_tile/core/numeric/{dtype_file}.hpp>
#include \"ck_tiled_fmha_{mode}_forward.h\"
"""

FMHA_FORWARD_INSTANCE_TEMPLATE = """
{extern}template void run_{mode}_forward_mask_bias_dropout_dispatch<
    {dtype},
    {has_mask},
    {has_bias},
    {has_dropout},
    {max_k}>({cap_mode}ForwardParams& param, hipStream_t stream);
"""

FMHA_FORWARD_INSTANCE_FNAME = (
    "fmha_{mode}_forward_{dtype_str}_{has_or_no_mask_str}_"
    "{has_or_no_bias_str}_{has_or_no_dropout_str}_{max_k_str}.cpp"
)

FMHA_BACKWARD_INSTANCE_TEMPLATE_INC = """
#include <ck_tile/core/numeric/{dtype_file}.hpp>
#include \"ck_tiled_fmha_{mode}_backward.h\"
"""

FMHA_BACKWARD_INSTANCE_TEMPLATE = """
{extern}template void run_{mode}_backward_mask_bias_dropout_dispatch<
    {dtype},
    {has_mask},
    {has_bias},
    {has_bias_grad},
    {has_dropout},
    {max_k}>({cap_mode}BackwardParams& param, hipStream_t stream);
"""

FMHA_BACKWARD_INSTANCE_FNAME = (
    "fmha_{mode}_backward_{dtype_str}_{has_or_no_mask_str}_"
    "{has_or_no_bias_str}_{has_or_no_biasgrad_str}_{has_or_no_dropout_str}_{max_k_str}.cpp"
)

FMHA_INSTANCE_REF_FNAME = "fmha_{mode}_{function}_{dtype}_instances_ref.h"

BOOL_MAP = {True: "true", False: "false"}

BOOL_MAP_MASK = {
    True: "has_mask",
    False: "no_mask",
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

INT_MAP_MAX_K = {hd: f"maxk_{hd}" for hd in [32, 64, 96, 128, 256, 512]}

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


def create_infer_instances(instance_dir: Path, headdims: List) -> None:
    for mode in ["batched", "grouped"]:
        for dtype in ["fp16", "bf16"]:
            for has_mask in [True, False]:
                for has_bias in [True, False]:
                    for has_dropout in [True, False]:
                        for max_k in headdims:
                            fname = FMHA_INFER_INSTANCE_FNAME.format(
                                mode=mode,
                                dtype_str=dtype,
                                has_or_no_mask_str=BOOL_MAP_MASK[has_mask],
                                has_or_no_bias_str=BOOL_MAP_BIAS[has_bias],
                                has_or_no_dropout_str=BOOL_MAP_DROPOUT[has_dropout],
                                max_k_str=INT_MAP_MAX_K[max_k],
                            )
                            infer_instance_inc = (
                                FMHA_INFER_INSTANCE_TEMPLATE_INC.format(
                                    mode=mode,
                                    dtype_file=TYPE_FNAME_MAP[dtype],
                                )
                            )
                            infer_instance = FMHA_INFER_INSTANCE_TEMPLATE.format(
                                extern="",
                                mode=mode,
                                dtype=TYPE_CTYPE_MAP[dtype],
                                has_mask=BOOL_MAP[has_mask],
                                has_bias=BOOL_MAP[has_bias],
                                has_dropout=BOOL_MAP[has_dropout],
                                max_k=max_k,
                                cap_mode=MODE_NAME_MAP[mode],
                            )
                            (instance_dir / fname).write_text(
                                FMHA_COPYRIGHT_HEADER
                                + infer_instance_inc
                                + infer_instance
                            )


def create_infer_instances_ref(instance_dir: Path, headdims: List) -> None:
    for mode in ["batched", "grouped"]:
        for dtype in ["fp16", "bf16"]:
            ref_fname = FMHA_INSTANCE_REF_FNAME.format(
                mode=mode,
                function="infer",
                dtype=dtype,
            )
            ref_fname_path = instance_dir / ref_fname
            infer_instance_inc = FMHA_INFER_INSTANCE_TEMPLATE_INC.format(
                mode=mode,
                dtype_file=TYPE_FNAME_MAP[dtype],
            )
            with open(ref_fname_path, "a") as file:
                file.write(FMHA_COPYRIGHT_HEADER)
                file.write(infer_instance_inc)
                for max_k in headdims:
                    for has_bias in [True, False]:
                        for has_dropout in [True, False]:
                            for has_mask in [True, False]:
                                infer_instance = FMHA_INFER_INSTANCE_TEMPLATE.format(
                                    extern="extern ",
                                    mode=mode,
                                    dtype=TYPE_CTYPE_MAP[dtype],
                                    has_mask=BOOL_MAP[has_mask],
                                    has_bias=BOOL_MAP[has_bias],
                                    has_dropout=BOOL_MAP[has_dropout],
                                    max_k=max_k,
                                    cap_mode=MODE_NAME_MAP[mode],
                                )
                                file.write(infer_instance)


def create_forward_instances(instance_dir: Path, headdims: List) -> None:
    for mode in ["batched", "grouped"]:
        for dtype in ["fp16", "bf16"]:
            for has_mask in [True, False]:
                for has_bias in [True, False]:
                    for has_dropout in [True, False]:
                        for max_k in headdims:
                            fname = FMHA_FORWARD_INSTANCE_FNAME.format(
                                mode=mode,
                                dtype_str=dtype,
                                has_or_no_mask_str=BOOL_MAP_MASK[has_mask],
                                has_or_no_bias_str=BOOL_MAP_BIAS[has_bias],
                                has_or_no_dropout_str=BOOL_MAP_DROPOUT[has_dropout],
                                max_k_str=INT_MAP_MAX_K[max_k],
                            )
                            forward_instance_inc = (
                                FMHA_FORWARD_INSTANCE_TEMPLATE_INC.format(
                                    mode=mode,
                                    dtype_file=TYPE_FNAME_MAP[dtype],
                                )
                            )
                            forward_instance = FMHA_FORWARD_INSTANCE_TEMPLATE.format(
                                extern="",
                                mode=mode,
                                dtype=TYPE_CTYPE_MAP[dtype],
                                has_mask=BOOL_MAP[has_mask],
                                has_bias=BOOL_MAP[has_bias],
                                has_dropout=BOOL_MAP[has_dropout],
                                max_k=max_k,
                                cap_mode=MODE_NAME_MAP[mode],
                            )
                            (instance_dir / fname).write_text(
                                FMHA_COPYRIGHT_HEADER
                                + forward_instance_inc
                                + forward_instance
                            )


def create_forward_instances_ref(instance_dir: Path, headdims: List) -> None:
    for mode in ["batched", "grouped"]:
        for dtype in ["fp16", "bf16"]:
            ref_fname = FMHA_INSTANCE_REF_FNAME.format(
                mode=mode,
                function="forward",
                dtype=dtype,
            )
            ref_fname_path = instance_dir / ref_fname
            forward_instance_inc = FMHA_FORWARD_INSTANCE_TEMPLATE_INC.format(
                mode=mode,
                dtype_file=TYPE_FNAME_MAP[dtype],
            )
            with open(ref_fname_path, "a") as file:
                file.write(FMHA_COPYRIGHT_HEADER)
                file.write(forward_instance_inc)
                for max_k in headdims:
                    for has_bias in [True, False]:
                        for has_dropout in [True, False]:
                            for has_mask in [True, False]:
                                forward_instance = (
                                    FMHA_FORWARD_INSTANCE_TEMPLATE.format(
                                        extern="extern ",
                                        mode=mode,
                                        dtype=TYPE_CTYPE_MAP[dtype],
                                        has_mask=BOOL_MAP[has_mask],
                                        has_bias=BOOL_MAP[has_bias],
                                        has_dropout=BOOL_MAP[has_dropout],
                                        max_k=max_k,
                                        cap_mode=MODE_NAME_MAP[mode],
                                    )
                                )
                                file.write(forward_instance)


def create_backward_instances(instance_dir: Path, headdims: List) -> None:
    for mode in ["batched", "grouped"]:
        for dtype in ["fp16", "bf16"]:
            for has_mask in [True, False]:
                for has_bias, has_bias_grad in [
                    [True, False],
                    [True, True],
                    [False, False],
                ]:
                    for has_dropout in [True, False]:
                        for max_k in headdims:
                            fname = FMHA_BACKWARD_INSTANCE_FNAME.format(
                                mode=mode,
                                dtype_str=dtype,
                                has_or_no_mask_str=BOOL_MAP_MASK[has_mask],
                                has_or_no_bias_str=BOOL_MAP_BIAS[has_bias],
                                has_or_no_biasgrad_str=BOOL_MAP_BIASGRAD[has_bias_grad],
                                has_or_no_dropout_str=BOOL_MAP_DROPOUT[has_dropout],
                                max_k_str=INT_MAP_MAX_K[max_k],
                            )
                            backward_instance_inc = (
                                FMHA_BACKWARD_INSTANCE_TEMPLATE_INC.format(
                                    mode=mode,
                                    dtype_file=TYPE_FNAME_MAP[dtype],
                                )
                            )
                            backward_instance = FMHA_BACKWARD_INSTANCE_TEMPLATE.format(
                                extern="",
                                mode=mode,
                                dtype=TYPE_CTYPE_MAP[dtype],
                                has_mask=BOOL_MAP[has_mask],
                                has_bias=BOOL_MAP[has_bias],
                                has_bias_grad=BOOL_MAP[has_bias_grad],
                                has_dropout=BOOL_MAP[has_dropout],
                                max_k=max_k,
                                cap_mode=MODE_NAME_MAP[mode],
                            )
                            (instance_dir / fname).write_text(
                                FMHA_COPYRIGHT_HEADER
                                + backward_instance_inc
                                + backward_instance
                            )


def create_backward_instances_ref(instance_dir: Path, headdims: List) -> None:
    for mode in ["batched", "grouped"]:
        for dtype in ["fp16", "bf16"]:
            ref_fname = FMHA_INSTANCE_REF_FNAME.format(
                mode=mode,
                function="backward",
                dtype=dtype,
            )
            ref_fname_path = instance_dir / ref_fname
            backward_instance_inc = FMHA_BACKWARD_INSTANCE_TEMPLATE_INC.format(
                mode=mode,
                dtype_file=TYPE_FNAME_MAP[dtype],
            )
            with open(ref_fname_path, "a") as file:
                file.write(FMHA_COPYRIGHT_HEADER)
                file.write(backward_instance_inc)
                for max_k in headdims:
                    for has_bias, has_bias_grad in [
                        [True, False],
                        [True, True],
                        [False, False],
                    ]:
                        for has_dropout in [True, False]:
                            for has_mask in [True, False]:
                                backward_instance = (
                                    FMHA_BACKWARD_INSTANCE_TEMPLATE.format(
                                        extern="extern ",
                                        mode=mode,
                                        dtype=TYPE_CTYPE_MAP[dtype],
                                        has_mask=BOOL_MAP[has_mask],
                                        has_bias=BOOL_MAP[has_bias],
                                        has_bias_grad=BOOL_MAP[has_bias_grad],
                                        has_dropout=BOOL_MAP[has_dropout],
                                        max_k=max_k,
                                        cap_mode=MODE_NAME_MAP[mode],
                                    )
                                )
                                file.write(backward_instance)


if __name__ == "__main__":
    headdims_fwd = [32, 64, 96, 128, 256, 512]
    headdims_bwd = [32, 64, 96, 128, 256]

    this_dir = os.path.dirname(__file__)
    output_dir = Path(this_dir) / "instances"
    output_dir.mkdir(parents=True, exist_ok=True)

    # remove existing files in the directory
    files = os.listdir(output_dir)
    for ff in files:
        file_path = os.path.join(output_dir, ff)
        os.remove(file_path)

    create_infer_instances(output_dir, headdims_fwd)
    create_infer_instances_ref(output_dir, headdims_fwd)
    create_forward_instances(output_dir, headdims_fwd)
    create_forward_instances_ref(output_dir, headdims_fwd)
    create_backward_instances(output_dir, headdims_bwd)
    create_backward_instances_ref(output_dir, headdims_bwd)
