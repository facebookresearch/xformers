# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


#  CREDITS: Adapted from https://github.com/mlpen/Nystromformer

import logging
import pickle
import sys

import numpy as np

sys.path.append("./datasets/long-range-arena")
sys.path.append("./datasets/long-range-arena/lra_benchmarks/matching/")
import input_pipeline  # type: ignore # noqa

logging.getLogger().setLevel(logging.INFO)

train_ds, eval_ds, test_ds, encoder = input_pipeline.get_matching_datasets(  # type: ignore
    n_devices=1,
    task_name=None,
    data_dir="./datasets/lra_release/lra_release/tsv_data/",
    batch_size=1,
    fixed_vocab=None,
    max_length=4096,
    tokenizer="char",
    vocab_file_path=None,
)

mapping = {"train": train_ds, "dev": eval_ds, "test": test_ds}
for component in mapping:
    ds_list = []
    for idx, inst in enumerate(iter(mapping[component])):
        ds_list.append(
            {
                "input_ids_0": np.concatenate([inst["inputs1"].numpy()[0]]),
                "input_ids_1": np.concatenate([inst["inputs2"].numpy()[0]]),
                "label": inst["targets"].numpy()[0],
            }
        )
        if idx % 100 == 0:
            logging.info(f"{idx}\t\t")
    with open(f"./datasets/retrieval.{component}.pickle", "wb") as f:
        pickle.dump(ds_list, f)
