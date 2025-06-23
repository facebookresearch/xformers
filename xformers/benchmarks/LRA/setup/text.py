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
sys.path.append("./datasets/long-range-arena/lra_benchmarks/text_classification/")
import input_pipeline  # type: ignore  # noqa

logging.getLogger().setLevel(logging.INFO)

train_ds, eval_ds, test_ds, encoder = input_pipeline.get_tc_datasets(  # type: ignore
    n_devices=1,
    task_name="imdb_reviews",
    data_dir=None,
    batch_size=1,
    fixed_vocab=None,
    max_length=4000,
)

mapping = {"train": train_ds, "dev": eval_ds, "test": test_ds}
for component in mapping:
    ds_list = []
    for idx, inst in enumerate(iter(mapping[component])):
        ds_list.append(
            {
                "input_ids_0": np.concatenate(
                    [inst["inputs"].numpy()[0], np.zeros(96, dtype=np.int32)]
                ),
                "label": inst["targets"].numpy()[0],
            }
        )
        if idx % 100 == 0:
            logging.info(f"{idx}\t\t")
    with open(f"./datasets/text.{component}.pickle", "wb") as f:
        pickle.dump(ds_list, f)
