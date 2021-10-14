# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


#  CREDITS: Adapted from https://github.com/mlpen/Nystromformer

import logging
import pickle
import sys

sys.path.append("./datasets/long-range-arena")
sys.path.append("./datasets/long-range-arena/lra_benchmarks/image/")
import input_pipeline  # type: ignore # noqa

(
    train_ds,
    eval_ds,
    test_ds,
    num_classes,
    vocab_size,
    input_shape,
) = input_pipeline.get_cifar10_datasets(  # type: ignore
    n_devices=1, batch_size=1, normalize=False
)

logging.getLogger().setLevel(logging.INFO)

mapping = {"train": train_ds, "dev": eval_ds, "test": test_ds}
max_iter = {"train": 45000, "dev": 5000, "test": 10000}
for component in mapping:
    ds_list = []
    for idx, inst in enumerate(iter(mapping[component])):
        ds_list.append(
            {
                "input_ids_0": inst["inputs"].numpy()[0].reshape(-1),
                "label": inst["targets"].numpy()[0],
            }
        )
        if idx % 1000 == 0:
            logging.info(f"{idx}")

        # The dataset from LRA repeats
        if idx > max_iter[component]:
            break
    logging.info(f"{component} dataset processed")

    with open(f"./datasets/image.{component}.pickle", "wb") as f:
        pickle.dump(ds_list, f)
