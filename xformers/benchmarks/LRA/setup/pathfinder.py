# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


#  CREDITS: Adapted from https://github.com/mlpen/Nystromformer

import logging
import os
import pickle
import random

import numpy as np
import tensorflow as tf

logging.getLogger().setLevel(logging.INFO)

root_dir = "./datasets/lra_release/lra_release/"
subdir = "pathfinder32"
for diff_level in ["curv_baseline", "curv_contour_length_9", "curv_contour_length_14"]:
    data_dir = os.path.join(root_dir, subdir, diff_level)
    metadata_list = [
        os.path.join(data_dir, "metadata", file)
        for file in os.listdir(os.path.join(data_dir, "metadata"))
        if file.endswith(".npy")
    ]
    ds_list = []
    for idx, metadata_file in enumerate(metadata_list):
        logging.info(idx, len(metadata_list), metadata_file, "\t\t")
        for inst_meta in (
            tf.io.read_file(metadata_file).numpy().decode("utf-8").split("\n")[:-1]
        ):
            metadata = inst_meta.split(" ")
            img_path = os.path.join(data_dir, metadata[0], metadata[1])
            img_bin = tf.io.read_file(img_path)
            if len(img_bin.numpy()) == 0:
                logging.warning("detected empty image")
                continue
            img = tf.image.decode_png(img_bin)
            seq = img.numpy().reshape(-1).astype(np.int32)
            label = int(metadata[3])
            ds_list.append({"input_ids_0": seq, "label": label})

    random.shuffle(ds_list)

    bp80 = int(len(ds_list) * 0.8)
    bp90 = int(len(ds_list) * 0.9)
    train = ds_list[:bp80]
    dev = ds_list[bp80:bp90]
    test = ds_list[bp90:]

    with open(f"./datasets/{subdir}-{diff_level}.train.pickle", "wb") as f:
        pickle.dump(train, f)
    with open(f"./datasets/{subdir}-{diff_level}.dev.pickle", "wb") as f:
        pickle.dump(dev, f)
    with open(f"./datasets/{subdir}-{diff_level}.test.pickle", "wb") as f:
        pickle.dump(test, f)
