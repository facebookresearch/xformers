#!/usr/bin/env bash

declare -a Tasks=("text" "image" "listops" "retrieval" "pathfinder32-curv_contour_length_14")
declare -a Attentions=("scaled_dot_product" "nystrom" "favor" "fourier_mix" "linformer" "lambda")

config_path=$1
checkpoint_path=$2

for attention in ${Attentions[@]}; do
    for task in ${Tasks[@]}; do
        python3 run_with_submitit.py --attention $attention  --task $task --config $config_path --checkpoint_dir $checkpoint_path/$attention
    done
done
