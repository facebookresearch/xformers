#!/usr/bin/env bash

declare -a Tasks=("text" "image" "listops" "retrieval" "pathfinder32-curv_contour_length_14" )
declare -a Attentions=("scaled_dot_product" "nystrom" "favor" "fourier_mix" "linformer" "lambda")

checkpoint_path=$1

for attention in ${Attentions[@]}; do
    for task in ${Tasks[@]}; do
        echo "**** " $attention $task
        tail $checkpoint_path/$attention/"$task"__"$attention"__0_output.log
        echo ""
    done
done
