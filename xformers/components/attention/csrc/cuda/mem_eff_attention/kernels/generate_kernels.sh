#!/bin/bash
set -ex
rm -f *.cu
IFS=","
for kernel in "BACKWARD"; do
    kernel_lower=`echo "\$kernel" | awk '{print tolower($0)}'`
    for aligned in "false" "true"; do
        [[ $aligned = "true" ]] && aligned_suffix="_aligned" || aligned_suffix=""
        for dtype_name in "f32" "f16"; do
            case "$dtype_name" in
                "f32") dtype="float" ;;
                "f16") dtype="cutlass::half_t" ;;
            esac
            FNAME="${kernel_lower}_${dtype_name}${aligned_suffix}.cu"
            echo $FNAME
            cat <<EOF > $FNAME
// This file is auto-generated. See "generate_kernels.sh"
#include "../kernel_backward.h"
EOF
            for sm in 50 70 75 80; do
                echo "INSTANTIATE_ATTENTION_KERNEL_${kernel}_SM${sm}($dtype, $aligned);" >> $FNAME
            done;
        done;
    done
done;
