#!/usr/bin/env sh

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


red_echo() {
    echo -e "\e[1;31m"
    echo -e $1
    echo -e "\e[0m"
}

maybe_success() {
    if [ ! $? -eq 0 ]; then
        red_echo "[ ] Error\n"
    else
        red_echo "[x] Done\n"
    fi
}

mkdir -p datasets
cd datasets
touch '__init__.py'

if [ ! -d "long-range-arena" ]; then
    red_echo "Cloning the original LRA repository\n"
    git clone git@github.com:google-research/long-range-arena.git
fi

if [ ! -d "lra_release" ]; then
    red_echo "Installing the LRA release files\n"
    wget https://storage.googleapis.com/long-range-arena/lra_release.gz

    red_echo "Decompressing the LRA release archive\n"
    tar -xvf lra_release.gz
    rm lra_release.gz
fi
cd ..

red_echo "\n*** All required data fetched, now installing datasets ***\n"

if [ ! -f "datasets/pathfinder32-curv_baseline.dev.pickle" ]; then
    red_echo "Installing the pathfinder dataset\n"
    python3 setup/pathfinder.py
    maybe_success
else
    red_echo "[x] Pathfinder dataset seems to be installed already\n"
fi

if [ ! -f "datasets/listops.dev.pickle" ]; then
    red_echo "Installing the listops dataset\n"
    python3 setup/listops.py
    maybe_success
else
    red_echo "[x] Listops dataset seems to be installed already\n"
fi

if [ ! -f "datasets/retrieval.dev.pickle" ]; then
    red_echo "Installing the retrieval dataset\n"
    python3 setup/retrieval.py
    maybe_success
else
    red_echo "[x] Retrieval dataset seems to be installed already\n"
fi

if [ ! -f "datasets/text.dev.pickle" ]; then
    red_echo "Installing the text dataset\n"
    python3 setup/text.py
    maybe_success
else
    red_echo "[x] Text dataset seems to be installed already\n"
fi

red_echo "Installing the cifar10 dataset\n"
python3 setup/cifar10.py
