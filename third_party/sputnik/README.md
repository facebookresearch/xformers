# Sputnik

Sputnik is a library of sparse linear algebra kernels and utilities for deep learning.

## Build

Sputnik uses the CMake build system. Sputnik depends on the CUDA toolkit (v10.1+) and supports SM70+. The only additional dependency for the library is [google/glog](https://github.com/google/glog). To build the library, enter the project directory and run the following commands:

`mkdir build && cd build`

`cmake .. -DCMAKE_BUILD_TYPE=Release`

`make -j12`

The test and benchmark suites additionally depend on [abseil/abseil-cpp](https://github.com/abseil/abseil-cpp), [google/googltest](https://github.com/google/googletest), and [google/benchmark](https://github.com/google/benchmark). These dependencies are includes as submodules in [third_party](https://github.com/google-research/sputnik/tree/os-build/third_party). To build the test suite and/or benchmark suite, set `-DBUILD_TEST=ON` and/or `-DBUILD_BENCHMARK=ON` in your `cmake` command.

`cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TEST=ON -DBUILD_BENCHMARK=ON -DCUDA_ARCHS="70;75"`

## Docker

Sputnik provides a [Dockerfile](https://github.com/google-research/sputnik/blob/os-build/Dockerfile) that builds the proper environment with all dependencies. Note that [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) must be installed to run on GPU. To build the image, run the following command:

`docker build . -t sputnik-dev`

To launch the container with the sputnik source mounted under `/mount` (assuming you're working out of $HOME), run the following:

`sudo docker run --runtime=nvidia -v ~/:/mount/ -it sputnik-dev:latest`

## Citation

If you make use of this library, please cite:

```
@inproceedings{sgk_sc2020,
  author    = {Trevor Gale and Matei Zaharia and Cliff Young and Erich Elsen},
  title     = {Sparse {GPU} Kernels for Deep Learning},
  booktitle = {Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, {SC} 2020},
  year      = {2020},
}
```

The sparse models and dataset of sparse matrices from deep neural networks from the above paper can be found [here](https://github.com/google-research/google-research/tree/master/sgk).

## Disclaimer
This is not an official Google product.
