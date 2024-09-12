<img src="./docs/assets/logo.png" width=800>

![Install with conda](https://anaconda.org/xformers/xformers/badges/installer/conda.svg)
![Downloads](https://anaconda.org/xformers/xformers/badges/downloads.svg)
![License](https://anaconda.org/xformers/xformers/badges/license.svg)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/facebookresearch/xformers/blob/main/docs/source/xformers_mingpt.ipynb)
<br/><!--
![PyPI](https://img.shields.io/pypi/v/xformers)
![PyPI - License](https://img.shields.io/pypi/l/xformers)
[![Documentation Status](https://github.com/facebookresearch/xformers/actions/workflows/gh-pages.yml/badge.svg)](https://github.com/facebookresearch/xformers/actions/workflows/gh-pages.yml/badge.svg)
-->
[![CircleCI](https://circleci.com/gh/facebookresearch/xformers.svg?style=shield)](https://app.circleci.com/pipelines/github/facebookresearch/xformers/)
[![Codecov](https://codecov.io/gh/facebookresearch/xformers/branch/main/graph/badge.svg?token=PKGKDR4JQM)](https://codecov.io/gh/facebookresearch/xformers)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
<br/>
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
<!--
[![Downloads](https://pepy.tech/badge/xformers)](https://pepy.tech/project/xformers)
-->
--------------------------------------------------------------------------------

## xFormers - Toolbox to Accelerate Research on Transformers

xFormers is:
- **Customizable building blocks**: Independent/customizable building blocks that can be used without boilerplate code. The components are domain-agnostic and xFormers is used by researchers in vision, NLP and more.
- **Research first**: xFormers contains bleeding-edge components, that are not yet available in mainstream libraries like PyTorch.
- **Built with efficiency in mind**: Because speed of iteration matters, components are as fast and memory-efficient as possible. xFormers contains its own CUDA kernels, but dispatches to other libraries when relevant.

## Installing xFormers

* **(RECOMMENDED, linux) Install latest stable with conda**: Requires [PyTorch 2.4.1 installed with conda](https://pytorch.org/get-started/locally/)

```bash
# (python 3.10/3.11 only)
conda install xformers -c xformers
```

* **(RECOMMENDED, linux & win) Install latest stable with pip**: Requires [PyTorch 2.4.1](https://pytorch.org/get-started/locally/)

```bash
# [linux only] cuda 11.8 version
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118
# [linux only] cuda 12.1 version
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121
# [linux & win] cuda 12.4 version
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu124
# [linux only] (EXPERIMENTAL) rocm 6.1 version
pip3 install -U xformers --index-url https://download.pytorch.org/whl/rocm6.1
```

* **Development binaries**:

```bash
# Use either conda or pip, same requirements as for the stable version above
conda install xformers -c xformers/label/dev
pip install --pre -U xformers
```

* **Install from source**: If you want to use with another version of PyTorch for instance (including nightly-releases)

```bash
# (Optional) Makes the build much faster
pip install ninja
# Set TORCH_CUDA_ARCH_LIST if running and building on different GPU types
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
# (this can take dozens of minutes)
```


## Benchmarks

**Memory-efficient MHA**
![Benchmarks for ViTS](./docs/plots/mha/mha_vit.png)
*Setup: A100 on f16, measured total time for a forward+backward pass*

Note that this is exact attention, not an approximation, just by calling [`xformers.ops.memory_efficient_attention`](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.memory_efficient_attention)

**More benchmarks**

xFormers provides many components, and more benchmarks are available in [BENCHMARKS.md](BENCHMARKS.md).

### (Optional) Testing the installation

This command will provide information on an xFormers installation, and what kernels are built/available:

```python
python -m xformers.info
```

## Using xFormers

### Key Features

1. Optimized building blocks, beyond PyTorch primitives
   1. Memory-efficient exact attention - up to 10x faster
   2. sparse attention
   3. block-sparse attention
   4. fused softmax
   5. fused linear layer
   6. fused layer norm
   7. fused dropout(activation(x+bias))
   8. fused SwiGLU

### Install troubleshooting


* NVCC and the current CUDA runtime match. Depending on your setup, you may be able to change the CUDA runtime with `module unload cuda; module load cuda/xx.x`, possibly also `nvcc`
* the version of GCC that you're using matches the current NVCC capabilities
* the `TORCH_CUDA_ARCH_LIST` env variable is set to the architectures that you want to support. A suggested setup (slow to build but comprehensive) is `export TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.6"`
* If the build from source OOMs, it's possible to reduce the parallelism of ninja with `MAX_JOBS` (eg `MAX_JOBS=2`)
* If you encounter [`UnsatisfiableError`](https://github.com/facebookresearch/xformers/issues/390#issuecomment-1315020700) when installing with conda, make sure you have PyTorch installed in your conda environment, and that your setup (PyTorch version, cuda version, python version, OS) match [an existing binary for xFormers](https://anaconda.org/xformers/xformers/files)


### License

xFormers has a BSD-style license, as found in the [LICENSE](LICENSE) file.

## Citing xFormers

If you use xFormers in your publication, please cite it by using the following BibTeX entry.

``` bibtex
@Misc{xFormers2022,
  author =       {Benjamin Lefaudeux and Francisco Massa and Diana Liskovich and Wenhan Xiong and Vittorio Caggiano and Sean Naren and Min Xu and Jieru Hu and Marta Tintore and Susan Zhang and Patrick Labatut and Daniel Haziza and Luca Wehrstedt and Jeremy Reizenstein and Grigory Sizov},
  title =        {xFormers: A modular and hackable Transformer modelling library},
  howpublished = {\url{https://github.com/facebookresearch/xformers}},
  year =         {2022}
}
```

## Credits

The following repositories are used in xFormers, either in close to original form or as an inspiration:

* [Sputnik](https://github.com/google-research/sputnik)
* [GE-SpMM](https://github.com/hgyhungry/ge-spmm)
* [Triton](https://github.com/openai/triton)
* [LucidRain Reformer](https://github.com/lucidrains/reformer-pytorch)
* [RevTorch](https://github.com/RobinBruegger/RevTorch)
* [Nystromformer](https://github.com/mlpen/Nystromformer)
* [FairScale](https://github.com/facebookresearch/fairscale/)
* [Pytorch Image Models](https://github.com/rwightman/pytorch-image-models)
* [CUTLASS](https://github.com/nvidia/cutlass)
* [Flash-Attention](https://github.com/HazyResearch/flash-attention)
