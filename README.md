<img src="./docs/assets/logo.png" width=800>

![PyPI](https://img.shields.io/pypi/v/xformers)
![PyPI - License](https://img.shields.io/pypi/l/xformers)
[![Documentation Status](https://github.com/facebookresearch/xformers/actions/workflows/gh-pages.yml/badge.svg)](https://github.com/facebookresearch/xformers/actions/workflows/gh-pages.yml/badge.svg)
[![CircleCI](https://circleci.com/gh/facebookresearch/xformers.svg?style=shield)](https://app.circleci.com/pipelines/github/facebookresearch/xformers/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![codecov](https://codecov.io/gh/facebookresearch/xformers/branch/main/graph/badge.svg?token=PKGKDR4JQM)](https://codecov.io/gh/facebookresearch/xformers)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/facebookresearch/xformers/blob/main/docs/source/xformers_mingpt.ipynb)
[![Downloads](https://pepy.tech/badge/xformers)](https://pepy.tech/project/xformers)
--------------------------------------------------------------------------------

## Description

xFormers is a modular and field agnostic library to flexibly generate transformer architectures by interoperable and optimized building blocks.

## Getting started

The full [documentation](https://facebookresearch.github.io/xformers/) contains instructions for getting started, deep dives and tutorials about the various APIs.
If in doubt, please check out the [HOWTO](HOWTO.md). Only some general considerations are laid out in the README.

### Installation

To install xFormers, it is recommended to use a dedicated virtual environment, as often with python, through `python-virtualenv` or `conda` for instance.
There are two ways you can install it:

#### Directly from the pip package

  You can also fetch the latest release from PyPi. This will not contain the wheels for the sparse attention kernels, for which you will need to build from source.

  ```bash
  conda create --name xformer_env
  conda activate xformer_env
  pip install xformers
  ```

#### Build from source (dev mode)

  These commands will fetch the latest version of the code, create a dedicated `conda` environment, activate it then install xFormers from source. If you want to build the sparse attention CUDA kernels, please make sure that the next point is covered prior to running these instructions.

  ```bash
  git clone git@github.com:facebookresearch/xformers.git
  conda create --name xformer_env python=3.8
  conda activate xformer_env
  cd xformers
  pip install -r requirements.txt
  pip install -e .
  # or, for OSX
  MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ pip install -e .
  ```

#### Sparse attention kernels

Installing the CUDA-based sparse attention kernels may require extra care, as this mobilizes the CUDA toolchain. As a reminder, these kernels are built when you run `pip install -e .` and the CUDA buildchain is available (NVCC compiler). Re-building can for instance be done via `python3 setup.py clean && python3 setup.py develop`, so similarly wipe the `build` folder and redo a pip install -e.

Some advices related to building these CUDA-specific components, tentatively adressing common pitfalls. Please make sure that:

* NVCC and the current CUDA runtime match. Depending on your setup, you may be able to change the CUDA runtime with `module unload cuda module load cuda/xx.x`, possibly also `nvcc`
* the version of GCC that you're using matches the current NVCC capabilities
* the `TORCH_CUDA_ARCH_LIST` env variable is set to the architures that you want to support. A suggested setup (slow to build but comprehensive) is `export TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.2;8.0;8.6"`

#### Triton

Some parts of xFormers use [Triton](http://www.triton-lang.org), and will only expose themselves if Triton is installed, and a compatible GPU is present (nVidia GPU with tensor cores). If Triton was not installed as part of the testing procedure, you can install it directly by running `pip install triton`. You can optionally test that the installation is successful by running one of the Triton-related benchmarks, for instance `python3 xformers/benchmarks/benchmark_triton_softmax.py`

Triton will cache the compiled kernels to `/tmp/triton` by default. If this becomes an issue, this path can be specified through the `TRITON_CACHE_DIR` environment variable.

### Testing the installation

This will run a benchmark of the attention mechanisms exposed by xFormers, and generate a runtime and memory plot.
If this concludes without errors, the installation is successful. This step is optional, and you will need some extra dependencies for it to
be able to go through : `pip install -r requirements-benchmark.txt`.

Once this is done, you can run this particular benchmark as follows:

```python
python3 xformers/benchmarks/benchmark_encoder.py --activations relu  --plot -emb 256 -bs 32 -heads 16
```

## Using xFormers

### Transformers key concepts

Let's start from a classical overview of the Transformer architecture (illustration from Lin et al,, "A Survey of Transformers")
<p align="center">
  <img src="./docs/assets/Transformer_arch_Lin_et_al.png" width=600>
</p>

You'll find the key repository boundaries in this illustration: a Transformer is generally made of a collection of attention mechanisms, embeddings to encode some positional information, feed-forward blocks and a residual path (typically referred to as pre- or post- layer norm). These boundaries do not work for all models, but we found in practice that given some accomodations it could capture most of the state of the art.

Models are thus not implemented in monolithic files, which are typically complicated to handle and modify. Most of the concepts present in the above illustration correspond to an abstraction level, and when variants are present for a given sub-block it should always be possible to select any of them. You can focus on a given encapsulation level and modify it as needed.


### Repo map

```bash
├── components                  # Parts zoo, any of which can be used directly
│   ├── attention
│   │    └ ...                  # all the supported attentions
│   ├── feedforward             #
│   │    └ ...                  # all the supported feedforwards
│   ├── positional_embedding    #
│   │    └ ...                  # all the supported positional embeddings
│   ├── activations.py          #
│   └── multi_head_dispatch.py  # (optional) multihead wrap
│
├── factory                     # Build model programatically
│   ├── block_factory.py        # (optional) helper to programatically generate layers
│   └── model_factory.py        # (optional) helper to programatically generate models
│
├── benchmarks
│     └ ...                     # A lot of benchmarks that you can use to test some parts
└── triton
      └ ...                     # (optional) all the triton parts, requires triton + CUDA gpu
```

<details><summary> Attention mechanisms</summary><p>

- [Scaled dot product](xformers/components/attention/scaled_dot_product.py)
  - *[Attention is all you need, Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)*
- [Sparse](xformers/components/attention/scaled_dot_product.py)
  - whenever a sparse enough mask is passed
- [BlockSparse](xformers/components/attention/blocksparse.py)
  - *courtesy of [Triton](www.triton-lang.org)*
- [Linformer](xformers/components/attention/linformer.py)
  - *[Linformer, self-attention with linear complexity, Wang et al., 2020](https://arxiv.org/abs/2006.04768)*
- [Nystrom](xformers/components/attention/nystrom.py)
  - *[Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention, Xiong et al., 2021](https://arxiv.org/abs/2102.03902)*
- [Local](xformers/components/attention/local.py).
 Notably used in (and many others)
  - *[Longformer: The Long-Document Transformer, Beltagy et al., 2020](https://arxiv.org/abs/2004.05150)*
  - *[BigBird, Transformer for longer sequences, Zaheer et al., 2020](https://arxiv.org/abs/2007.14062)*

- [Favor/Performer](xformers/components/attention/favor.py)
  - *[Rethinking Attention with Performers, Choromanski et al., 2020](https://arxiv.org/abs/2009.14794v1)*
- [Orthoformer](xformers/components/attention/ortho.py)
  - *[Keeping Your Eye on the Ball: Trajectory Attention in Video Transformers,
Patrick et al., 2021](https://arxiv.org/abs/2106.05392)*
- [Random](xformers/components/attention/random.py)
  - See BigBird, Longformers,..
- [Global](xformers/components/attention/random.py)
  - See BigBird, Longformers,..
- [FourierMix](xformers/components/attention/fourier_mix.py)
  - *[FNet: Mixing Tokens with Fourier Transforms, Lee-Thorp et al.](https://arxiv.org/abs/2105.03824v1)*
- [CompositionalAttention](xformers/components/attention/compositional.py)
  - *[Compositional Attention: Disentangling search and retrieval, S. Mittal et al.](https://arxiv.org/pdf/2110.09419v1.pdf)*

- ... add a new one [see Contribution.md](CONTRIBUTING.md)

</p></details>

<details><summary>Feed forward mechanisms </summary><p>

- [MLP](xformers/components/feedforward/mlp.py)
- [Fused](xformers/components/feedforward/fused_mlp.py)
- [Mixture of Experts](xformers/components/feedforward/mixture_of_experts.py)

</p></details>

<details><summary>Positional embedding </summary><p>

- [Sine](xformers/components/positional_embedding/sine.py)
- [Vocabulary](xformers/components/positional_embedding/vocab.py)
- [Rotary](xformers/components/positional_embedding/rotary.py)

</p></details>

<details><summary>Residual paths </summary><p>

- [Pre](https://arxiv.org/pdf/2002.04745v1.pdf)
- [Post](https://arxiv.org/pdf/2002.04745v1.pdf)
- [DeepNorm](https://arxiv.org/pdf/2203.00555v1.pdf)

</p></details>

### Key Features

1. Many attention mechanisms, interchangeables
2. Optimized building blocks, beyond PyTorch primitives
   1. sparse attention
   2. block-sparse attention
   3. fused softmax
   4. fused linear layer
   5. fused layer norm
   6. fused dropout(activation(x+bias))
3. Benchmarking and testing tools
   1. [micro benchnmarks](BENCHMARKS.md)
   2. transformer block benchmark
   3. [LRA](xformers/benchmarks/LRA/README.md), with SLURM suppot
4. Programatic and sweep friendly layer and model construction
5. Hackable
   1. Not using monolithic CUDA kernels, composable building blocks
   2. Using [Triton](https://triton-lang.org/) for some optimized parts, explicit, pythonic and user-accessible
   3. Native support for SquaredReLU (on top of ReLU, LeakyReLU, GeLU, ..), extensible activations

### FAQ ?

We've tried to collect a relatively exhaustive list of explanations in the [HOWTO](HOWTO.md)

### License

xFormers has a BSD-style license, as found in the [LICENSE](LICENSE) file.

## Citing xFormers

If you use xFormers in your publication, please cite it by using the following BibTeX entry.

``` bibtex
@Misc{xFormers2021,
  author =       {Benjamin Lefaudeux, Francisco Massa, Diana Liskovich, Wenhan Xiong, Vittorio Caggiano, Sean Naren, Min Xu, Jieru Hu, Marta Tintore, Susan Zhang},
  title =        {xFormers: A modular and hackable Transformer modelling library},
  howpublished = {\url{https://github.com/facebookresearch/xformers}},
  year =         {2021}
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
