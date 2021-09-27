# xFormers

![xFormers Logo](./docs/assets/logo.png)

Flexible Transformers, defined by interoperable and optimized building blocks.

## Key concepts

- **Field agnostic**. This repo is not focused on NLP, speech or vision, by design. The focus is on making sure that building transformers is a shared building block in between. This makes the models, ideas and optimizations easier to share.

- **Composable**. There are basically two obvious takes to the Transformer ecosystem:
  - expose all the architectures proposed in different papers more or less individually
  - break all the achitectures into a _block zoo_, which allows you to recompose said reference models, but also study ablations or architecture search. This repo aims at supporting this second way.

- **Extensible**. This is meant to support a research effort, and not to substitute to modelling and field specific know how, or to constraint into the use of specific blocks. xFormers aims at being _easy to extend locally_, so that one can focus on a specific improvement, and easily compare it against the state of the art.

- **Optimized**. Reusing building blocks across teams and domains means that engineering efforts can be more valued, so that investing in speeding up some key building blocks (without breaking compatibility thanks to the settled interface) is possible. Some results [here](BENCHMARKS.md).

- **Tested**. Each and every of the variant in the repo is _tested, alone and composed with the other relevant blocks_. This happens automatically anytime somebody proposes a new variant through a PR, via a registry mechanism.

- **Crowd Sourced**. This is probably the single most important part. All of the above should make it possible for people interested to contribute: contributing on a small block is easier than on a full model, unit tests and common interfaces should help, the ability to extend the library locally and test the relevance prior to a PR should also help. PRs are really welcome, [see for details](CONTRIBUTING.md).

## Using xFormers

If in doubt, please check out the [HOWTO](HOWTO.md). Only some general considerations are laid out in the README.

### Installing the repository

It is recommended to use a dedicated virtual environment, as often with python, through `python-virtualenv` or `conda` for instance.
`pip install -e .` is all you need to install in dev mode (you can change the repo code and the installation will follow suit), if you just want to install from source and not change it afterwards `pip install .` is what you'll need.

### Repo map

```bash
├── components                  # Parts zoo, any of which can be used directly
│   ├── attention               # all the supported attentions
│   ├── feedforward             # ..
│   ├─- positional_embedding    # ..
│   ├── activations.py          # ..
│   └── multi_head_dispatch.py  # (optional) multihead wrap
├── factory
│   ├── block_factory.py        # (optional) helper to programatically generate layers
│   └── model_factory.py        # (optional) helper to programatically generate models
├── models
...                             # Full models, ready to be used
```

### Using Attentions

You can find some more details in the [HOWTO](HOWTO.md), but in short:

- How does multi-head attention works ?
  - The multi-head wrapper handles the dimension changes, so maybe that is what you would like to use. In a nutshell, we fold the head dimension into the batch, since both relate to parallel, independent computations. Feel free to skip the multi-head wrapper if that's easier for you
- Do all attentions expose the same settings ?
  - As much as possible, yes. Some options do not relate well to some attention mechanisms though, maybe because there's no real attention map to begin with, or that the dimensions are arbitrarily changed
- Can I just cherry pick an attention from the repo and run with it ?
  - Yes, go for it !
- How can I easily test out different attentions ?
  - Either you import the ones you're interested in directly in your code base, their API should be very close and you would own everything. The dimension expectations are explained in the HOWTO
  - Alternatively, a `build_attention` helper is provided, which takes a dict as an input. By sweeping over several settings (attention names for instance), you can try out several options in a pretty compact code footprint

### Sparse attention

Below you will find a set of notebooks that will show you how you can use xFormers in your project

- [Creating complex sparsity patterns with xFormers](docs/source/2d_attention_patterns.ipynb)
- [Changing ViT to use sparse attention, benchmarking the effects](docs/source/vision_transformers.ipynb)

### Adding new models

Models live in `xformers/models`. As a general rule, one should try to write them using the blocks present in `xformers/components` (or upstream PyTorch), so that ulterior improvements are propagated to each implementation.


## Bibliograpy

Some references or papers used in the repo

- [Attention is all you need, Vaswani et al., 2017](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- [Linformer, self-attention with linear complexity, Wang et al., 2020](https://arxiv.org/pdf/2006.04768.pdf)
- [BigBird, Transformer for longer sequences, Zaheer et al., 2020](https://papers.nips.cc/paper/2020/file/c8512d142a2d849725f31a9a7a361ab9-Paper.pdf)
- [Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention, Xiong et al., 2021](https://arxiv.org/abs/2102.03902)
- [Rethinking Attention with Performers, Choromanski et al., 2020](https://arxiv.org/abs/2009.14794v1)
- [Reformer: The Efficient Transformer, Kitaev et al., 2020](https://arxiv.org/abs/2001.04451)
- [Longformer: The Long-Document Transformer, Beltagy et al., 2020](https://arxiv.org/pdf/2004.05150.pdf)
- [Long Range Arena: a benchmark for efficient Transformers, Tay et al., 2020](https://arxiv.org/abs/2011.04006)
- [FNet: Mixing tokens with Fourier transform, Lee-Thorp et al., 2021](https://arxiv.org/pdf/2105.03824v1.pdf)
- [The reversible residual network: Backpropagation without storing activations. Gomez,et al. 2017](https://arxiv.org/abs/1707.04585)
