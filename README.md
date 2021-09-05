# xFormers

![xFormers Logo](./docs/assets/logo.png)

Flexible Transformers, defined by interoperable and optimized building blocks.

## Key concepts

- **Field agnostic**. This repo is not focused on NLP, speech or vision, by design. The focus is on making sure that building transformers is a shared building block in between. This makes the models, ideas and optimizations easier to share.

- **Composable**. There are basically two obvious takes to the Transformer ecosystem:
  - expose all the architectures proposed in different papers more or less individually
  - break all the achitectures into a _block zoo_, which allows you to recompose said reference models, but also study ablations or architecture search. This repo aims at supporting this second way.

- **Extensible**. This is meant to support a research effort, and not to substitute to modelling and field specific know how, or to constraint into the use of specific blocks. xFormers aims at being _easy to extend locally_, so that one can focus on a specific improvement, and easily compare it against the state of the art.

- **Optimized**. Reusing building blocks across teams and domains means that engineering efforts can be more valued, so that investing in speeding up some key building blocks (without breaking compatibility thanks to the settled interface) is possible.

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

## Some benchmark tools

### Benchmark a full encoder block

Sweeping over different attention settings to log max memory use and runtime can for instance be done by invoking
`python3 xformers/benchmarks/benchmark_encoder.py`. Specifying a subset to test is done through command line arguments, for instance `python3 xformers/benchmarks/benchmark_encoder.py --causal True --attentions random --activations gelu -fp16 True`.

Please note that:

- These numbers are dependent of hyperparameters (dimensions chosen for Linformer, sparsity of the pattern), they are mostly an illustration
- The sparse attention patterns tested here are just presets, as explained in the linked notebook generating any new sparse attention pattern should be relatively easy, while keeping the benefits of optimized computations.

Some examples, generated with `python3 xformers/benchmarks/benchmark_encoder.py --activations gelu --plot -emb 256 -bs 32 -heads 16`

![Memory use for different attentions](docs/plots/memory_vs_attention.png)  ![Runtime for different attentions](docs/plots/runtime_vs_attention.png)

### Benchmark the core sparse attention mechanisms

`python3 xformers./benchmarks/benchmark_core.py` will measure the speed of the core sparse attention mechanism. The current numbers are as follows (times in microseconds (us)):

|                        | **matmul_with_mask**  |                        | **softmax**           |                        | **bmm**               |                        |
| ---------------------- | --------------------- | ---------------------- | --------------------- | ---------------------- | --------------------- | ---------------------- |
|                        | **B=8, M=256, K=128** | **B=8, M=1024, K=256** | **B=8, M=256, K=128** | **B=8, M=1024, K=256** | **B=8, M=256, K=128** | **B=8, M=1024, K=256** |
| dense                  | 62.3                  | 510.3                  | 12.8                  | 141.9                  | 31.0                  | 590.7                  |
| dense with masking     | 84.2                  | 805.3                  | -                     | -                      | -                     | -                      |
| sparsity pytorch: 0.50 | 392.4                 | 6197.4                 | 1140.9                | 8081.4                 | 577.0                 | 13830.2                |
| sparsity pytorch: 0.80 | 336.2                 | 4437.3                 | 515.0                 | 3494.8                 | 254.4                 | 5944.0                 |
| sparsity pytorch: 0.90 | 244.1                 | 3017.4                 | 367.3                 | 1932.6                 | 162.0                 | 3063.0                 |
| sparsity pytorch: 0.95 | 193.2                 | 1899.5                 | 293.6                 | 1078.9                 | 161.6                 | 1692.3                 |
| sparsity pytorch: 0.99 | 195.6                 | 695.0                  | 252.1                 | 342.4                  | 161.9                 | 433.4                  |
| sparsity sputnik: 0.50 | 77.9                  | 1695.9                 | 32.8                  | 164.7                  | 64.6                  | 1640.5                 |
| sparsity sputnik: 0.80 | 43.8                  | 793.0                  | 32.9                  | 50.8                   | 39.6                  | 703.3                  |
| sparsity sputnik: 0.90 | 43.6                  | 435.5                  | 33.0                  | 33.5                   | 39.6                  | 391.4                  |
| sparsity sputnik: 0.95 | 43.2                  | 258.6                  | 32.5                  | 32.7                   | 39.7                  | 223.6                  |
| sparsity sputnik: 0.99 | 43.5                  | 145.4                  | 33.2                  | 32.7                   | 39.7                  | 77.4                   |

### LRA

The code for this benchmark has been adapted from [this repository](https://github.com/mlpen/Nystromformer/tree/main/LRA). [A dedicated README is available here](xformers/benchmarks/LRA/README.md)

__Some results:__

| Attention                   | ListOps  | Text      | Retrieval | Image     | Pathfinder | *Avg*     | *Est. Gflops* | *Peak mem (mb)* |
| --------------------------- | -------- | --------- | --------- | --------- | ---------- | --------- | ------------- | --------------- |
| _Chance_                    | _10_     | _50_      | _50_      | _10_      | _50_       | _34_      | _0_           | _0_             |
| Standard                    | **37.5** | 62.66     | 79.24     | 38.69     | **70.37**  | **57.69** | 1.21          | 2291            |
| Nystromformer-128           | 36.29    | 63.24     | 78.18     | **42.86** | 67.49      | 57.61     | 0.62          | 383             |
| Favor-256 (redraw)          | 19.56    | 62.76     | **81.1**  | 36.09     | 67.23      | 53.35     | 0.49          | 445             |
| FourierMix                  | 36.29    | 60.72     | 76.41     | 36.53     | 54.07      | 52.8      | **0.17**      | **87**          |
| Linformer-seq/4 (no redraw) | 36.69    | 57.39     | 76.41     | 35.57     | 65.12      | 54.2      | 0.67          | 719             |
| Lambda                      | 19.76    | 62.47     | 79.11     | 35.04     | 49.74      | 49.224    | x             | 1023            |
| Orthoformer-32              | 27.42    | **63.96** | 77.96     | 34.5      | 67.11      | 54.19     | 0.187         | 155             |

- Contrary to the initial LRA proposal, __we use the same model architecture for all tasks (2 layers).__
- The training schedule for ListOps has been lengthened, while keeping it the fastest of all tasks, which reduces the seed dependence in the final accuracy figure.
- Estimated flops and peak memory are on the ListOps task, using 4 GPUs. Note that LRA is not completely well defined, in that hyperparameters and model architectures can vary (should the same architecture be used everywhere ? Similar hyperparams ?). This could be improved in the future, but in the meantime one should probably not read too much into small differences for some tasks, probably not meaningful.

_Note_: The estimated flops currently miss accounting for many operators, and are almost certainly an undercount. See issue [#154](https://github.com/fairinternal/xformers/issues/154)

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
