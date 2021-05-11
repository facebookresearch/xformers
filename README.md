[![CircleCI](https://circleci.com/gh/fairinternal/xformers.svg?style=shield)](https://app.circleci.com/pipelines/github/fairinternal/xformers/)

# xFormers
Flexible Transformers, defined by interoperable and optimized building blocks that you can trust.

# Key concepts
- **Field agnostic**. This repo is not focused on NLP, speech or vision, by design. The focus is on making sure that building transformers is a shared building block in between
- **Composable**. There are basically two obvious takes to the Transformer ecosystem:
  - support all the architectures proposed in different papers more or less individually
  - break all the achitectures into a block zoo, which allows you to recompose said reference models, but also study ablations, extensions easily, or field specific architectures (ie: sweep over the SOTA to see whether some technique is more applicable to a specific field).
    This repo aims at supporting this second way.

- **Extensible**. This is meant to support a research effort, and not to substitute to modelling and field specific know how, or to constraint into the use of specific blocks.
  xFormers aims at being easy to extend locally, so that one can focus on a specific improvement, and easily compare it against the state of the art.

- **Optimized**. This is largely aspirational for now, but reusing building blocks across teams and domains should mean that engineering efforts can be more valued, so that investing in speeding up some key building blocks (without breaking compatibility thanks to the settled interface) would make sense.

- **Tested**. Each and every of the variant in the repo should be tested, alone and composed with the other relevant blocks. This should happen automatically anytime somebody proposes a new variant through a PR

- **Crowd Sourced**. This is probably the single most important part. All of the above should make it possible for people interested to contribute: contributing on a small block is easier than on a full model, unit tests and common interfaces should help, the ability to extend the library locally and test the relevance prior to a PR should also help. PRs are really welcome.

# Using xFormers
Below you will find a set of notebooks that will show you how you can use xFormers in your project
- [Creating complex sparsity patterns with xformers](docs/source/2d_attention_patterns.ipynb)
- [Changing ViT to use sparse attention, benchmarking the effects](docs/source/vision_transformers.ipynb)


## Adding new variants
Here are a couple of guidelines which should make it easier to add a new block variant to this repo:
* Blocks live in `xformers/components`
* Make sure that the block and its config inherit from the ones defined in the base file
* Default values need to be defined in the class constructor, not in the config.
  * Using the config objects is optional, people should feel free to cherry pick the blocks as is
  * Prevent duplication or colliding definitions
  * Make sure that the configurations are composable, in that a subset of the configs is enough to instantiate all the blocks with reasonable settings.
* Fields which have default values in the block constructor should be typed as `Optional[Type]`
* Please follow the CONTRIBUTING guide to make sure that formatting and linting is checked
* `@register` your new block variant with a unique and hopefully descriptive name
* Just define the (pure pytorch) constructor and forward call typically, no need to handle enything specific to this repo (except for inheritance)
* keep `*args` and `**kwargs` in your constructor, this is important for the config composition
* No need to change unit tests or benchmarks, the new variant will be automatically picked up

That's it. Rest assured that the community will be thankful for your contribution !


## Adding new models
Models live in `xformers/models`. As a general rule, one should try to write them using the blocks present in `xformers/components` (or upstream PyTorch), so that ulterior improvements are propagated to each implementation.

## Some benchmark tools
These live in `xformers/benchmarks`.

### Benchmark a full encoder block
Sweeping over different attention settings to log max memory use and runtime can for instance be done by invoking
`python3 benchmarks/benchmark_encoder.py`. Specifying a subset to test is done through command line arguments, for instance `python3 benchmarks/benchmark_encoder.py --causal True --attentions random --activations gelu -fp16 True`.

Please note that:
- These numbers are dependent of hyperparameters (dimensions chosen for Linformer, sparsity of the pattern), they are mostly an illustration
- The sparse attention patterns tested here are just presets, as explained in the linked notebook generating any new sparse attention pattern should be relatively easy, while keeping the benefits of optimized computations.

Some examples, generated with `python3 benchmarks/benchmark_encoder.py --activations gelu --plot -emb 256 -bs 32 -heads 16`

![](docs/plots/memory_vs_attention.png)

![](docs/plots/runtime_vs_attention.png)

### Benchmark the core sparse attention mechanisms
`python3 benchmarks/benchmark_core.py` will measure the speed of the core sparse attention mechanism. The current numbers are as follows:

```
[--------------------------- matmul_with_mask --------------------------]
                              |  B=8, M=256, K=128  |  B=8, M=1024, K=256
1 threads: --------------------------------------------------------------
      dense                   |         62.3        |         510.3
      dense with masking      |         84.2        |         805.3
      sparsity pytorch: 0.50  |        392.4        |        6197.4
      sparsity pytorch: 0.80  |        336.2        |        4437.3
      sparsity pytorch: 0.90  |        244.1        |        3017.4
      sparsity pytorch: 0.95  |        193.2        |        1899.5
      sparsity pytorch: 0.99  |        195.6        |         695.0
      sparsity sputnik: 0.50  |         77.9        |        1695.9
      sparsity sputnik: 0.80  |         43.8        |         793.0
      sparsity sputnik: 0.90  |         43.6        |         435.5
      sparsity sputnik: 0.95  |         43.2        |         258.6
      sparsity sputnik: 0.99  |         43.5        |         145.4

Times are in microseconds (us).

[------------------------------- softmax -------------------------------]
                              |  B=8, M=256, K=128  |  B=8, M=1024, K=256
1 threads: --------------------------------------------------------------
      dense                   |          12.8       |         141.9
      sparsity pytorch: 0.50  |        1140.9       |        8081.4
      sparsity pytorch: 0.80  |         515.0       |        3494.8
      sparsity pytorch: 0.90  |         367.3       |        1932.6
      sparsity pytorch: 0.95  |         293.6       |        1078.9
      sparsity pytorch: 0.99  |         252.1       |         342.4
      sparsity sputnik: 0.50  |          32.8       |         164.7
      sparsity sputnik: 0.80  |          32.9       |          50.8
      sparsity sputnik: 0.90  |          33.0       |          33.5
      sparsity sputnik: 0.95  |          32.5       |          32.7
      sparsity sputnik: 0.99  |          33.2       |          32.7

Times are in microseconds (us).

[--------------------------------- bmm ---------------------------------]
                              |  B=8, M=256, K=128  |  B=8, M=1024, K=256
1 threads: --------------------------------------------------------------
      dense                   |         31.0        |         590.7
      sparsity pytorch: 0.50  |        577.0        |       13830.2
      sparsity pytorch: 0.80  |        254.4        |        5944.0
      sparsity pytorch: 0.90  |        162.0        |        3063.0
      sparsity pytorch: 0.95  |        161.6        |        1692.3
      sparsity pytorch: 0.99  |        161.9        |         433.4
      sparsity sputnik: 0.50  |         64.6        |        1640.5
      sparsity sputnik: 0.80  |         39.6        |         703.3
      sparsity sputnik: 0.90  |         39.6        |         391.4
      sparsity sputnik: 0.95  |         39.7        |         223.6
      sparsity sputnik: 0.99  |         39.7        |          77.4

Times are in microseconds (us).
```

## Bibliography
Some references or papers used in the repo
- [Attention is all you need, Vaswani et al., 2017](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- [Linformer, self-attention with linear complexity, Wang et al., 2020](https://arxiv.org/pdf/2006.04768.pdf)
- [BigBird, Transformer for longer sequences, Zaheer et al., 2020](https://papers.nips.cc/paper/2020/file/c8512d142a2d849725f31a9a7a361ab9-Paper.pdf)
- [Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention, Xiong et al., 2021](https://arxiv.org/abs/2102.03902)
- [Rethinking Attention with Performers, Choromanski et al., 2020](https://arxiv.org/abs/2009.14794v1)
- [Reformer: The Efficient Transformer, Kitaev et al., 2020](https://arxiv.org/abs/2001.04451)
- [Longformer: The Long-Document Transformer, Beltagy et al., 2020](https://arxiv.org/pdf/2004.05150.pdf)
- [Long Range Arena: a benchmark for efficient Transformers, Tay et al., 2020](https://arxiv.org/abs/2011.04006)
