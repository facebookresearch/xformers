[![CircleCI](https://circleci.com/gh/fairinternal/xformers.svg?style=shield)](https://app.circleci.com/pipelines/github/fairinternal/xformers/)

# xFormers
Flexible Transformers, defined by interoperable and optimized building blocks that you can trust.

(all of this is inspirational for now..)

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



# (Known) TODOs:
## CI
    [ ] Tests
        [x] Auto load new variants in tests
        [ ] Waay more tests, find more invariants depending on the blocks

    [ ] Benchmark:
        [ ] add at least something basic to check training
        [ ] measure throughput and memory
            [ ] autogenerate text report
            [ ] autogenerate curves

## Architecture, code
    [ ] Remove the AttrDict dependency
    [x] Handle encoder/decoder builds
    [ ] MHA: expose the projection, possibly make it swappable
    [ ] MHA: expose "max sequence size" intead of fixed size

## Repo features
    [ ] Decent "bibliography" section
    [ ] Decent full model presets (matching bibliography ideally)

## Variants, at least first ones to add

    [ ] Performer
    [x] Local attention
    [ ] Big Bird
    [ ] ...


## Bibliography
DRAFT, needs a proper citation format, ..

### Attention is all you need
    https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf

### BigBird
    https://github.com/google-research/bigbird
    https://papers.nips.cc/paper/2020/file/c8512d142a2d849725f31a9a7a361ab9-Paper.pdf

### Nystromformer
    https://arxiv.org/abs/2102.03902

### Performer
    https://arxiv.org/abs/2009.14794v1

### Reformer
    https://arxiv.org/abs/2001.04451

### Longformer
    https://github.com/lucidrains/local-attention
    https://arxiv.org/pdf/2004.05150.pdf

### Adaptive Attention Span
    https://www.aclweb.org/anthology/P19-1032.pdf
