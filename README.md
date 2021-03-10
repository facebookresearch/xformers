[![CircleCI](https://circleci.com/gh/fairinternal/xformers.svg?style=shield)](https://app.circleci.com/pipelines/github/fairinternal/xformers/)

## xFormers
Flexible Transformers, defined by interoperable and optimized building blocks that you can trust.

(all of this is inspirational for now..)

# TODOs:
## CI

    [x] Auto load new variants in tests
    [ ] Waay more tests
    [ ] Benchmark:
        [ ] add at least something basic to check training
        [ ] measure throughput and memory, autogenerate text report
        [ ] measure throughput and memory, autogenerate curves

## Architecture, code
    [ ] Remove the AttrDict dependency
    [ ] Handle encoder/decoder builds

## Repo features
    [ ] Decent "bibliography" section
    [ ] Full model presets

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
