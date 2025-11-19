# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.34] - 2025-??-??

### Removed
- Removed optimized fast-path of SwiGLU (which was only available for A100 GPUs)

## [0.0.33] - 2025-11-12
Pre-built binary wheels are available for PyTorch 2.9.0.

### Added
- cutlass fmha Op for Blackwell GPUs
- Support flash-attention package up to 2.8.3
- expose FA3 deterministic mode
- FW+BW pass overlap for DeepSeek-like comms/compute overlap

### Improved
- merge_attentions support for irregular head dimension


## [0.0.32] - 2025-08-13
Pre-built binary wheels are available for PyTorch 2.8.0.

### Added
- Support flash-attention package up to 2.8.2
- Speed improvements to `python -m xformers.profiler.find_slowest`

### Removed
- Removed autograd backward pass for merge_attentions as it is easy to use incorrectly.
- Attention biases are no longer `torch.Tensor` subclasses. This is no longer
necessary for torch.compile to work, and was adding more complexity


## [0.0.31] - 2025-06-25
Pre-built binary wheels are available for PyTorch 2.7.1.
### Added
- xFormers wheels are now python-version agnostic: this means that the same wheel can be used for python 3.9, 3.10, ... 3.13
- Added support for Flash-Attention 3 on Ampere GPUs
### Removed
- We will no longer support V100 or older GPUs, following PyTorch (pytorch/pytorch#147607)
- Deprecated support for building Flash-Attention 2 as part of xFormers. For Ampere GPUs, we now use Flash-Attention 3 on windows, and Flash-Attention 2 can still be used through PyTorch on linux.

## [0.0.30] - 2025-04-28
Pre-built binary wheels are available for PyTorch 2.7.0. Following PyTorch, we build wheels for CUDA 11.8, 12.6, and 12.8 only (we no longer build for CUDA 12.4).
xFormers now requires PyTorch >= 2.7
### Added
- [fMHA] Added support for local attention on the Flash3 backend (H100)
- [fMHA] Added a new paged gappy attention bias
### Improved
- [fMHA] The FlashAttention3 backend now ships with more head dimensions to support MLA, and with a FLOPs formula in order to be compatible with PyTorch's partitioner-base automatic activation checkpointing
- The fused operators for sequence parallelism were migrated to PyTorch's SymmetricMemory
- The profiler prepends the traces' filenames with the rank of the process when doing distributed training
### Removed
- Removed documentation for legacy unmaintained components

## [0.0.29.post2] - 2025-01-31
Pre-built binary wheels are available for PyTorch 2.6.0. Following PyTorch, we build wheels for CUDA 11.8, 12.4, and 12.6 only (we no longer build for CUDA 12.1).
xFormers now requires PyTorch >= 2.6


## [0.0.29] - 2024-12-27
### Improved:
- [fMHA] Creating a `LowerTriangularMask` no longer creates a CUDA tensor
- [fMHA] Updated Flash-Attention to `v2.7.2.post1`
- [fMHA] Flash-Attention v3 will now be used by `memory_efficient_attention` by default when available, unless the operator is enforced with the `op` keyword-argument. Switching from Flash2 to Flash3 can make transformer trainings ~10% faster end-to-end on H100s
- [fMHA] Fixed a performance regression with the `cutlass` backend for the backward pass (facebookresearch/xformers#1176) - mostly used on older GPUs (eg V100)
- Fixed swiglu operator compatibility with torch-compile with PyTorch 2.6
- Fix activation checkpointing of SwiGLU when AMP is enabled (facebookresearch/xformers#1152)
### Removed:
- Following PyTorch, xFormers no longer builds binaries for conda. Pip is now the only recommended way to get xFormers
- Removed unmaintained/deprecated components in `xformers.components.*` (see facebookresearch/xformers#848)

## [0.0.28.post3] - 2024-10-30
Pre-built binary wheels require PyTorch 2.5.1

## [0.0.28.post2] - 2024-10-18
Pre-built binary wheels require PyTorch 2.5.0

## [0.0.28.post1] - 2024-09-13
Properly upload wheels for cuda 12.4

## [0.0.28] - 2024-09-12
Pre-built binary wheels require PyTorch 2.4.1
### Added
- Added wheels for cuda 12.4
- Added conda builds for python 3.11
- Added wheels for rocm 6.1
### Improved
- Profiler: Fix computation of FLOPS for the attention when using xFormers
- Profiler: Fix MFU/HFU calculation when multiple dtypes are used
- Profiler: Trace analysis to compute MFU & HFU is now much faster
- fMHA/splitK: Fixed `nan` in the output when using a `torch.Tensor` bias where a lot of consecutive keys are masked with `-inf`
- Update Flash-Attention version to `v2.6.3` *when building from scratch*
- When using the most recent version of Flash-Attention, it is no longer possible to mix it with the cutlass backend. In other words, it is no longer possible to use the cutlass Fw with the flash Bw.
### Removed
- fMHA: Removed `decoder` and `small_k` backends
- profiler: Removed `DetectSlowOpsProfiler` profiler
- Removed compatibility with PyTorch < 2.4
- Removed conda builds for python 3.11
- Removed windows pip wheels for cuda 12.1 and 11.8

## [0.0.27.post2] - 2024-07-26
Pre-built binary wheels require PyTorch 2.4.0

## [0.0.27.post1] - 2024-07-25
Pre-built binary wheels require PyTorch 2.4.0

## [0.0.27] - 2024-07-10
Pre-built binary wheels require PyTorch 2.3.1
### Added
- fMHA: `PagedBlockDiagonalGappyKeysMask`
- fMHA: heterogeneous queries in `triton_splitk`
- fMHA: support for paged attention in flash
- fMHA: Added backwards pass for `merge_attentions`
- fMHA: Added `torch.compile` support for 3 biases (`LowerTriangularMask`, `LowerTriangularMaskWithTensorBias` and `BlockDiagonalMask`) - some might require PyTorch 2.4
- fMHA: Added `torch.compile` support in `memory_efficient_attention` when passing the flash operator explicitely (eg `memory_efficient_attention(..., op=(flash.FwOp, flash.BwOp))`)
- fMHA: `memory_efficient_attention` now expects its `attn_bias` argument to be on the same device as the other input tensor. Previously, it would convert the bias to the right device.
- fMHA: `AttentionBias` subclasses are now constructed by default on the `cuda` device if available - they used to be created on the CPU device
- 2:4 sparsity: Added `xformers.ops.sp24.sparsify24_ste` for Straight Through Estimator (STE) with options to rescale the gradient differently for masked out/kept values
### Improved
- fMHA: Fixed out-of-bounds reading for Split-K triton implementation
- Profiler: fix bug with modules that take a single tuple as argument
- Profiler: Added manual trigger for a profiling step, by creating a `trigger` file in the profiling directory
### Removed
- Removed support for PyTorch version older than 2.2

## [0.0.26] - 2024-04-29
Pre-built binary wheels require PyTorch 2.3.0
### Added
- [2:4 sparsity] Added support for Straight-Through Estimator for `sparsify24` gradient (`GRADIENT_STE`)
- [2:4 sparsity] `sparsify24_like` now supports the cuSparseLt backend, and the STE gradient
- Basic support for `torch.compile` for the `memory_efficient_attention` operator. Currently only supports Flash-Attention, and without any bias provided. We want to expand this coverage progressively.
### Improved
- merge_attentions no longer needs inputs to be stacked.
- fMHA: triton_splitk now supports additive bias
- fMHA: benchmark cleanup

## [0.0.25.post1] - 2024-03-29
Pre-built binary wheels require PyTorch 2.2.2

## [0.0.25] - 2024-03-14
Pre-built binary wheels require PyTorch 2.2.1
### Added
- New `merge_attentions` function
- fMHA: New gappy attention biases.
### Improved
- fMHA: Updated Flash-Attention to v2.5.6: this has a performance improvement for multiquery.
- fMHA: triton_splitk changed and expanded. Now amalgamates using LSE. Can autotune, supports causal with a small number of queries - not just 1. Experimental support for paged attention.
- `rope_padded`: Fixed CUDA error with many queries (more than 65k)
- `rmsnorm`: Fixed CUDA error with large inputs (enables 512k+ sequence length on Llama2 70B)
### Removed
- fMHA: Removed triton operator (`fmha.triton.*`, `xformers.ops.MemoryEfficientAttentionTritonFwdFlashBwOp`, `xformers.ops.TritonFlashAttentionOp`), as it has correctness issues under some conditions, and is slower than other implementations.

## [0.0.24] - 2024-01-31
Pre-built binary wheels require PyTorch 2.2.0
### Added
- Added components for model/sequence parallelism, as near-drop-in replacements for FairScale/Megatron Column&RowParallelLinear modules. They support fusing communication and computation for sequence parallelism, thus making the communication effectively free. [Read more](https://twitter.com/d_haziza/status/1753030654118211593)
- Added kernels for training models with 2:4-sparsity. We introduced a very fast kernel for converting a matrix A into 24-sparse format, which can be used during training to sparsify weights dynamically, activations etc... xFormers also provides an API that is compatible with torch-compile, see `xformers.ops.sparsify24`.
### Improved
- Make selective activation checkpointing be compatible with torch.compile.
### Removed
- Triton kernels now require a GPU with compute capability 8.0 at least (A100 or newer). This is due to newer versions of triton not supporting older GPUs correctly
- Removed support for PyTorch version older than 2.1.0

## [0.0.23] - 2023-12-05
Pre-built binary wheels require PyTorch 2.1.1 (xFormers `0.0.23`) or PyTorch 2.1.2 (xFormers `0.0.23.post1`).
### Fixed
- fMHA: Fixed a bug in cutlass backend forward pass where the logsumexp was not correctly calculated, resulting in wrong results in the BW pass. This would happen with MQA when one sequence has a query with `length%64 == 1`
- fMHA: Updated Flash-Attention to v2.3.6 - this fixes a performance regression in causal backward passes, and now supports `BlockDiagonalCausalWithOffsetPaddedKeysMask`
### Added
- fMHA: Added `LocalAttentionFromBottomRightMask` (local)
- fMHA: Added `LowerTriangularFromBottomRightMask` (causal)
- fMHA: Added `LowerTriangularFromBottomRightLocalAttentionMask` (local + causal)
### Removed
- Removed `xformers.triton.sum_strided`

## [0.0.22] - 2023-09-27
### Fixed
- fMHA: Backward pass now works in PyTorch deterministic mode (although slower)
### Added
- fMHA: Added experimental support for Multi-Query Attention and Grouped-Query Attention. This is handled by passing 5-dimensional inputs to `memory_efficient_attention`, see the documentation for more details
- fMHA: Added experimental support for Local Attention biases to `memory_efficient_attention`
- Added an example of efficient [LLaMa decoding](https://github.com/facebookresearch/xformers/tree/main/examples/llama_inference) using xformers operators
- Added Flash-Decoding for faster attention during Large Language Model (LLM) decoding - up to 50x faster for long sequences (token decoding up to 8x faster end-to-end)
- Added an efficient rope implementation in triton, to be used in LLM decoding
- Added selective activation checkpointing, which gives fine-grained control of which activations to keep and which activations to recompute
- `xformers.info` now indicates the Flash-Attention version used
### Removed
- fMHA: Removed `smallK` backend support for CPU. `memory_efficient_attention` only works for CUDA/GPU tensors now
- **DEPRECATION**: Many classes in `xformers.factory`, `xformers.triton` and `xformers.components` have been or will be deprecated soon (see tracking issue facebookresearch/xformers#848)

## [0.0.21] - 2023-08-18
### Improved
- fMHA: Updated [flash-attention](https://github.com/Dao-AILab/flash-attention) to v2, with massive performance improvements for both the forward pass and backward pass. This implementation is now used by default when it's available
### Bug fixes
- fMHA/cutlass: Fix potential race condition in the FW/BW passes
- fMHA/cutlass: Fix `attn_bias` stride overflow for very long sequences (>32k)
- `LowerTriangularMask` is now backward compatible with older xformers versions
### Breaking changes
- `memory_efficient_attention` now expects the `attn_bias` argument to have a head dimension
- `memory_efficient_attention` no longer broadcasts the batch/head dimensions of `attn_bias`. Please use `.expand` if you need to broadcast the bias
- Remove `causal_diagonal` argument from `BlockDiagonalCausalWithOffsetPaddedKeysMask`
### Added
- Binary wheels on pypi/conda now contain H100 kernels
- fMHA: Added backend specialized for decoding that does not use TensorCores - useful when not using multiquery

**NOTE**: Binary wheels are now provided only for PyTorch 2 with cuda 11.8. It is still possible to use xFormers with older versions of PyTorch by building from source or using conda.


## [0.0.20] - 2023-05-23
### Improved
- fMHA/cutlass (backward): Massive performance improvements when `batch_size * num_heads` is low (10x+)
- fMHA/cutlass: Further performance improvements for both the forward & backward kernels
- fMHA (backward): Now dispatching to cutlass when `embed_dim>64`
- fMHA: Updated Flash-Attention to `v1.0.5`
### Added
- fMHA now runs on H100 (support is experimental)

## [0.0.19] - 2023-04-28
### Added
- Display `nvcc` version used to compile `xformers` in `python -m xformers.info`

### Fixed
- Fixed performance regression with `nvcc>11.6` (facebookresearch/xformers#712)
- fMHA/cutlass: Fixed `nan` in the output when using a `torch.Tensor` with `-inf` prefixes as `attn_bias` (facebookresearch/xformers#722)
- fMHA/cutlass: Fixed `nan` in the output when the sequence length is larger than `2 ** 15` (facebookresearch/xformers#719)
- fMHA/cutlass: Significative performance improvements (up to 2x) for both the forward pass and backward pass
- fMHA/cutlass: The kernel are now deterministic
- fMHA/cutlass: Fixed backward pass correctness when using dropout (facebookresearch/xformers#724)

## [0.0.18] - 2023-03-31
### Added
- Added `xformers.ops.index_select_cat` and `xformers.ops.scaled_index_add` - those are experimental functions that only work with a few shapes, and can be used to write efficient stochastic depth in transformer architectures for instance

### Fixed
- fMHA: `memory_efficient_attention` now accepts `torch.Tensor` as attention bias for any seqlen, although there are still requirements on the alignment of the bias tensor (see facebookresearch/xformers#683)

## [0.0.17] - 2023-03-28
### Fixed
- fMHA: Fixed BW pass on Sm86/Sm89 GPUs when `K > 64` (RTX 3090, RTX 4090, A6000, ..) [facebookresearch/xformers#631]

### Added
- fMHA/CUTLASS: Added tensor attn bias support [facebookresearch/xformers#587] - contribution from [@jfc4050](https://github.com/jfc4050)
- fMHA/CUTLASS: Added tensor attn bias grad support [facebookresearch/xformers#587] - contribution from [@jfc4050](https://github.com/jfc4050)
- fMHA/CUTLASS: Added dropout support [facebookresearch/xformers#587] - contribution from [@jfc4050](https://github.com/jfc4050)
- fMHA: Added support for varying sequence lengths [facebookresearch/xformers#500]


## [0.0.16] - 2023-01-31
### Fixed
- Updated triton dependency [facebookresearch/xformers#418]
- Stripe lineinfo from binaries, reducing the binary size [facebookresearch/xformers#549]
- Added support for pip wheels [facebookresearch/xformers#588, facebookresearch/xformers#573, facebookresearch/xformers#534, facebookresearch/xformers#523, ...] big thanks to [@AbdBarho](https://github.com/AbdBarho)!
- Fixed compatibility with Python 3.7 [facebookresearch/xformers#541] - thanks to [@susumuota](https://github.com/susumuota)
- fMHA: Fixed strides for QKV gradients for cutlass attention [facebookresearch/xformers#535]
- fMHA: Stricter inputs validation to avoid CUDA errors for unsupported inputs [facebookresearch/xformers#592]
- fMHA/Flash-Attention: Updated to https://github.com/HazyResearch/flash-attention/commit/a1f49a2b92b6fa022379bbebafed9d7f5e96a675 with multiple changes from [@TriDao](https://github.com/tridao) that make the operator up to 20% faster
- fMHA/Flash-Attention: Fixed backward pass wrapper, where non-contiguous gradients could give the wrong result [facebookresearch/xformers#548]
- fMHA: Separate each operator into forward and backward operators. It's now possible to use any combination of forward+backward (for instance Triton forward and Flash-Attention backward) [facebookresearch/xformers#560]

### Added
- fMHA: Added Triton operator for forward pass from [Flash-Attention](https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attn_triton.py) authored by [@TriDao](https://github.com/tridao), will be automatically used on A100 when compatible
- fMHA: Added [`xformers.ops.memory_efficient_attention_forward`](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.memory_efficient_attention_forward), [`xformers.ops.memory_efficient_attention_forward_requires_grad`](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.memory_efficient_attention_forward_requires_grad), [`xformers.ops.memory_efficient_attention_backward`](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.memory_efficient_attention_backward) for power-users who write custom autograd functions [facebookresearch/xformers#560]
- fMHA: Support for custom scaling for the CUTLASS-based kernel [facebookresearch/xformers#530] - contribution from [@comaniac](https://github.com/comaniac)

## [0.0.15] - Skipped

## [0.0.14] - 2022-11-10
### Fixed
- fMHA/CUTLASS: The current CUDA stream is now used by the kernel [facebookresearch/xformers#491]
- fMHA/CUTLASS: Improve overall performance

### Added
- SwiGLU: Added `xformers.ops.SwiGLU` and its functional counterpart (`xformers.ops.swiglu`) [facebookresearch/xformers#490]
- fMHA: Possible to combine CUTLASS's forward with flash-attention's backward pass [facebookresearch/xformers#469] - improves performance on A100 for K = 128
- fMHA: Add custom `xformers.ops.unbind` operator to avoid a cat in the attention block [facebookresearch/xformers#458]

## [0.0.13] - 2022-09-26
### Added
- fMHA: Added CUTLASS-based kernel for `xformers.ops.memory_efficient_attention`. This kernel is automatically depending on the inputs, and works on any GPU after P100 [facebookresearch/xformers#362]

## [0.0.12] - 2022-08-08
### Fixed
- Removed duplicated biases in the FusedMLP layers [facebookresearch/xformers#317]
- Rotary embeddings respecting input types [facebookresearch/xformers#326]
- Poolformer style instantiating useless projection layers [facebookresearch/xformers#349]
- Fix layer position not being properly tracked, causing extra layernorms for programmatic xformers [facebookresearch/xformers#348]
- Pass use_triton flag to LayerNorm module [facebookresearch/xformers#336]

### Added
- Four blocksparsity layouts from DeepSpeed [facebookresearch/xformers#320]
- Support several initialization options [facebookresearch/xformers#312]
- Conv2DFeedforward feedforward part [facebookresearch/xformers#321]
- VisualAttention [facebookresearch/xformers#329]
- Automatic blocksparse for causal attention [facebookresearch/xformers#334]
- Better hierarchical transformer generation [facebookresearch/xformers#345]
- Fused operations with AOTAutograd/NVFuser, integration into MLP [facebookresearch/xformers#357]
- Refactor LRA code to use Pytorch Lightning [facebookresearch/xformers#343]

## [0.0.11] - 2022-05-30
### Fixed
- Fix some torchscriptability [facebookresearch/xformers#246]
- Fix FourierMix being compatible with AMP [facebookresearch/xformers#258]
- Better asserts on QKV dimensions [facebookresearch/xformers#264]
- Better perfs for FusedMLP and FusedLinearLayer [facebookresearch/xformers#283]
- Deepnorm init missing self-attention [facebookresearch/xformers#284]

### Added
- Simplicial Embeddings [facebookresearch/xformers#259]
- Mem efficient attention, FW pass [facebookresearch/xformers#267]
- MHA benchmark
- MLP benchmark
- Move all triton kernels to triton v2 [facebookresearch/xformers#272]
- Mem efficient attention, BW pass [facebookresearch/xformers#281]
- Metaformer support [facebookresearch/xformers#294]

## [0.0.10] - 2022-03-14
### Fixed
- Expose bias flag for feedforwards, same default as Timm [facebookresearch/xformers#220]
- Update eps value for layernorm, same default as torch [facebookresearch/xformers#221]
- PreNorm bugfix, only one input was normalized [facebookresearch/xformers#233]
- Fix bug where embedding dimensions that did not match model dim would lead to a crash [facebookresearch/xformers#244]

### Added
- Add DeepNet (DeepNorm) residual path and init [facebookresearch/xformers#227]

## [0.0.9] - 2022-02-09
### Added
- Compositional Attention [facebookresearch/xformers#41]
- Experimental Ragged attention [facebookresearch/xformers#189]
- Mixture of Experts [facebookresearch/xformers#181]
- BlockSparseTensor [facebookresearch/xformers#202]
- Nd-tensor support for triton softmax [facebookresearch/xformers#210]

### Fixed
- Bugfix Favor, single feature map [facebookresearch/xformers#183]
- Sanity check blocksparse settings [facebookresearch/xformers#207]
- Fixed some picklability [facebookresearch/xformers#204]

## [0.0.8] - 2022-01-07
### Fixed
- Much faster fused dropout [facebookresearch/xformers#164]
- Fused dropout repeatability [facebookresearch/xformers#173]

### Added
- Embedding weight tying option [facebookresearch/xformers#172]

## [0.0.7] - 2021-11-30
### Fixed
- Dropout setting not properly passed in many attentions [facebookresearch/xformers#123]

## [0.0.6] - 2021-11-24
### Fixed
- Fix self attention optimization not being triggered, broken residual path [facebookresearch/xformers#119]
- Improve speed by not using contiguous Tensors when not needed [facebookresearch/xformers#119]

### Added
- Attention mask wrapper [facebookresearch/xformers#113]
- ViT comparison benchmark [facebookresearch/xformers#117]

## [0.0.4] - 2021-11-16
### Fixed
- Homogenizing the masks, additive or bool [facebookresearch/xformers#79][facebookresearch/xformers#85][facebookresearch/xformers#86]
- Fix causality flag not being respected [facebookresearch/xformers#103]
- Enabling FusedLayerNorm by default in the factory if Triton is available
- Fixing Favor with fp16
- Fixing Favor trainability

### Added
- Fused dropout/bias/activation layer [facebookresearch/xformers#58]
- Fused layernorm used by default in the factory [facebookresearch/xformers#92]


## [0.0.3] - 2021-11-01
### Fixed
- Nystrom causal attention [facebookresearch/xformers#75]


## [0.0.2] - 2021-11-01
### Fixed
- More robust blocksparse [facebookresearch/xformers#24]

### Added
- Rotary embeddings [facebookresearch/xformers#32]
- More flexible layernorm [facebookresearch/xformers#50]
