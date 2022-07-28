How to Enable Fused Operations Using AOTAutograd and NVFuser
===================================================================

AOT Autograd is a toolkit from FuncTorch_ which can be used to accelerate model training in xFormers.
Broadly, it extracts a computational graph of the forward and backward passes of a model ahead of time.
This allows for some joint graph optimizations enables deep learning compilers such as NVFuser_ to perform operator fusion.
The `memory_efficient_fusion`_ wrapper function provides a convenient way to leverage AOTAutograd and NVFuser on GPU.

.. _FuncTorch: https://pytorch.org/functorch/stable/
.. _NVFuser: https://github.com/pytorch/pytorch/blob/release/1.12/torch/csrc/jit/codegen/cuda/README.md
.. _memory_efficient_fusion: https://pytorch.org/functorch/stable/generated/functorch.compile.memory_efficient_fusion.html#functorch.compile.memory_efficient_fusion

XFormers uses `memory_efficient_fusion` to combine sequences of fusable operations together into single fused function layers.
These parts can be found inside `xformers/components/nvfuser`. A notable example is `NVFusedBiasActivationDropout`, which is readily used inside the `MLP`_ feedforward component.

.. _MLP: https://github.com/facebookresearch/xformers/blob/main/xformers/components/feedforward/mlp.py

A benchmark of these fused patterns across some representative shapes shows significant speed increases compared to the unfused,
Pytorch eager approach―up to 3.5x speedup for the forward pass and 2.2x for the forward and backward passes together. On average, peak memory usage of fused patterns is also lower,
although we see some infrequent cases of up to 1.6x Pytorch peak memory usage on larger shapes. We also see better overall performance against our implementation of fused Bias,
Activation, and Dropout using Triton (see_) as well. Full benchmark plots can be found here_.

.. _see: https://github.com/facebookresearch/xformers/blob/main/xformers/triton/dropout.py
.. _here: https://github.com/facebookresearch/xformers/tree/main/docs/plots/nvfuser

Please note from README that the `_is_functorch_available` flag must be enabled for xFormers to use these optimizations.
This allows the fused layers to be used and changes the behavior of the `MLP` feedforward component,
causing it to default to using the fused `NVFusedBiasActivationDropout` layer.

AOT Autograd offers a great deal a flexibility to the user, as `memory_efficient_fusion` can accept either a Python function or an entire `nn.Module` as input for fusion.
Currently in xFormers, however, it is only used with Python function inputs because initial attempts with fusing xFormers layers and blocks have yielded memory issues and other CUDA errors.
We are currently exploring further testing and benchmarking.
