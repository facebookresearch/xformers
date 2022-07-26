AOTAutograd and NVFuser
==========================

AOT Autograd is a toolkit from FuncTorch_ can be used to accelerate model training in xFormers. Broadly, it extracts a computational graph of the forward and backward passes of a model ahead of time. This allows for some joint graph optimizations as well as enables deep learning compilers such as NVFuser_ to perform operator fusion. The `memory_efficient_fusion`_ wrapper function provides a convenient way to leverage AOTAutograd and NVFuser on GPU.

.. _FuncTorch: https://pytorch.org/functorch/stable/
.. _NVFuser: https://github.com/pytorch/pytorch/blob/release/1.12/torch/csrc/jit/codegen/cuda/README.md
.. _memory_efficient_fusion: https://pytorch.org/functorch/stable/generated/functorch.compile.memory_efficient_fusion.html#functorch.compile.memory_efficient_fusion

XFormers uses `memory_efficient_fusion` to combine sequences of fusable operations together into a single fused function layer. These parts can can be found in `xformers/components/nvfuser`_. A notable example is `NVFusedBiasActivationDropout`_, which is easily implementable inside the `MLP`_ feedforward component.

.. _xformers/components/nvfuser: https://github.com/facebookresearch/xformers/tree/main/xformers/components/nvfuser
.. _NVFusedBiasActivationDropout: https://github.com/facebookresearch/xformers/blob/main/xformers/components/nvfuser/bias_act_dropout.py
.. _MLP: https://github.com/facebookresearch/xformers/blob/main/xformers/components/feedforward/mlp.py

A benchmark of these fused patterns across some representative shapes shows significant speed increases compared to the unfused, Pytorch eager approach-- up to 3.5x speedup for the forward pass and 2.2x for the forward and backward passes together. We also see better overall performance against our implementation of fused Bias, Activation, and Dropout using Triton (see_) as well. Peak memory usage of fused patterns is also lower on average, although we see some infrequent cases of up to 0.6x higher peak memory usage on larger shapes. Full benchmark plots can be found here_.

.. _see: https://github.com/facebookresearch/xformers/blob/main/xformers/triton/dropout.py
.. _here: https://github.com/facebookresearch/xformers/tree/main/docs/plots/nvfuser

Below is a simple example use case of AOT Autograd.

.. code-block:: python

    import torch
    from functorch.compile import memory_efficient_fusion

    NORM_AXIS = 2

    def fn(a, b, c, norm_axis):
        x = a + b
        y = torch.nn.functional.layer_norm(x, (x.size(norm_axis),))
        return y + c

    # Test that it works
    a, b, c = [torch.randn(2, 4, 8, requires_grad=True) for _ in range(3)]
    ref = fn(a, b, c, NORM_AXIS)
    loss = ref.sum()
    loss.backward()

    # memory_efficient_fusion defaults to using NVFuser as the compiler backend
    # Label the non-tensor arguments of fn as static
    aot_fn = memory_efficient_fusion(fn, static_argnums=(3,))

    # Cloning inputs to check grads
    cloned_inputs = [x.clone().detach().requires_grad_(True) for x in (a, b, c)]
    c_a, c_b, c_c = cloned_inputs

    # Perform forward pass on fused function
    res = aot_fn(*cloned_inputs, NORM_AXIS)
    loss = res.sum()
    loss.backward()

    # Check for correctness
    assert torch.allclose(ref, res)
    assert torch.allclose(a.grad, c_a.grad)
    assert torch.allclose(b.grad, c_b.grad)
    assert torch.allclose(c.grad, c_c.grad)

AOT Autograd offers a great deal a flexibility to the user, as `memory_efficient_fusion` can accept either a Python function or an entire `nn.Module` as input for fusion. Currently in xFormers, however, it is only used with Python function inputs because initial attempts with fusing xFormers layers and blocks have yielded memory issues and other CUDA errors. We are currently exploring further testing and benchmarking.
