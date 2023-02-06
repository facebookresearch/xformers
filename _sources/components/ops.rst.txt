xFormers optimized operators
============================================================

Memory-efficient attention
---------------------------

.. automodule:: xformers.ops
    :members: memory_efficient_attention, AttentionOpBase, AttentionBias
    :show-inheritance:
    :imported-members:


Available implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: xformers.ops.fmha.cutlass
    :members: FwOp, BwOp
    :member-order: bysource

.. automodule:: xformers.ops.fmha.flash
    :members: FwOp, BwOp
    :member-order: bysource

.. automodule:: xformers.ops.fmha.triton
    :members: FwOp, BwOp
    :member-order: bysource

.. automodule:: xformers.ops.fmha.small_k
    :members: FwOp, BwOp
    :member-order: bysource

Attention biases
~~~~~~~~~~~~~~~~~~~~

.. automodule:: xformers.ops.fmha.attn_bias
    :members:
    :show-inheritance:
    :member-order: bysource


Non-autograd implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: xformers.ops.fmha
    :members: memory_efficient_attention_forward, memory_efficient_attention_forward_requires_grad, memory_efficient_attention_backward
    :show-inheritance:
    :imported-members:
    :member-order: bysource
