xFormers optimized operators
============================================================

Memory-efficient attention
---------------------------

.. automodule:: xformers.ops
    :members: memory_efficient_attention, AttentionOpBase
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

.. automodule:: xformers.ops.fmha.small_k
    :members: FwOp, BwOp
    :member-order: bysource

.. automodule:: xformers.ops.fmha.ck
    :members: FwOp, BwOp
    :member-order: bysource

.. automodule:: xformers.ops.fmha.ck_decoder
    :members: FwOp
    :member-order: bysource

.. automodule:: xformers.ops.fmha.ck_splitk
    :members: FwOp
    :member-order: bysource

Attention biases
~~~~~~~~~~~~~~~~~~~~

.. automodule:: xformers.ops.fmha.attn_bias
    :members:
    :show-inheritance:
    :member-order: bysource

Partial Attention
~~~~~~~~~~~~~~~~~~~~

.. automodule:: xformers.ops.fmha
    :members: memory_efficient_attention_partial, merge_attentions
    :member-order: bysource


Non-autograd implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: xformers.ops.fmha
    :members: memory_efficient_attention_forward, memory_efficient_attention_forward_requires_grad, memory_efficient_attention_backward
    :show-inheritance:
    :imported-members:
    :member-order: bysource


AETHER Geometric Sparse Attention
----------------------------------

AETHER (Adaptive Event-driven Threshold Hybrid Entangled Rendering) is a geometric
sparse attention operator that achieves O(N_relevant) scaling by pruning blocks
that fall outside the active query manifold using Cauchy-Schwarz upper bounds.

.. automodule:: xformers.ops.aether_attention
    :members: aether_attention, AetherAttention, compute_block_geometry
    :show-inheritance:
    :member-order: bysource
