Using the Reversible block
==========================

Intro
-------

This block applies to residual paths, and was first proposed by Gomez et al ([1]_).
Its application in the Transformer ([3]_) context was first proposed in the `Reformer` ([2]_) paper,
and is largely unrelated to the other proposals from this paper (LSH and chunked MLP processing).

We use and very lightly adapt the implementation by Robin Bruegger_ and some blocks from LucidRains_.

A reversible layer requires two inputs (x1, x2) and produces two outputs (y1, y2)
via two functions F and G, following the relations

::

    y1 = x1 + F(x2)
    y2 = x2 + G(y1)


In turn, this means that (x1, x2) can be recovered from (y1, y2) (see [1]_ for details)

::

    x2 = y2 - G(y1)  # Note that another FW-like pass is needed
    x1 = y1 - F(x2)

The effect is comparable to activation checkpointing, in that it opens up for a tradeoff in between GPU memory
and compute. One benefit is that no extra wrap is needed, all the residual paths can be naturally checkpointed.
In a distributed setting, freeing up GPU memory can help using less GPUs, and the saved communication cost can more than make up for the extra compute.

Moreover, if your model is made of a stack of reversible blocks, then the memory requirement does not increase with the number of blocks.


Transformer
-----------

Considering the multi-head attention and feedforward blocks (including the residual paths), one can set F as MHA (+ layer norm) and G as Feedforward (+ layer norm) and get to something very close (but not exactly the same) to the original Transformer formulation from [Vaswani et al.][3], as follows
::

    y1 = x1 + MHA(x2)
    y2 = x2 + Feedforward(y1)

A difference is that the residual path in the Feedforward deals with the original input, and not the MHA output,
but in practice if `dim(x1) == dim(x2) == dim(model)`, the accuracy should not be affected, as verified in [2]_ and in xFormers.


In practice
-----------

This repository exposes two main helpers in `xformers.components.reversible`: ReversibleBlock and ReversibleSequence. `ReversibleBlock` will take `f` and `g` as defined above, and `ReversibleSequence` can combine them sequentially, similarly to `torch.nn.ModuleList`.

.. code-block:: python

    class ReversibleBlock(nn.Module):
        def __init__(self, f: nn.Module, g: nn.Module):
            ...

        def forward(self, x: torch.Tensor, f_args={}, g_args={}):
            ...


    class ReversibleSequence(nn.Module):
        def __init__(self, blocks: nn.ModuleList):
            ...

        def forward(self, x, arg_route=(True, False), **kwargs):
            """
            arg_route: whether to route the kwargs to f and g
            """
            ...

Reversible layers are also exposed as a boolean option in when building complete xFormers (which is optional), as defined in `xformers.factory.model_factory`. Please note that the reversible layer is not yet compatible with the use of multiple forward passes and DDP.

.. code-block:: python

    class xFormerStackConfig:
        block_config: Union[xFormerEncoderConfig, xFormerDecoderConfig]
        num_layers: int
        reversible: bool  # the sequence of layers becomes reversible


.. [1] Gomez, A. N., Ren, M., Urtasun, R., & Grosse, R. B. (2017).
    The reversible residual network: Backpropagation without storing activations.

.. [2] Kitaev, N., Kaiser, Ł., & Levskaya, A. (2020).
    Reformer: The Efficient Transformer.

.. [3] Vaswani et al.,
    Attention is all you need, 2017

.. _Bruegger: https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py
.. _LucidRains: https://github.com/lucidrains/reformer-pytorch/blob/master/reformer_pytorch/reversible.py
