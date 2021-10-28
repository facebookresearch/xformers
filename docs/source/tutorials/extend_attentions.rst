Extend the xFormers parts zoo
=============================

This can be done in a private fork of xFormers, if this is a work in progress or not something that you would
like to share at this point, or directly in xFormers in order to submit a PR_.

We follow a register-based architecture, which is practical for unit testing, and loose inheritance
(not all blocks expose the exact same interface).

Let's consider for instance the Nystrom-based attention mechanism:

.. _PR: https://github.com/fairinternal/xformers/pulls

.. code-block:: python

    @dataclass
    class NystromSelfAttentionConfig(AttentionConfig):
        ...

    @register_attention("nystrom", NystromSelfAttentionConfig)
    class NystromAttention(Attention):
        def __init__(
            self,
            dropout: float,
            num_heads: int,
            num_landmarks: int = 64,
            landmark_pooling: Optional[nn.Module] = None,
            causal: bool = False,
            use_razavi_pinverse: bool = True,
            pinverse_original_init: bool = False,
            inv_iterations: int = 6,  # recommended default in paper was 6.
            v_skip_connection: Optional[nn.Module] = None,
            conv_kernel_size: Optional[int] = None,
            *args,
            **kwargs,
        ):
            ...


        def forward(
            self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *args, **kwargs
        ):
            ...


There are a couple of things to remark, which would be true of any other extension. This also applies to the other components in xformers:

- The attention mechanism inherits from the base component
- We define a configuration for this block, which is not explicitly used in the constructor, but is required if you want to register this block. It is for instance used for unit testing and benchmarking.
- The construction needs to accept extra `*args, **kwargs` arguments, so that the same configuration can be used by different blocks, even if not all fields are useful
- The registration is done with the following snippet, which both registers this attention with a given name, and ties a configuration to it. The same would apply to the other component types.

.. code-block:: python

    @register_attention("nystrom", NystromSelfAttentionConfig)
    class NystromAttention(Attention):


Doing this opens up at least three tools in the xFormers toolbox:

- the relevant unit tests will now automatically pick up this new variant. You can call all of them in one go with ::

    pytest -x -k my_component_name

- if applicable (attention mechanism), the attention benchmark will pick up this new variant automatically
- the LRA benchmarks will pick up this new block option. You can define a JSON config with your new part and trigger LRA jobs.

As a reminder (more information in the dedicated README) you can trigger a LRA job locally with::

    python3 run_tasks.py --attention <your attention name> --task <task> --config <config_path> --world_size N

or even submit a batch of jobs to a SLURM enabled cluster with::

    python3 batch_submit.py -c code/config.json -ck <your checkpoing and log path> -a <your attention name>
