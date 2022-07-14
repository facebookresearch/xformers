Building an encoder, comparing to PyTorch
=========================================

Let's now walk up the hierarchy, and consider a whole encoder block. You may be used to the PyTorch encoder layer so we'll consider it as a point of comparison, but other libraries would probably expose similar interfaces.

PyTorch Encoder Layer
---------------------

PyTorch already exposes a TransformerEncoderLayer_. Its constructor is:

.. _TransformerEncoderLayer: https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html?highlight=encoder#torch.nn.TransformerEncoderLayer

.. code-block:: python

    TransformerEncoderLayer(
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation='relu',
        layer_norm_eps=1e-05,
        batch_first=False,
        device=None,
        dtype=None
        ):
        ...

Note that you cannot change the attention mechanism, so this example will use the "Scaled Dot Product", as proposed by Vaswani et al., but in the xFormers case this is a free floating parameter.

Warning
-------

It’s worth noting that **xFormer’s blocks expect tensors to be batch first, while PyTorch’s transformers uses a sequence first convention. Don’t forget to permute if you use xFormers’s blocks as drop-in replacements.**

Similarly, the attention masks conventions are different: in PyTorch, the mask is *True* when an element should *not* be attended to, whereas in xFormer it’s the opposite. Don’t forget to negate your attention masks to use xFormers’ blocks as drop-in replacements.

Block factory
-------------

We don't have the exact same interfaces, but we have something fairly close to PyTorch with the model_factory_. Please note that, similarly to the attention example above, you can also directly import the `xFormerEncoderBlock` and construct it from there, but we'll assume here that you could be interested in systematic evaluation of different architectures, and that as such something which can be easily automated is preferred, so the "factory" path is the one put forward.

The equivalent to the PyTorch example above would look like the following. You can think of it  as a declaration of the sequence of blocks that you would like instantiated. We're trying to:

- make it very explicit what is in this block
- keep everything pythonic
- make this sweep and automation friendly in general

With this said, you can build an encoder directly as follows:

.. code-block:: python

    from xformers.factory import xFormerEncoderBlock, xFormerEncoderConfig
    import torch

    BATCH = 8
    SEQ = 1024
    EMB = 384
    VOCAB = 64

    encoder_config = {
        "dim_model": EMB,
        "residual_norm_style": "pre",  # Optional, pre/post
        "position_encoding_config": {
            "name": "vocab",  # whatever position encodinhg makes sense
            "seq_len": SEQ,
            "vocab_size": VOCAB,
        },
        "multi_head_config": {
            "num_heads": 4,
            "residual_dropout": 0,
            "attention": {
                "name": "linformer",  # whatever attention mechanism
                "dropout": 0,
                "seq_len": SEQ,
            },
        },
        "feedforward_config": {
            "name": "MLP",
            "dropout": 0,
            "activation": "relu",
            "hidden_layer_multiplier": 4,
        },
    }

    # "constructing" the config will lead to a lot of type checking,
    # which could catch some errors early on
    config = xFormerEncoderConfig(**encoder_config)

    encoder = xFormerEncoderBlock(config)

    #  Test out with dummy inputs
    x = (torch.rand((BATCH, SEQ)) * VOCAB).abs().to(torch.int)
    y = encoder(x, x, x)
    print(y)


Building full models
====================


 Now let's build a full Tranformers/xFormer model. Please note that this is just an example, because building the whole model from explicit parts is always an option, from pure PyTorch building blocks or adding some xFormers primitives.

PyTorch Transformer
-------------------

Am implementation of a full Transformer is supported directly by PyTorch, see the PyTorchTransformer_ for more options.

.. _PyTorchTransformer: https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html?highlight=transformer#torch.nn.Transformer

.. code-block:: python

    Transformer(
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation='relu',
        custom_encoder=None, # the xFormers exemple below defines that
        custom_decoder=None, # Same
        layer_norm_eps=1e-05,
        batch_first=False,
        device=None,
        dtype=None):
        .

model factory
-------------

We don't have the exact same interfaces, but we have something to propose with the model_factory_.
Please note that, similarly to the attention example above, you can also directly import the `xFormer` and `xFormerConfig`
and construct it from there, but we'll assume here that you could be interested in systematic evaluation of different architectures,
and that as such something which can be easily automated is preferred, so the "factory" path is the one put forward.

.. _model_factory: https://github.com/facebookresearch/xformers/blob/main/xformers/factory/model_factory.py

The equivalent to the PyTorch example above would look like the following.
You can think of it  as a declaration of the sequence of blocks that you would like instantiated.
This is not really apples to apples, because we define a custom encoder and decoder here.
There's also an added flexibility with xFormers in that attention mechanisms can be chosen at will, on a per-layer basis.

.. code-block:: python

    from xformers.factory.model_factory import xFormer, xFormerConfig
    import torch

    EMB = 384
    SEQ = 1024
    BATCH = 16
    VOCAB = 64

    my_config = [
        # A list of the encoder or decoder blocks which constitute the Transformer.
        # Note that a sequence of different encoder blocks can be used, same for decoders
        {
            "reversible": False,  # Optionally make these layers reversible, to save memory
            "block_type": "encoder",
            "num_layers": 3,  # Optional, this means that this config will repeat N times
            "dim_model": EMB,
            "residual_norm_style": "pre",  # Optional, pre/post
            "position_encoding_config": {
                "name": "vocab",  # whatever position encodinhg makes sense
                "seq_len": 1024,
                "vocab_size": VOCAB,
            },
            "multi_head_config": {
                "num_heads": 4,
                "residual_dropout": 0,
                "attention": {
                    "name": "linformer",  # whatever attention mechanism
                    "dropout": 0,
                    "causal": False,
                    "seq_len": SEQ,
                },
            },
            "feedforward_config": {
                "name": "MLP",
                "dropout": 0,
                "activation": "relu",
                "hidden_layer_multiplier": 4,
            },
        },
        {
            "reversible": False,  # Optionally make these layers reversible, to save memory
            "block_type": "decoder",
            "num_layers": 3,  # Optional, this means that this config will repeat N times
            "dim_model": EMB,
            "residual_norm_style": "pre",  # Optional, pre/post
            "position_encoding_config": {
                "name": "vocab",  # whatever position encodinhg makes sense
                "seq_len": SEQ,
                "vocab_size": VOCAB,
            },
            "multi_head_config_masked": {
                "num_heads": 4,
                "residual_dropout": 0,
                "attention": {
                    "name": "nystrom",  # whatever attention mechanism
                    "dropout": 0,
                    "causal": True,
                    "seq_len": SEQ,
                },
            },
            "multi_head_config_cross": {
                "num_heads": 4,
                "residual_dropout": 0,
                "attention": {
                    "name": "favor",  # whatever attention mechanism
                    "dropout": 0,
                    "causal": True,
                    "seq_len": SEQ,
                },
            },
            "feedforward_config": {
                "name": "MLP",
                "dropout": 0,
                "activation": "relu",
                "hidden_layer_multiplier": 4,
            },
        },
    ]

    # This part of xFormers is entirely type checked and needs a config object,
    # could be changed in the future
    config = xFormerConfig(my_config)
    model = xFormer.from_config(config)

    #  Test out with dummy inputs
    x = (torch.rand((BATCH, SEQ)) * VOCAB).abs().to(torch.int)
    y = model(src=x, tgt=x)
    print(y)


Note that this exposes quite a few more knobs than the PyTorch Transformer interface, but in turn is probably a little more flexible. There are a couple of repeated settings here (dimensions mostly), this is taken care of in the `LRA benchmarking config`_.

.. _LRA benchmarking config: https://github.com/facebookresearch/xformers/blob/main/xformers/benchmarks/LRA/code/config.json

You can compare the speed and memory use of the vanilla PyTorch Transformer Encoder and an equivalent from xFormers, there is an existing benchmark for that (see_).
It can be run with `python3 xformers/benchmarks/benchmark_pytorch_transformer.py`, and returns the loss values for every step along with the training time for a couple of shapes that you can customize.
Current results are as follows, on a nVidia V100 (PyTorch 1.9, Triton 1.1, xFormers 0.0.2):

.. _see: https://github.com/facebookresearch/xformers/blob/main/xformers/benchmarks/benchmark_pytorch_transformer.py

.. code-block:: bash

    --- Transformer training benchmark - runtime ---
    | Units: s | emb 128 - heads 8 | emb 1024 - heads 8 | emb 2048 - heads 8 |
    | -------- | ----------------- | ------------------ | ------------------ |
    | xformers | 0.3               | 0.4                | 0.7                |
    | pytorch  | 0.2               | 0.6                | 0.8                |

    --- Transformer training benchmark - memory use ---
    | Units: MB | emb 128 - heads 8 | emb 1024 - heads 8 | emb 2048 - heads 8 |
    | --------- | ----------------- | ------------------ | ------------------ |
    | xformers  | 89                | 1182               | 2709               |
    | pytorch   | 155               | 1950               | 4117               |



Build an `xFormer` model with Hydra
-----------------------------------

Alternatively, you can use Hydra_ to build an xFormer model.
We've included an example `here <https://github.com/facebookresearch/xformers/tree/main/examples/build_model/>`_.
The example replicates the model from the above example and demonstrates one way to use Hydra to minimize config duplication.
The example is built on top of some more advanced Hydra features. If you are new to Hydra, you can start these docs:
`basic tutorials <https://hydra.cc/docs/tutorials/intro/>`_, `extending configs <https://hydra.cc/docs/patterns/extending_configs/>`_,
`Hydra packages <https://hydra.cc/docs/advanced/overriding_packages/>`_ and
`instantiation API <https://hydra.cc/docs/advanced/instantiate_objects/overview/>`_.

.. _Hydra: https://hydra.cc/

.. code-block:: yaml

    defaults:
        - /stack@xformer.stack_configs:
            - encoder_local
            - encoder_random
            - decoder_nystrom_favor
        - _self_

    xformer:
        _target_: xformers.factory.model_factory.xFormer


Building a model this way makes it possible for you to leverage many features Hydra has to offer.
For example, you can override the model architecture from the commandline:

.. code-block:: bash

    python examples/build_model/my_model.py  'stack@xformer.stack_configs=[encoder_local]'

    Built a model with 1 stack: dict_keys(['encoder_local'])
        xFormer(
        (encoders): ModuleList(
            (0): xFormerEncoderBlock(
            (pose_encoding): VocabEmbedding(
                (dropout): Dropout(p=0, inplace=False)
                (position_embeddings): Embedding(1024, 384)
                (word_embeddings): Embedding(64, 384)
            )
            (mha): MultiHeadDispatch(
                (attention): LocalAttention(
                (attn_drop): Dropout(p=0.0, inplace=False)
                )
                (in_proj_container): InputProjection()
                (resid_drop): Dropout(p=0, inplace=False)
                (proj): Linear(in_features=384, out_features=384, bias=True)
            )
            (feedforward): MLP(
                (mlp): Sequential(
                (0): Linear(in_features=384, out_features=1536, bias=True)
                (1): ReLU()
                (2): Dropout(p=0, inplace=False)
                (3): Linear(in_features=1536, out_features=384, bias=True)
                (4): Dropout(p=0, inplace=False)
                )
            )
            (wrap_att): Residual(
                (layer): PreNorm(
                (norm): FusedLayerNorm()
                (sublayer): MultiHeadDispatch(
                    (attention): LocalAttention(
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    )
                    (in_proj_container): InputProjection()
                    (resid_drop): Dropout(p=0, inplace=False)
                    (proj): Linear(in_features=384, out_features=384, bias=True)
                )
                )
            )
            (wrap_ff): PostNorm(
                (norm): FusedLayerNorm()
                (sublayer): Residual(
                (layer): PreNorm(
                    (norm): FusedLayerNorm()
                    (sublayer): MLP(
                    (mlp): Sequential(
                        (0): Linear(in_features=384, out_features=1536, bias=True)
                        (1): ReLU()
                        (2): Dropout(p=0, inplace=False)
                        (3): Linear(in_features=1536, out_features=384, bias=True)
                        (4): Dropout(p=0, inplace=False)
                    )
                    )
                )
                )
            )
            )
        )
        (decoders): ModuleList()
        )


You can also launch multiple runs of your application with different architectures:

.. code-block:: bash

    $ python my_model.py  --multirun 'stack@xformer.stack_configs=[encoder_local], [encoder_random]'
    [HYDRA] Launching 2 jobs locally
    [HYDRA]        #0 : stack@xformer.stack_configs=[encoder_local]
    Built a model with 1 stack: dict_keys(['encoder_local'])
    xFormer(
    (encoders): ModuleList(
        (0): xFormerEncoderBlock(
        (pose_encoding): VocabEmbedding(
            (dropout): Dropout(p=0, inplace=False)
            (position_embeddings): Embedding(1024, 384)
            (word_embeddings): Embedding(64, 384)
        )
        (mha): MultiHeadDispatch(
            (attention): LocalAttention(
            (attn_drop): Dropout(p=0.0, inplace=False)
            )
            (in_proj_container): InputProjection()
            (resid_drop): Dropout(p=0, inplace=False)
            (proj): Linear(in_features=384, out_features=384, bias=True)
        )
        (feedforward): MLP(
            (mlp): Sequential(
            (0): Linear(in_features=384, out_features=1536, bias=True)
            (1): ReLU()
            (2): Dropout(p=0, inplace=False)
            (3): Linear(in_features=1536, out_features=384, bias=True)
            (4): Dropout(p=0, inplace=False)
            )
        )
        (wrap_att): Residual(
            (layer): PreNorm(
            (norm): FusedLayerNorm()
            (sublayer): MultiHeadDispatch(
                (attention): LocalAttention(
                (attn_drop): Dropout(p=0.0, inplace=False)
                )
                (in_proj_container): InputProjection()
                (resid_drop): Dropout(p=0, inplace=False)
                (proj): Linear(in_features=384, out_features=384, bias=True)
            )
            )
        )
        (wrap_ff): PostNorm(
            (norm): FusedLayerNorm()
            (sublayer): Residual(
            (layer): PreNorm(
                (norm): FusedLayerNorm()
                (sublayer): MLP(
                (mlp): Sequential(
                    (0): Linear(in_features=384, out_features=1536, bias=True)
                    (1): ReLU()
                    (2): Dropout(p=0, inplace=False)
                    (3): Linear(in_features=1536, out_features=384, bias=True)
                    (4): Dropout(p=0, inplace=False)
                )
                )
            )
            )
        )
        )
    )
    (decoders): ModuleList()
    )
    [HYDRA]        #1 : stack@xformer.stack_configs=[encoder_random]
    Built a model with 1 stack: dict_keys(['encoder_random'])
    xFormer(
    (encoders): ModuleList(
        (0): xFormerEncoderBlock(
        (pose_encoding): VocabEmbedding(
            (dropout): Dropout(p=0, inplace=False)
            (position_embeddings): Embedding(1024, 384)
            (word_embeddings): Embedding(64, 384)
        )
        (mha): MultiHeadDispatch(
            (attention): RandomAttention(
            (attn_drop): Dropout(p=0.0, inplace=False)
            )
            (in_proj_container): InputProjection()
            (resid_drop): Dropout(p=0, inplace=False)
            (proj): Linear(in_features=384, out_features=384, bias=True)
        )
        (feedforward): MLP(
            (mlp): Sequential(
            (0): Linear(in_features=384, out_features=1536, bias=True)
            (1): ReLU()
            (2): Dropout(p=0, inplace=False)
            (3): Linear(in_features=1536, out_features=384, bias=True)
            (4): Dropout(p=0, inplace=False)
            )
        )
        (wrap_att): Residual(
            (layer): PreNorm(
            (norm): FusedLayerNorm()
            (sublayer): MultiHeadDispatch(
                (attention): RandomAttention(
                (attn_drop): Dropout(p=0.0, inplace=False)
                )
                (in_proj_container): InputProjection()
                (resid_drop): Dropout(p=0, inplace=False)
                (proj): Linear(in_features=384, out_features=384, bias=True)
            )
            )
        )
        (wrap_ff): PostNorm(
            (norm): FusedLayerNorm()
            (sublayer): Residual(
            (layer): PreNorm(
                (norm): FusedLayerNorm()
                (sublayer): MLP(
                (mlp): Sequential(
                    (0): Linear(in_features=384, out_features=1536, bias=True)
                    (1): ReLU()
                    (2): Dropout(p=0, inplace=False)
                    (3): Linear(in_features=1536, out_features=384, bias=True)
                    (4): Dropout(p=0, inplace=False)
                )
                )
            )
            )
        )
        )
    )
    (decoders): ModuleList()
    )
