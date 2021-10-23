I'm used to PyTorch Transformer Encoder, do you have an equivalent?
===================================================================

PyTorch already exposes a couple of pure Transformer blocks,
for instance TransformerEncoder and TransformerEncoderLayer_.

.. _TransformerEncoderLayer: https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html?highlight=encoder#torch.nn.TransformerEncoderLayer

Their interfaces are:

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

    Transformer(
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation='relu',
        custom_encoder=None,
        custom_decoder=None,
        layer_norm_eps=1e-05,
        batch_first=False,
        device=None,
        dtype=None):
        .


While xFormers doesn't have the exact same interfaces, it has something fairly close
through the `model_factory`.

The equivalent with xFormers would look like the following.
You can think of it as a declaration of the sequence of blocks that you would like instantiated.

.. code-block:: python

    from xformers.factory.model_factory import xFormer, xFormerConfig

    my_config =  [
        # A list of the encoder or decoder blocks which constitute the Transformer.
        # Note that a sequence of different encoder blocks can be used, same for decoders
        {
            "reversible": False,  # Optionally make these layers reversible, to save memory
            "block_config": {
                "block_type": "encoder",
                "num_layers": 3,  # Optional, this means that this config will repeat N times
                "dim_model": 384,
                "layer_norm_style": "pre",  # Optional, pre/post
                "position_encoding_config": {
                    "name": "vocab",  # whatever position encodinhg makes sense
                    "dim_model": 384,
                    "seq_len": 1024,
                    "vocab_size": 64,
                },
                "multi_head_config": {
                    "num_heads": 4,
                    "dim_model": 384,
                    "residual_dropout": 0,
                    "attention": {
                        "name": "linformer", # whatever attention mechanism
                        "dropout": 0,
                        "causal": False,
                        "seq_len": 512,
                    },
                },
                "feedforward_config": {
                    "name": "MLP",
                    "dim_model": 384,
                    "dropout": 0,
                    "activation": "relu",
                    "hidden_layer_multiplier": 4,
                },
            }
        }
    ]

    config = xFormerConfig(**my_config)  # This part of xFormers is entirely type checked and needs a config object, could be changed in the future
    model = xFormer.from_config(config).to(device)


Note that this exposes a couple more knobs than the PyTorch Transformer interface,
but in turn is probably a little more flexible.
There are a couple of repeated settings here (dimensions mostly),
this is taken care of in the LRA benchmarking config.
