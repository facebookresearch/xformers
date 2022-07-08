Hierarchical Transformers
=========================

The original Transformer proposal processes ("transforms") sequences of tokens, across possibly many layers. Crucially, the number of tokens is unchanged cross the depth of the model, and this prove to be really efficient in many domains.

It seems that some domains could however benefit from an architecture more typical from CNN, where there's a tradeoff across the depth of the model in between the spatial extent (ie: number of tokens) and their expressiveness (ie: the model or embedding dimension). These architectures are handled in xformers, through the "patch_embedding" element, which translates the sequence of tokens from one layer to another.

A small helper is provided to make it easier to generate matching configurations, as follows. We present in this example a truncated version of a small Metaformer_.

.. _Metaformer: https://arxiv.org/abs/2111.11418v1

.. code-block:: python

    from xformers.factory import xFormer, xFormerConfig
    from xformers.helpers.hierarchical_configs import (
        BasicLayerConfig,
        get_hierarchical_configuration,
    )


    base_hierarchical_configs = [
        BasicLayerConfig(
            embedding=64,   # the dimensions just have to match along the layers
            attention_mechanism="scaled_dot_product",   # anything you like
            patch_size=7,
            stride=4,
            padding=2,
            seq_len=image_size * image_size // 16,
            feedforward="MLP",
        ),
        BasicLayerConfig(
            embedding=128,
            attention_mechanism="scaled_dot_product",
            patch_size=3,
            stride=2,
            padding=1,
            seq_len=image_size * image_size // 64,
            feedforward="MLP",
        ),
        BasicLayerConfig(
            embedding=320,
            attention_mechanism="scaled_dot_product",
            patch_size=3,
            stride=2,
            padding=1,
            seq_len=image_size * image_size // 256,
            feedforward="MLP",
        ),
    ]

    # Fill in the gaps in the config
    xformer_config = get_hierarchical_configuration(
        base_hierarchical_configs,
        residual_norm_style="pre",
        use_rotary_embeddings=False,
        mlp_multiplier=4,
        dim_head=32,
    )
    config = xFormerConfig(xformer_config)
