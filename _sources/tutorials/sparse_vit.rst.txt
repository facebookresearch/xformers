Replace all attentions from an existing ViT model with a sparse equivalent?
===========================================================================

Let's say you're used to working with a given Transformer based model, and want to experiment with one of the attention mechanisms supported by xFormers.

The following example shows how to do that in a particular example (reusing a reference ViT from pytorch-image-models_), but some aspects will translate just as well
when considering other model sources. In any case, please check the notebooks in the repository_ for a more exhaustive take.


.. code-block:: python

    import timm
    from timm.models.vision_transformer import VisionTransformer
    from xformers.components.attention import ScaledDotProduct
    from xformers.helpers.timm_sparse_attention import TimmSparseAttention
    img_size = 224
    patch_size = 16

    # Get a reference ViT model
    model = VisionTransformer(img_size=img_size, patch_size=patch_size,
                                embed_dim=96, depth=8, num_heads=8, mlp_ratio=3.,
                                qkv_bias=False, norm_layer=nn.LayerNorm).cuda()


    # Define the mask that we want to use
    # We suppose in this snipper that you have a precise mask in mind already
    # but several helpers and examples are proposed in  `xformers.components.attention.attention_patterns`
    my_fancy_mask : torch.Tensor  # This would be for you to define

    # Define a recursive monkey patching function
    def replace_attn_with_xformers_one(module, att_mask):
        module_output = module
        if isinstance(module, timm.models.vision_transformer.Attention):
            qkv = module.qkv
            dim = qkv.weight.shape[1] * module.num_heads
            # Extra parameters can be exposed in TimmSparseAttention, this is a minimal example
            module_output = TimmSparseAttention(dim, module.num_heads, attn_mask=att_mask)
        for name, child in module.named_children():
            module_output.add_module(name, replace_attn_with_xformers_one(child, att_mask))
        del module

        return module_output

    # Now we can just patch our reference model, and get a sparse-aware variation
    model = replace_attn_with_xformers_one(model, my_fancy_mask)

Note that in practice exchanging all the attentions with a sparse alternative may not be a good idea, as the attentions closer to the output are not typically exhibiting a clear sparsity pattern. You can alter `replace_attn_with_xformers_one` above, or replace manually the attentions which would like to sparsify, but not all


.. _pytorch-image-models: https://github.com/rwightman/pytorch-image-models
.. _repository: https://github.com/facebookresearch/xformers
