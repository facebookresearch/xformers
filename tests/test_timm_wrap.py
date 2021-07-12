import pytest
import timm
import torch
from timm.models.vision_transformer import VisionTransformer

from xformers.helpers.timm_sparse_attention import TimmSparseAttention

_device_list = ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"]


@pytest.mark.parametrize("device", _device_list)
def test_timm_wrapper(device):
    img_size = 224
    patch_size = 16
    batch = 8

    # Instantiate the reference model
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=96,
        depth=8,
        num_heads=8,
        mlp_ratio=3.0,
        qkv_bias=False,
        norm_layer=torch.nn.LayerNorm,
    ).to(device)

    # Monkey patch all attentions to test the sparse-aware wrap
    def replace_attn_with_xformers_one(module, att_mask):
        module_output = module
        if isinstance(module, timm.models.vision_transformer.Attention):
            qkv = module.qkv
            dim = qkv.weight.shape[1] * module.num_heads
            module_output = TimmSparseAttention(
                dim, module.num_heads, attn_mask=att_mask
            )
        for name, child in module.named_children():
            module_output.add_module(
                name, replace_attn_with_xformers_one(child, att_mask)
            )
        del module

        return module_output

    H, W = img_size // patch_size, img_size // patch_size
    mask = (torch.rand((H * W + 1, H * W + 1), device=device) > 0.5).bool()
    model = replace_attn_with_xformers_one(model, att_mask=mask)

    # Check that we can throw a couple of random pictures at it
    inputs = torch.rand((batch, 3, img_size, img_size), device=device)
    _ = model(inputs)
