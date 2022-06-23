# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path

import pytorch_lightning as pl
import torch
from pl_bolts.datamodules import ImagenetDataModule
from torch import nn
from torchmetrics import Accuracy

from examples.cifar_ViT import Classifier, VisionTransformer
from xformers.components import MultiHeadDispatch
from xformers.components.attention import ScaledDotProduct
from xformers.components.patch_embedding import PatchEmbeddingConfig  # noqa
from xformers.components.patch_embedding import build_patch_embedding  # noqa
from xformers.factory import xFormer, xFormerConfig
from xformers.helpers.hierarchical_configs import (
    BasicLayerConfig,
    get_hierarchical_configuration,
)

# This example is very close to the generic "cifarMetaFormer" example, but this time
# implements a more specific "EfficientFormer" (https://arxiv.org/abs/2206.01191) model


class EfficientFormer(VisionTransformer):
    def __init__(
        self,
        steps,
        learning_rate=1e-2,
        betas=(0.9, 0.99),
        weight_decay=0.03,
        image_size=32,
        num_classes=10,
        dim=384,
        linear_warmup_ratio=0.1,
        classifier=Classifier.GAP,
    ):

        super(VisionTransformer, self).__init__()

        # all the inputs are saved under self.hparams (hyperparams)
        self.save_hyperparameters()

        # Generate the skeleton of our hierarchical Transformer
        # This implements a model close to the L1 suggested in "EfficientFormer" (https://arxiv.org/abs/2206.01191)
        # but a pooling layer has been removed, due to the very small image dimensions in Cifar (32x32)
        base_hierarchical_configs = [
            BasicLayerConfig(
                embedding=48,
                attention_mechanism="pooling",
                patch_size=3,
                stride=2,
                padding=1,
                seq_len=image_size * image_size // 16,
                feedforward="Conv2DFeedforward",
            ),
            BasicLayerConfig(
                embedding=96,
                attention_mechanism="pooling",
                patch_size=3,
                stride=2,
                padding=1,
                seq_len=image_size * image_size // 64,
                feedforward="Conv2DFeedforward",
            ),
            BasicLayerConfig(
                embedding=224,
                attention_mechanism="pooling",
                patch_size=3,
                stride=2,
                padding=1,
                seq_len=image_size * image_size // 256,
                feedforward="Conv2DFeedforward",
            ),
            BasicLayerConfig(
                embedding=448,
                attention_mechanism="pooling",
                patch_size=3,
                stride=2,
                padding=1,
                seq_len=image_size * image_size // 1024,
                feedforward="Conv2DFeedforward",
            ),
        ]

        # Fill in the gaps in the config
        xformer_config = get_hierarchical_configuration(
            base_hierarchical_configs,
            layernorm_style="pre",
            use_rotary_embeddings=False,
            mlp_multiplier=4,
            in_channels=24,  # 24 if L1, there's a stem prior to the trunk
        )

        # Now instantiate the EfficientFormer trunk
        config = xFormerConfig(xformer_config)
        config.weight_init = "moco"

        self.trunk = xFormer.from_config(config)

        # This model requires a pre-stem (a conv prior to going through all the layers above)
        self.pre_stem = build_patch_embedding(
            PatchEmbeddingConfig(
                in_channels=3, out_channels=24, kernel_size=3, stride=2, padding=1
            )
        )

        # This model requires a final Attention step
        self.attention = MultiHeadDispatch(
            dim_model=448, num_heads=8, attention=ScaledDotProduct()
        )

        # The classifier head
        dim = base_hierarchical_configs[-1].embedding
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.val_accuracy = Accuracy()

    def forward(self, x):
        x = x.flatten(-2, -1).transpose(-1, -2)  # BCHW to BSE
        x = self.pre_stem(x)
        x = self.trunk(x)
        x = self.attention(x)
        x = self.ln(x)

        x = x.mean(dim=1)  # mean over sequence len
        x = self.head(x)
        return x


if __name__ == "__main__":
    pl.seed_everything(42)

    # Adjust batch depending on the available memory on your machine.
    # You can also use reversible layers to save memory
    REF_BATCH = 128
    BATCH = 32  # lower if not enough GPU memory

    MAX_EPOCHS = 50
    NUM_WORKERS = 4
    GPUS = 1

    torch.cuda.manual_seed_all(42)
    torch.manual_seed(42)

    # We'll use a datamodule here, which already handles dataset/dataloader/sampler
    # See https://pytorchlightning.github.io/lightning-tutorials/notebooks/lightning_examples/cifar10-baseline.html
    # for a full tutorial
    dm = ImagenetDataModule(
        data_dir=Path.home()
        / Path(
            "Data/ImageNet/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC"
        ),
        batch_size=BATCH,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    image_size = dm.size(-1)  # 32 for CIFAR
    num_classes = dm.num_classes  # 10 for CIFAR

    # compute total number of steps
    batch_size = BATCH * GPUS
    steps = dm.num_samples // REF_BATCH * MAX_EPOCHS
    lm = EfficientFormer(
        steps=steps,
        image_size=image_size,
        num_classes=num_classes,
    )
    trainer = pl.Trainer(
        gpus=GPUS,
        max_epochs=MAX_EPOCHS,
        precision=16,
        accumulate_grad_batches=REF_BATCH // BATCH,
    )
    trainer.fit(lm, dm)

    # check the training
    trainer.test(lm, datamodule=dm)
