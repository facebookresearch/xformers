# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import pytorch_lightning as pl
import torch
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from torch import nn
from torchmetrics import Accuracy
from torchvision import transforms

from examples.microViT import Classifier, VisionTransformer
from xformers.factory import xFormer, xFormerConfig
from xformers.helpers.hierarchical_configs import (
    BasicLayerConfig,
    get_hierarchical_configuration,
)


class MetaVisionTransformer(VisionTransformer):
    def __init__(
        self,
        steps,
        learning_rate=5e-3,
        betas=(0.9, 0.99),
        weight_decay=0.03,
        image_size=32,
        num_classes=10,
        dim=384,
        attention="scaled_dot_product",
        layer_norm_style="pre",
        use_rotary_embeddings=True,
        linear_warmup_ratio=0.1,
        classifier=Classifier.GAP,
    ):

        super(VisionTransformer, self).__init__()

        # all the inputs are saved under self.hparams (hyperparams)
        self.save_hyperparameters()

        # Generate the skeleton of our hierarchical Transformer

        # This is a small poolformer configuration, adapted to the small CIFAR10 pictures (32x32)
        # Any other related config would work,
        # and the attention mechanisms don't have to be the same across layers
        base_hierarchical_configs = [
            BasicLayerConfig(
                embedding=64,
                attention_mechanism=attention,
                patch_size=3,
                stride=2,
                padding=1,
                seq_len=image_size * image_size // 4,
            ),
            BasicLayerConfig(
                embedding=128,
                attention_mechanism=attention,
                patch_size=3,
                stride=2,
                padding=1,
                seq_len=image_size * image_size // 16,
            ),
            BasicLayerConfig(
                embedding=320,
                attention_mechanism=attention,
                patch_size=3,
                stride=2,
                padding=1,
                seq_len=image_size * image_size // 64,
            ),
            BasicLayerConfig(
                embedding=512,
                attention_mechanism=attention,
                patch_size=3,
                stride=2,
                padding=1,
                seq_len=image_size * image_size // 256,
            ),
        ]

        # Fill in the gaps in the config
        xformer_config = get_hierarchical_configuration(
            base_hierarchical_configs,
            layernorm_style=layer_norm_style,
            use_rotary_embeddings=use_rotary_embeddings,
            mlp_multiplier=4,
            dim_head=32,
        )

        # Now instantiate the metaformer trunk
        config = xFormerConfig(xformer_config)
        print(config)
        self.trunk = xFormer.from_config(config)
        print(self.trunk)

        # The classifier head
        dim = base_hierarchical_configs[-1].embedding
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.val_accuracy = Accuracy()

    def forward(self, x):
        x = self.trunk(x)
        x = self.ln(x)

        x = x.mean(dim=1)  # mean over sequence len
        x = self.head(x)
        return x


if __name__ == "__main__":
    pl.seed_everything(42)

    # Adjust batch depending on the available memory on your machine.
    # You can also use reversible layers to save memory
    REF_BATCH = 512
    BATCH = 512  # lower if not enough GPU memory

    MAX_EPOCHS = 50
    NUM_WORKERS = 4
    GPUS = 1

    torch.cuda.manual_seed_all(42)
    torch.manual_seed(42)

    train_transforms = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )

    # We'll use a datamodule here, which already handles dataset/dataloader/sampler
    # See https://pytorchlightning.github.io/lightning-tutorials/notebooks/lightning_examples/cifar10-baseline.html
    # for a full tutorial
    dm = CIFAR10DataModule(
        data_dir="data",
        batch_size=BATCH,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    dm.train_transforms = train_transforms
    dm.test_transforms = test_transforms
    dm.val_transforms = test_transforms

    image_size = dm.size(-1)  # 32 for CIFAR
    num_classes = dm.num_classes  # 10 for CIFAR

    # compute total number of steps
    batch_size = BATCH * GPUS
    steps = dm.num_samples // REF_BATCH * MAX_EPOCHS
    lm = MetaVisionTransformer(
        steps=steps,
        image_size=image_size,
        num_classes=num_classes,
        attention="scaled_dot_product",
        layer_norm_style="pre",
        use_rotary_embeddings=True,
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
