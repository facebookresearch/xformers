# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


# CREDITS:
# inspired by
# https://github.com/nateraw/lightning-vision-transformer
# which in turn references https://github.com/lucidrains/vit-pytorch

# Orignal author: Sean Naren

import math

import pytorch_lightning as pl
import torch
from einops import rearrange
from pl_bolts.datamodules import CIFAR10DataModule
from torch import nn
from torchmetrics import Accuracy
from torchvision import transforms

from xformers.factory import xFormer, xFormerConfig


class VisionTransformer(pl.LightningModule):
    def __init__(
        self,
        steps,
        learning_rate=1e-4,
        weight_decay=0.0001,
        image_size=32,
        num_classes=10,
        patch_size=4,
        dim=256,
        n_layer=12,
        n_head=4,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        mlp_pdrop=0.1,
        attention="scaled_dot_product",
        hidden_layer_multiplier=4,
        linear_warmup_ratio=0.05,
        classifier="gap",
    ):

        super().__init__()

        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.steps = steps
        self.linear_warmup_ratio = linear_warmup_ratio

        assert image_size % patch_size == 0

        num_patches = (image_size // patch_size) ** 2
        self.dim = dim
        self.classifier = classifier

        # A list of the encoder or decoder blocks which constitute the Transformer.
        xformer_config = [
            {
                "block_config": {
                    "block_type": "encoder",
                    "num_layers": self.hparams.n_layer,
                    "dim_model": self.hparams.dim,
                    "layer_norm_style": "pre",
                    "multi_head_config": {
                        "num_heads": self.hparams.n_head,
                        "residual_dropout": self.hparams.resid_pdrop,
                        "use_rotary_embeddings": True,
                        "attention": {
                            "name": self.hparams.attention,
                            "dropout": self.hparams.attn_pdrop,
                            "causal": True,
                        },
                    },
                    "feedforward_config": {
                        "name": "MLP",
                        "dropout": self.hparams.mlp_pdrop,
                        "activation": "gelu",
                        "hidden_layer_multiplier": self.hparams.hidden_layer_multiplier,
                    },
                }
            }
        ]

        config = xFormerConfig(xformer_config)
        self.transformer = xFormer.from_config(config)

        # init positional embedding with 0.02 from BERT
        self.pos_emb = nn.Parameter(
            torch.randn(1, num_patches + (classifier == "token"), dim) * 0.02
        )
        self.patch_emb = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)

        if classifier == "token":
            self.clf_token = nn.Parameter(torch.zeros(dim))

        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.val_accuracy = Accuracy()

    @staticmethod
    def linear_warmup_cosine_decay(warmup_steps, total_steps):
        """
        Linear warmup for warmup_steps, with cosine annealing to 0 at total_steps
        """

        def fn(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))

            progress = float(step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return fn

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=self.weight_decay,
        )

        warmup_steps = int(self.linear_warmup_ratio * self.steps)

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer, self.linear_warmup_cosine_decay(warmup_steps, self.steps)
            ),
            "interval": "step",
        }

        return [optimizer], [scheduler]

    def forward(self, x):
        batch, *_ = x.shape
        x = self.patch_emb(x)

        # flatten patches into sequence
        x = rearrange(x, "b c h w -> b (h w) c")

        if self.classifier == "token":
            # prepend classification token
            clf_token = torch.ones(1, batch, self.dim, device=x.device) * self.clf_token
            x = torch.cat([clf_token, x[:-1, :, :]], axis=0)

        # add position embedding
        x += self.pos_emb.expand_as(x)
        x = self.transformer(x)
        x = self.ln(x)

        if self.classifier == "token":
            x = x[:, 0]
        elif self.classifier == "gap":
            x = x.mean(dim=1)  # mean over sequence len

        x = self.head(x)
        return x

    def training_step(self, batch, _):
        x, y = batch
        y_hat = self(x)

        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("valid_acc", self.val_accuracy(y_hat, y))
        self.log("valid_loss", loss)


if __name__ == "__main__":
    pl.seed_everything(42)
    BATCH_SIZE = 256
    LR = 0.01
    PATCH_SIZE = 4
    DIM = 256
    LAYERS = 12
    HEADS = 8
    MAX_EPOCHS = 5
    NUM_WORKERS = 8
    GPUS = 1

    train_transforms = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    dm = CIFAR10DataModule(
        data_dir="data", batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    dm.train_transforms = train_transforms
    dm.test_transforms = test_transforms
    dm.val_transforms = test_transforms
    image_size = dm.size(-1)  # 32 for CIFAR
    num_classes = dm.num_classes  # 10 for CIFAR

    # compute total number of steps
    batch_size = BATCH_SIZE * GPUS
    steps = dm.num_samples // batch_size * MAX_EPOCHS
    lm = VisionTransformer(
        steps=steps,
        learning_rate=LR,
        n_layer=LAYERS,
        n_head=HEADS,
        patch_size=PATCH_SIZE,
        dim=DIM,
    )
    trainer = pl.Trainer(
        gpus=GPUS, max_epochs=MAX_EPOCHS, terminate_on_nan=True, precision=16
    )
    trainer.fit(lm, dm)
