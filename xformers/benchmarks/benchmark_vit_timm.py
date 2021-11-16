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
from enum import Enum

import pytorch_lightning as pl
import torch
from pl_bolts.datamodules import CIFAR10DataModule
from timm.models.vision_transformer import VisionTransformer as TimmVisionTransformer
from torch import nn
from torchmetrics import Accuracy
from torchvision import transforms

from xformers.factory import xFormer, xFormerConfig


class Classifier(str, Enum):
    GAP = "gap"
    TOKEN = "token"


class VisionTransformer(pl.LightningModule):
    def __init__(
        self,
        steps,
        learning_rate=1e-2,
        weight_decay=0.0001,
        image_size=32,
        num_classes=10,
        patch_size=4,
        dim=256,
        n_layer=12,
        n_head=8,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        mlp_pdrop=0.1,
        attention="scaled_dot_product",
        hidden_layer_multiplier=4,
        linear_warmup_ratio=0.05,
        classifier: Classifier = Classifier.GAP,
        use_timm: bool = False,
    ):

        super().__init__()

        # all the inputs are saved under self.hparams (hyperparams)
        self.save_hyperparameters()

        assert image_size % patch_size == 0

        num_patches = (image_size // patch_size) ** 2

        if use_timm:
            self.transformer = TimmVisionTransformer(
                img_size=image_size,
                patch_size=patch_size,
                in_chans=3,
                num_classes=num_classes,
                embed_dim=dim,
                depth=n_layer,
                num_heads=n_head,
                mlp_ratio=hidden_layer_multiplier,
                qkv_bias=True,
                distilled=False,
                drop_rate=resid_pdrop,
                attn_drop_rate=attn_pdrop,
                drop_path_rate=0,
                # embed_layer (nn.Module): patch embedding layer
                # norm_layer: (nn.Module): normalization layer
                # weight_init: (str): weight init scheme
            )

        else:
            # A list of the encoder or decoder blocks which constitute the Transformer.
            xformer_config = [
                {
                    "block_config": {
                        "block_type": "encoder",
                        "num_layers": n_layer,
                        "dim_model": dim,
                        "seq_len": num_patches,
                        "layer_norm_style": "pre",
                        "multi_head_config": {
                            "num_heads": n_head,
                            "residual_dropout": resid_pdrop,
                            "attention": {
                                "name": attention,
                                "dropout": attn_pdrop,
                                "causal": False,
                            },
                        },
                        "feedforward_config": {
                            "name": "FusedMLP",
                            "dropout": mlp_pdrop,
                            "activation": "gelu",
                            "hidden_layer_multiplier": hidden_layer_multiplier,
                        },
                    }
                }
            ]

            config = xFormerConfig(xformer_config)
            self.transformer = xFormer.from_config(config)

            # init positional embedding with 0.02 from BERT
            self.pos_emb = nn.Parameter(
                torch.randn(1, num_patches + (classifier == Classifier.TOKEN), dim)
                * 0.02
            )
            self.patch_emb = nn.Conv2d(
                3, dim, kernel_size=patch_size, stride=patch_size
            )

            if classifier == Classifier.TOKEN:
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
            lr=self.hparams.learning_rate,
            momentum=0.9,
            weight_decay=self.hparams.weight_decay,
        )

        warmup_steps = int(self.hparams.linear_warmup_ratio * self.hparams.steps)

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                self.linear_warmup_cosine_decay(warmup_steps, self.hparams.steps),
            ),
            "interval": "step",
        }

        return [optimizer], [scheduler]

    def forward(self, x):
        batch, *_ = x.shape  # BCHW

        if self.hparams.use_timm:
            x = self.transformer(x)
        else:
            x = self.patch_emb(x)

            # flatten patches into sequence
            x = x.flatten(2, 3).transpose(1, 2).contiguous()  # B HW C

            if self.hparams.classifier == Classifier.TOKEN:
                # prepend classification token
                clf_token = (
                    torch.ones(1, batch, self.hparams.dim, device=x.device)
                    * self.clf_token
                )
                x = torch.cat([clf_token, x[:-1, :, :]], axis=0)

            # add position embedding
            x += self.pos_emb.expand_as(x)

            x = self.transformer(x)
            x = self.ln(x)

            if self.hparams.classifier == Classifier.TOKEN:
                x = x[:, 0]
            elif self.hparams.classifier == Classifier.GAP:
                x = x.mean(dim=1)  # mean over sequence len

            x = self.head(x)
        return x

    def training_step(self, batch, _):
        x, y = batch
        y_hat = self(x)

        loss = self.criterion(y_hat, y)

        self.logger.log_metrics(
            {
                "train_loss": loss.mean(),
                "learning_rate": self.lr_schedulers().get_last_lr()[0],
            },
            step=trainer.global_step,
        )

        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = self.val_accuracy(y_hat, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")


if __name__ == "__main__":
    pl.seed_everything(42)
    BATCH_SIZE = 512
    MAX_EPOCHS = 10
    NUM_WORKERS = 4
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

    # We'll use a datamodule here, which already handles dataset/dataloader/sampler
    # See https://pytorchlightning.github.io/lightning-tutorials/notebooks/lightning_examples/cifar10-baseline.html
    # for a full tutorial
    dm = CIFAR10DataModule(
        data_dir="data",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        val_transforms=test_transforms,
    )

    image_size = dm.size(-1)  # 32 for CIFAR
    num_classes = dm.num_classes  # 10 for CIFAR

    # compute total number of steps
    batch_size = BATCH_SIZE * GPUS
    steps = dm.num_samples // batch_size * MAX_EPOCHS
    lm = VisionTransformer(
        steps=steps,
        image_size=image_size,
        num_classes=num_classes,
        attention="scaled_dot_product",
        use_timm=False,
    )
    trainer = pl.Trainer(
        gpus=GPUS, max_epochs=MAX_EPOCHS, detect_anomaly=True, precision=16
    )
    trainer.fit(lm, dm)

    # check the training
    trainer.test(lm, datamodule=dm)
