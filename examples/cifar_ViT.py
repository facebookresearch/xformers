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
from torch import nn
from torchmetrics import Accuracy

from xformers.factory import xFormer, xFormerConfig


class Classifier(str, Enum):
    GAP = "gap"
    TOKEN = "token"


class VisionTransformer(pl.LightningModule):
    def __init__(
        self,
        steps,
        learning_rate=5e-4,
        betas=(0.9, 0.99),
        weight_decay=0.03,
        image_size=32,
        num_classes=10,
        patch_size=2,
        dim=384,
        n_layer=6,
        n_head=6,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
        mlp_pdrop=0.0,
        attention="scaled_dot_product",
        residual_norm_style="pre",
        hidden_layer_multiplier=4,
        use_rotary_embeddings=True,
        linear_warmup_ratio=0.1,
        classifier: Classifier = Classifier.TOKEN,
    ):

        super().__init__()

        # all the inputs are saved under self.hparams (hyperparams)
        self.save_hyperparameters()

        assert image_size % patch_size == 0

        num_patches = (image_size // patch_size) ** 2

        # A list of the encoder or decoder blocks which constitute the Transformer.
        xformer_config = [
            {
                "block_type": "encoder",
                "num_layers": n_layer,
                "dim_model": dim,
                "residual_norm_style": residual_norm_style,
                "multi_head_config": {
                    "num_heads": n_head,
                    "residual_dropout": resid_pdrop,
                    "use_rotary_embeddings": use_rotary_embeddings,
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
                "position_encoding_config": {
                    "name": "learnable",
                    "seq_len": num_patches,
                    "dim_model": dim,
                    "add_class_token": classifier == Classifier.TOKEN,
                },
                "patch_embedding_config": {
                    "in_channels": 3,
                    "out_channels": dim,
                    "kernel_size": patch_size,
                    "stride": patch_size,
                },
            }
        ]

        # The ViT trunk
        config = xFormerConfig(xformer_config)
        self.vit = xFormer.from_config(config)
        print(self.vit)

        # The classifier head
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
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=self.hparams.betas,
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
        x = self.vit(x)
        x = self.ln(x)

        if self.hparams.classifier == Classifier.TOKEN:
            x = x[:, 0]  # only consider the token, we're classifying anyway
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
            step=self.global_step,
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

    def validation_step(self, batch, _):
        self.evaluate(batch, "val")

    def test_step(self, batch, _):
        self.evaluate(batch, "test")


if __name__ == "__main__":
    pl.seed_everything(42)

    # Adjust batch depending on the available memory on your machine.
    # You can also use reversible layers to save memory
    REF_BATCH = 512
    BATCH = 128

    MAX_EPOCHS = 30
    NUM_WORKERS = 4
    GPUS = 1

    # We'll use a datamodule here, which already handles dataset/dataloader/sampler
    # - See https://pytorchlightning.github.io/lightning-tutorials/notebooks/lightning_examples/cifar10-baseline.html
    # for a full tutorial
    # - Please note that default transforms are being used
    dm = CIFAR10DataModule(
        data_dir="data",
        batch_size=BATCH,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    image_size = dm.size(-1)  # 32 for CIFAR
    num_classes = dm.num_classes  # 10 for CIFAR

    # compute total number of steps
    batch_size = BATCH * GPUS
    steps = dm.num_samples // REF_BATCH * MAX_EPOCHS
    lm = VisionTransformer(
        steps=steps,
        image_size=image_size,
        num_classes=num_classes,
        attention="scaled_dot_product",
        classifier=Classifier.TOKEN,
        residual_norm_style="pre",
        use_rotary_embeddings=True,
    )
    trainer = pl.Trainer(
        gpus=GPUS,
        max_epochs=MAX_EPOCHS,
        detect_anomaly=False,
        precision=16,
        accumulate_grad_batches=REF_BATCH // BATCH,
    )
    trainer.fit(lm, dm)

    # check the training
    trainer.test(lm, datamodule=dm)
