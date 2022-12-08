# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


# CREDITS:
# inspired by
# https://github.com/nateraw/lightning-vision-transformer
# which in turn references https://github.com/lucidrains/vit-pytorch

# Orignal author: Sean Naren

import argparse
import math
from enum import Enum
from typing import Optional

import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.profiler import AdvancedProfiler, PyTorchProfiler
from timm.models.vision_transformer import VisionTransformer as TimmVisionTransformer
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms

from xformers.factory import xFormer, xFormerConfig, xFormerWeightInit


class Classifier(str, Enum):
    GAP = "gap"
    TOKEN = "token"


class FakeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        pic_size: int = 224,
        num_samples: int = 1000,
        num_classes=10,
        num_workers: int = 2,
        pin_memory: bool = True,
    ):
        super().__init__()

        self.transform = torchvision.transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        self.dataset = torchvision.datasets.FakeData(
            size=num_samples,
            image_size=(3, pic_size, pic_size),
            num_classes=num_classes,
            transform=self.transform,
        )
        self.num_samples = num_samples
        self.pic_size = pic_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            train_samples = int(self.num_samples * 0.9)
            val_samples = self.num_samples - train_samples

            self.data_train, self.data_val = random_split(
                self.dataset, [train_samples, val_samples]
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.data_test = torchvision.datasets.FakeData(
                size=int(self.num_samples * 0.1),
                image_size=(3, self.pic_size, self.pic_size),
                num_classes=self.num_classes,
                transform=self.transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=32,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=32,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=32,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


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
                    "block_type": "encoder",
                    "num_layers": n_layer,
                    "dim_model": dim,
                    "residual_norm_style": "pre",
                    "multi_head_config": {
                        "num_heads": n_head,
                        "residual_dropout": resid_pdrop,
                        "attention": {
                            "name": attention,
                            "dropout": attn_pdrop,
                            "causal": False,
                            "seq_len": num_patches,
                        },
                    },
                    "feedforward_config": {
                        "name": "FusedMLP",
                        "dropout": mlp_pdrop,
                        "activation": "gelu",
                        "hidden_layer_multiplier": hidden_layer_multiplier,
                    },
                }
            ]

            config = xFormerConfig(xformer_config, weight_init=xFormerWeightInit.Timm)
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
        # For torchmetrics > 0.11:
        # self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
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
        pass


if __name__ == "__main__":
    # Get the user requests
    parser = argparse.ArgumentParser("Benchmark xformers and timm ViTs")
    parser.add_argument(
        "-timm",
        "--timm",
        help="Use the ViT implementation from Timm",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "-profile",
        "--profile",
        help="Pofile the runtime and memory",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "-chrome_trace",
        "--chrome_trace",
        help="Pofile the runtime and memory using a chrome trace",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "-samples",
        "--samples",
        help="Number of random samples to generate",
        type=int,
        default=128,
    )

    args = parser.parse_args()

    pl.seed_everything(42)
    BATCH_SIZE = 16
    MAX_EPOCHS = 1
    NUM_WORKERS = 4
    GPUS = 1
    IMG_SIZE = 224
    NUM_SAMPLES = args.samples
    NUM_CLASSES = 10
    N_LAYERS = 4

    # We'll use a dummy datamodule here, which already handles dataset/dataloader/sampler
    dm = FakeDataModule(
        pic_size=IMG_SIZE,
        num_samples=NUM_SAMPLES,
        num_classes=NUM_CLASSES,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # compute total number of steps
    batch_size = BATCH_SIZE * GPUS
    steps = dm.num_samples // batch_size * MAX_EPOCHS
    lm = VisionTransformer(
        steps=steps,
        image_size=IMG_SIZE,
        num_classes=NUM_CLASSES,
        attention="scaled_dot_product",
        use_timm=args.timm,
        n_layer=N_LAYERS,
        patch_size=16,
    )
    profiler = (
        PyTorchProfiler(
            filename="timm_trace" if args.timm else "xFormers_trace",
            export_to_chrome=True,
            record_functions={"training_step_and_backward"},
        )
        if args.profile and args.chrome_trace
        else AdvancedProfiler(
            filename="timm_profile" if args.timm else "xFormers_profile",
        )
        if args.profile
        else None
    )

    trainer = pl.Trainer(
        gpus=GPUS,
        max_epochs=MAX_EPOCHS,
        detect_anomaly=True,
        precision=16,
        profiler=profiler,  # type: ignore
    )
    trainer.fit(lm, dm)

    # check the training
    trainer.test(lm, datamodule=dm)
