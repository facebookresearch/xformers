# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import pytorch_lightning as pl
import torch
from pl_bolts.datamodules import CIFAR10DataModule
from torch import nn
from torchmetrics import Accuracy

from examples.cifar_ViT import Classifier, VisionTransformer
from xformers.factory import xFormer, xFormerConfig
from xformers.helpers.hierarchical_configs import (
    BasicLayerConfig,
    get_hierarchical_configuration,
)

# This is very close to the cifarViT example, and reuses a lot of the training code, only the model part is different.
# There are many ways one can use xformers to write down a MetaFormer, for instance by
# picking up the parts from `xformers.components` and implementing the model explicitly,
# or by patching another existing ViT-like implementation.

# This example takes another approach, as we define the whole model configuration in one go (dict structure)
# and then use the xformers factory to generate the model. This obfuscates a lot of the model building
# (though you can inspect the resulting implementation), but makes it trivial to do some hyperparameter search


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
        feedforward="MLP",
        residual_norm_style="pre",
        use_rotary_embeddings=True,
        linear_warmup_ratio=0.1,
        classifier=Classifier.GAP,
    ):

        super(VisionTransformer, self).__init__()

        # all the inputs are saved under self.hparams (hyperparams)
        self.save_hyperparameters()

        # Generate the skeleton of our hierarchical Transformer
        # - This is a small poolformer configuration, adapted to the small CIFAR10 pictures (32x32)
        # - Please note that this does not match the L1 configuration in the paper, as this would correspond to repeated
        #   layers. CIFAR pictures are too small for this config to be directly meaningful (although that would run)
        # - Any other related config would work, and the attention mechanisms don't have to be the same across layers
        base_hierarchical_configs = [
            BasicLayerConfig(
                embedding=64,
                attention_mechanism=attention,
                patch_size=3,
                stride=2,
                padding=1,
                seq_len=image_size * image_size // 4,
                feedforward=feedforward,
                repeat_layer=1,
            ),
            BasicLayerConfig(
                embedding=128,
                attention_mechanism=attention,
                patch_size=3,
                stride=2,
                padding=1,
                seq_len=image_size * image_size // 16,
                feedforward=feedforward,
                repeat_layer=1,
            ),
            BasicLayerConfig(
                embedding=320,
                attention_mechanism=attention,
                patch_size=3,
                stride=2,
                padding=1,
                seq_len=image_size * image_size // 64,
                feedforward=feedforward,
                repeat_layer=1,
            ),
            BasicLayerConfig(
                embedding=512,
                attention_mechanism=attention,
                patch_size=3,
                stride=2,
                padding=1,
                seq_len=image_size * image_size // 256,
                feedforward=feedforward,
                repeat_layer=1,
            ),
        ]

        # Fill in the gaps in the config
        xformer_config = get_hierarchical_configuration(
            base_hierarchical_configs,
            residual_norm_style=residual_norm_style,
            use_rotary_embeddings=use_rotary_embeddings,
            mlp_multiplier=4,
            dim_head=32,
        )

        # Now instantiate the metaformer trunk
        config = xFormerConfig(xformer_config)
        config.weight_init = "moco"

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
    REF_BATCH = 768
    BATCH = 256  # lower if not enough GPU memory

    MAX_EPOCHS = 50
    NUM_WORKERS = 4
    GPUS = 1

    torch.cuda.manual_seed_all(42)
    torch.manual_seed(42)

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
    lm = MetaVisionTransformer(
        steps=steps,
        image_size=image_size,
        num_classes=num_classes,
        attention="scaled_dot_product",
        residual_norm_style="pre",
        feedforward="MLP",
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
