# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


# CREDITS: adapted from the Nystromformer repo
# https://github.com/mlpen/Nystromformer

from enum import Enum
from typing import Dict, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn

from xformers.components import build_attention
from xformers.components.multi_head_dispatch import MultiHeadDispatchConfig
from xformers.factory import xFormer, xFormerConfig, xFormerEncoderConfig
from xformers.utils import generate_matching_config

PLOutput = Dict[str, Union[float, torch.Tensor]]


class Pooling(str, Enum):
    MEAN = "mean"
    CLS = "cls"


def pooling(mode: Pooling):
    def pool_cls(inp):
        return inp[:, 0, :]

    def pool_mean(inp):
        return inp.mean(dim=1)

    return {Pooling.MEAN: pool_mean, Pooling.CLS: pool_cls}[mode]


def append_cls(inp, mask, vocab_size):
    batch_size = inp.size(0)
    cls_id = (
        (vocab_size - 1) * torch.ones(batch_size, dtype=torch.long, device=inp.device)
    ).long()
    cls_mask = torch.ones(batch_size, dtype=torch.float, device=mask.device)
    inp = torch.cat([cls_id[:, None], inp[:, :-1]], dim=-1)
    mask = torch.cat([cls_mask[:, None], mask[:, :-1]], dim=-1)
    return inp, mask


def patch_model_config(config, attention_name):
    # Rebuild a specific config out of generic + extra params
    commons = config["common"]
    try:
        extra_attention_settings = config["extra_settings"]["attention"][attention_name]
    except KeyError:
        extra_attention_settings = None

    for bc in config["xformer"]:
        bc["dim_model"] = commons["dim_model"]
        bc["position_encoding_config"].update(commons)
        bc["feedforward_config"].update(commons)
        bc["multi_head_config"].update(commons)
        bc["multi_head_config"]["attention"].update(commons)
        bc["multi_head_config"]["attention"]["name"] = attention_name
        bc["multi_head_config"]["attention"]["dim_head"] = (
            commons["dim_model"] / commons["num_heads"]
        )
        if extra_attention_settings is not None:
            bc["multi_head_config"]["attention"].update(extra_attention_settings)

        bc["multi_head_config"] = generate_matching_config(
            bc["multi_head_config"], MultiHeadDispatchConfig
        )
        bc["multi_head_config"].attention = build_attention(
            bc["multi_head_config"].attention
        )
        bc = generate_matching_config(bc, xFormerEncoderConfig)

    return config


class SCHead(nn.Module):
    def __init__(self, config, dim_embedding, dim_mlp):
        super().__init__()
        self.pooling = pooling(Pooling(config["pooling_mode"]))

        self.mlpblock = nn.Sequential(
            nn.Linear(dim_embedding, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, config["common"]["num_classes"]),
        )

    def forward(self, inp: torch.Tensor):
        seq_score = self.mlpblock(self.pooling(inp))
        return seq_score


class SCHeadDual(nn.Module):
    def __init__(self, config, dim_embedding, dim_mlp):
        super().__init__()
        self.pooling = pooling(Pooling(config["pooling_mode"]))

        self.mlpblock = nn.Sequential(
            nn.Linear(
                dim_embedding * 4,
                dim_mlp,
            ),
            nn.ReLU(),
            nn.Linear(dim_mlp, config["common"]["num_classes"]),
        )

    def forward(self, inp_0: torch.Tensor, inp_1: torch.Tensor):
        X_0 = self.pooling(inp_0)
        X_1 = self.pooling(inp_1)
        seq_score = self.mlpblock(torch.cat([X_0, X_1, X_0 * X_1, X_0 - X_1], dim=-1))
        return seq_score


class ModelTrunk(pl.LightningModule):
    def __init__(self, config, model_name):
        super().__init__()

        config_model = config["model"]
        self.config_training = config["training"]

        self.enable_amp = config["training"]["mixed_precision"]
        self.pooling_mode = Pooling(config_model["pooling_mode"])
        self.vocab_size = config_model["common"]["vocab_size"]

        # Rebuild a specific config out of generic + extra params
        self.config_model = patch_model_config(config_model, model_name)
        self.model = xFormer.from_config(xFormerConfig(config_model["xformer"]))
        self.norm = nn.LayerNorm(self.config_model["common"]["dim_model"])

        ff_config = self.config_model["xformer"][0]["feedforward_config"]
        self.dim_mlp = (
            self.config_model["common"]["dim_model"]
            * ff_config["hidden_layer_multiplier"]
        )

    def training_step(  # type: ignore
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> PLOutput:
        outputs = self(**batch)
        self.logger.log_metrics({f"train_{k}": v for k, v in outputs.items()})  # type: ignore
        self.log("train_accu", outputs["accu"], sync_dist=True)
        return outputs

    def training_epoch_end(self, outputs):
        logs = self.eval_epoch_end(outputs)
        self.log("train_accu_mean", logs["accu"], sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config_training["learning_rate"],
            betas=(0.9, 0.999),
            eps=1e-6,
            weight_decay=self.config_training["weight_decay"],
        )

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=self.config_training["learning_rate"],
            pct_start=self.config_training["warmup"]
            / self.config_training["num_train_steps"],
            anneal_strategy=self.config_training["lr_decay"],
            total_steps=self.config_training["num_train_steps"],
        )

        return [optimizer], [lr_scheduler]

    def eval_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> PLOutput:
        outputs = self(**batch)
        return outputs

    def eval_epoch_end(self, outputs, prefix: str = "train"):
        logs = {}
        counts = torch.tensor([x["count"] for x in outputs]).float()
        logs["count"] = counts.sum()
        for k in ("accu", "loss"):
            logs[k] = (torch.tensor([x[k] for x in outputs]) * counts).sum() / logs[
                "count"
            ]
            self.log(f"{prefix}_{k}_mean", logs[k], sync_dist=True)
        return logs

    def validation_step(  # type: ignore
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> PLOutput:
        outputs = self.eval_step(batch, batch_idx)
        self.logger.log_metrics({f"val_{k}": v for k, v in outputs.items()})  # type: ignore
        self.log("val_accu", outputs["accu"], sync_dist=True, prog_bar=True)
        return outputs

    def validation_epoch_end(self, outputs):
        self.eval_epoch_end(outputs, prefix="val")

    def test_step(  # type: ignore
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> PLOutput:
        return self.eval_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        self.eval_epoch_end(outputs, prefix="test")


class ModelForSC(ModelTrunk):
    def __init__(self, config, model_name):
        # Setup trunk
        super().__init__(config, model_name)

        self.seq_classifer = SCHead(
            self.config_model,
            dim_embedding=self.config_model["common"]["dim_model"],
            dim_mlp=self.dim_mlp,
        )

    def forward(  # type: ignore
        self, input_ids_0: torch.Tensor, mask_0: torch.Tensor, label: torch.Tensor
    ):

        if self.pooling_mode == Pooling.CLS:
            input_ids_0, mask_0 = append_cls(input_ids_0, mask_0, self.vocab_size)

        token_out = self.norm(
            self.model(input_ids_0, encoder_input_mask=mask_0)
        ) * mask_0.unsqueeze(-1)

        seq_scores = self.seq_classifer(token_out)

        seq_loss = torch.nn.CrossEntropyLoss(reduction="none")(seq_scores, label)
        seq_accu = (seq_scores.argmax(dim=-1) == label).to(torch.float32)
        outputs = {
            "loss": seq_loss.mean(),
            "accu": seq_accu.mean(),
            "count": label.size(0),
        }

        return outputs


class ModelForSCDual(ModelTrunk):
    def __init__(self, config, model_name):
        # Setup trunk
        super().__init__(config, model_name)

        self.seq_classifer = SCHeadDual(
            self.config_model,
            dim_embedding=self.config_model["common"]["dim_model"],
            dim_mlp=self.dim_mlp,
        )

    def forward(  # type: ignore
        self,
        input_ids_0: torch.Tensor,
        input_ids_1: torch.Tensor,
        mask_0: torch.Tensor,
        mask_1: torch.Tensor,
        label: torch.Tensor,
    ):

        mask_0, mask_1 = mask_0.long(), mask_1.long()

        if self.pooling_mode == Pooling.CLS:
            input_ids_0, mask_0 = append_cls(input_ids_0, mask_0, self.vocab_size)
            input_ids_1, mask_1 = append_cls(input_ids_1, mask_1, self.vocab_size)

        # Concatenate the two inputs into one batch
        input_ids = torch.cat([input_ids_0, input_ids_1], dim=0)
        masks = torch.cat([mask_0, mask_1], dim=0)

        tokens_out = self.norm(
            self.model(input_ids, encoder_input_mask=masks)
        ) * masks.unsqueeze(-1)

        seq_scores = self.seq_classifer(*torch.chunk(tokens_out, 2, dim=0))

        seq_loss = torch.nn.CrossEntropyLoss(reduction="none")(seq_scores, label)
        seq_accu = (seq_scores.argmax(dim=-1) == label).to(torch.float32)
        outputs = {
            "loss": seq_loss.mean(),
            "accu": seq_accu.mean(),
            "count": label.size(0),
        }

        return outputs
