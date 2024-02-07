# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import json
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Dict, Tuple, cast

import pytorch_lightning as pl
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, flop_count_str
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

from xformers.benchmarks.LRA.code.dataset import LRADataset
from xformers.benchmarks.LRA.code.model_wrapper import ModelForSC, ModelForSCDual
from xformers.components.attention import ATTENTION_REGISTRY


class Task(str, Enum):
    Retrieval = "retrieval"
    ListOps = "listops"
    Image = "image"
    PathfinderBaseline = "pathfinder32-curv_baseline"
    PathfinderContour9 = "pathfinder32-curv_contour_length_9"
    PathfinderContour14 = "pathfinder32-curv_contour_length_14"
    Text = "text"


def load_config(path: str) -> Dict:
    with open(Path(path).absolute(), "r") as fileio:
        config = json.load(fileio)

    # Duplicate the pathfinder configs
    config["pathfinder32-curv_baseline"] = config["pathfinder32"]
    config["pathfinder32-curv_contour_length_9"] = config["pathfinder32"]
    config["pathfinder32-curv_contour_length_14"] = config["pathfinder32"]
    return config


def build_model(args: argparse.Namespace, config: Dict) -> nn.Module:
    task = args.task
    attention_name = args.attention

    model = cast(
        pl.LightningModule,
        (
            ModelForSCDual(config[f"{task}"], attention_name)
            if task == Task.Retrieval
            else ModelForSC(config[f"{task}"], attention_name)
        ),
    )

    logging.info(model)
    summary = pl.utilities.model_summary.LayerSummary(model)
    logging.info(f"num_parameter: {summary.num_parameters // 1e3 / 1e3}M")

    with torch.no_grad():
        # Check the flops
        seq_len = config[f"{task}"]["model"]["common"]["seq_len"]
        x = torch.rand(1, seq_len).long()
        mask = torch.rand(1, seq_len).long()
        indices = torch.rand(1, seq_len).long()
        flops = FlopCountAnalysis(model.model, (x, mask, indices))
        logging.info(f"complexity: {round(flops.total()/1e9, 3)} GFlops")
        logging.info(flop_count_str(flops))

    return model


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attention",
        type=str,
        help=f"Attention mechanism to chose, among {list(ATTENTION_REGISTRY.keys())}. \
            A list can be passed to test several mechanisms in sequence",
        dest="attention",
        required=True,
    )
    parser.add_argument(
        "--task",
        type=Task,
        help=f"Task to chose, among {[t.value for t in Task]}.",
        dest="task",
        required=True,
    )
    parser.add_argument(
        "--skip_train",
        type=bool,
        help="Whether to skip training, and test an existing model",
        dest="skip_train",
        default=False,
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the config being used",
        dest="config",
        default="./config.json",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Path to the checkpoint directory",
        dest="checkpoint_dir",
        default=f"/checkpoints/{os.getenv('USER')}/xformers",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path to checkpoint",
    )
    parser.add_argument(
        "--debug",
        help="Make it easier to debug a possible issue",
        dest="debug",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--world_size",
        help="Number of GPUs used",
        dest="world_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--sweep_parameters",
        help="Rewrite some hyperparameters in the config",
        dest="sweep_parameters",
        type=dict,
        default=None,
    )
    return parser


def setup_log(args, attention_name, task) -> Tuple[str, TensorBoardLogger]:
    experiment_name = f"{task}__{attention_name}"
    logger = TensorBoardLogger(
        save_dir=args.checkpoint_dir,
        name="",  # remove lightning_logs subdirectory
        version=experiment_name,
    )
    log_dir = os.path.join(logger._save_dir, experiment_name)
    return log_dir, logger


def rewrite_hyper(config, rewrites):
    def replace(config_dict, k, v):
        if len(k.split(":")) == 1:
            config_dict[k] = v
            return
        first_key = k.split(":")[0]
        assert first_key in config_dict, first_key
        k = k[len(first_key) + 1 :]
        replace(config_dict[first_key], k, v)

    for k, v in rewrites.items():
        replace(config, k, v)
    return config


def build_dataloaders(
    args: argparse.Namespace,
    config_training: Dict,
    num_workers: int = 4,
) -> Dict[str, DataLoader]:
    datasets = {}
    for component in ("train", "dev", "test"):
        datasets[component] = LRADataset(
            file_path=f"datasets/{args.task}.{component}.pickle",
            seq_len=config_training["seq_len"],
        )

    # Gradient accumulation
    accumu_steps = config_training["gradient_accumulation"]
    logging.info(f"accumu_steps={accumu_steps}")

    # Batch size
    per_gpu_batch_size = (
        config_training["batch_size"] // args.world_size // accumu_steps
    )
    logging.warning(
        f"Requested batch size: {config_training['batch_size']}. Given world\
            size and grad accumulation, per-gpu batch is\
            {per_gpu_batch_size}"
    )

    dataloaders = {
        k: DataLoader(
            v,
            batch_size=per_gpu_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )
        for k, v in datasets.items()
    }
    return dataloaders


def get_eval_summary(trainer: pl.Trainer) -> Dict[str, float]:
    eval_summary: Dict[str, float] = {"train_step_idx": trainer.global_step}
    for k, v in trainer.callback_metrics.items():
        eval_summary[k] = v.item()
    return eval_summary


class BasicProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items


def benchmark(args):
    log_dir, logger = setup_log(args, f"{args.attention}", f"{args.task}")
    args.logger = logger

    config = load_config(args.config)

    config_task = config[f"{args.task}"]
    if args.sweep_parameters is not None:
        logging.info("Replacing hyperparameters")
        rewrite_hyper(config_task, args.sweep_parameters)

    config_training = config_task["training"]
    config_training["seq_len"] = config_task["model"]["common"]["seq_len"]
    logging.info(f"Learning rate: {config_training['learning_rate']}")

    pl.seed_everything(config_training.get("seed", 0))
    dataloaders = build_dataloaders(args, config_training)

    model = build_model(args, config)

    progress_bar = BasicProgressBar()
    checkpoint_callback = ModelCheckpoint(
        monitor="val_accu",
        mode="max",
        dirpath=args.checkpoint_dir,
        filename="{epoch}-{val_accu:.2f}",
        every_n_train_steps=config_training["eval_frequency"],
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        strategy=(
            DDPStrategy(find_unused_parameters=args.debug)
            if not args.skip_train
            else None
        ),
        accumulate_grad_batches=config_training["gradient_accumulation"],
        callbacks=[progress_bar, checkpoint_callback],
        detect_anomaly=args.debug,
        deterministic=True,
        gpus=args.world_size,
        limit_val_batches=config_training["num_eval_steps"],
        logger=logger,
        max_steps=config_training["num_train_steps"],
        num_sanity_val_steps=int(not args.skip_train),
        precision=16 if config_training["mixed_precision"] else 32,
        val_check_interval=config_training["eval_frequency"]
        / float(len(dataloaders["train"])),
    )

    if not args.skip_train:
        trainer.fit(
            model,
            train_dataloaders=dataloaders["train"],
            val_dataloaders=dataloaders["dev"],
        )
        ckpt_path = checkpoint_callback.best_model_path
    else:
        ckpt_path = args.checkpoint_path

    trainer.test(
        model,
        dataloaders=dataloaders["test"],
        ckpt_path=ckpt_path,
    )
    eval_summary = get_eval_summary(trainer)
    with open(os.path.join(log_dir, "test_eval_summary.json"), "w") as f:
        logging.info(f"Saving test results at {f.name}")
        json.dump(eval_summary, f)


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    if args.skip_train and args.checkpoint_path is None:
        raise parser.error("Must provide --checkpoint_path if --skip_train=True")
    benchmark(args)
