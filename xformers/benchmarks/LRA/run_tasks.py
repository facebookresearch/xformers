# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


# CREDITS: adapted from the Nystromformer repo
# https://github.com/mlpen/Nystromformer

import argparse
import datetime
import json
import logging
import math
import os
import random
import sys
import time
from contextlib import suppress
from enum import Enum
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, flop_count_str
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from xformers.benchmarks.LRA.code.dataset import LRADataset
from xformers.benchmarks.LRA.code.model_wrapper import ModelForSC, ModelForSCDual
from xformers.components.attention import ATTENTION_REGISTRY
from xformers.utils import temp_files_ctx


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

    if task == Task.Retrieval:
        model: nn.Module = ModelForSCDual(config[f"{task}"], attention_name)
    else:
        model = ModelForSC(config[f"{task}"], attention_name)

    args.logger.info(model)
    args.logger.info(
        f"num_parameter: {np.sum([np.prod(weight.size()) for weight in model.parameters()]) // 1e3 / 1e3}M"
    )

    with torch.no_grad():
        # Check the flops
        seq_len = config[f"{task}"]["model"]["common"]["seq_len"]
        x = torch.rand(1, seq_len).long()
        mask = torch.rand(1, seq_len).long()
        indices = torch.rand(1, seq_len).long()
        flops = FlopCountAnalysis(model.model, (x, mask, indices))
        args.logger.info(f"complexity: {round(flops.total()/1e9, 3)} GFlops")
        args.logger.info(flop_count_str(flops))

    return model


def build_training_setup(
    config_training: Dict,
    task: Task,
    model: nn.Module,
    rank: int = 0,
    world_size: int = 1,
):
    datasets = {}
    samplers = {}

    for component in ["train", "test", "dev"]:
        dataset = LRADataset(
            file_path=f"datasets/{task}.{component}.pickle",
            seq_len=config_training["seq_len"],
        )

        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=(component == "train"),
            drop_last=(component == "train"),
        )  # type:ignore
        datasets[component] = dataset
        samplers[component] = sampler

    logging.info(f"Learning rate: {config_training['learning_rate']}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config_training["learning_rate"],
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=config_training["weight_decay"],
    )

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(  # type: ignore
        optimizer=optimizer,
        max_lr=config_training["learning_rate"],
        pct_start=config_training["warmup"] / config_training["num_train_steps"],
        anneal_strategy=config_training["lr_decay"],
        total_steps=config_training["num_train_steps"],
    )

    amp_scaler = torch.cuda.amp.GradScaler(enabled=config_training["mixed_precision"])

    logging.info(f"Dataloader ready. Rank {rank} of {world_size}")

    return datasets, samplers, optimizer, lr_scheduler, amp_scaler


def print_summary(
    summary,
    save_if_improved,
    train_step_idx,
    model,
    checkpoint_path,
    logger,
    tb_logger=None,
):

    summary["loss"] = np.average(summary["loss"], weights=summary["count"])
    summary["accu"] = np.average(summary["accu"], weights=summary["count"])
    summary["count"] = np.sum(summary["count"]).astype(float)

    if summary["accu"] > summary["best_accu"]:
        summary["best_accu"] = summary["accu"]
        if save_if_improved:
            best_accu = summary["best_accu"]
            torch.save(
                {"model_state_dict": model.state_dict()},
                checkpoint_path,
            )
            logger.info(f"best_accu={best_accu:.3f}. Saved best model")

    summary["max_memory_mb"] = torch.cuda.max_memory_allocated() // 1e3 / 1e3

    summary_round = {"train_step_idx": train_step_idx}
    for key in summary:
        if type(summary[key]) is str:
            summary_round[key] = summary[key]
        else:
            summary_round[key] = round(summary[key], 4)

    if tb_logger:
        tb_logger.add_scalar("acc", summary["accu"], train_step_idx)
        tb_logger.add_scalar("loss", summary["loss"], train_step_idx)
        tb_logger.add_scalar("max_mem", summary["max_memory_mb"], train_step_idx)
        tb_logger.add_scalar("count", summary["count"], train_step_idx)

    logger.info(summary_round)
    logger.info(json.dumps(summary_round, sort_keys=True) + "\n")

    summary["t"] = 0
    summary["loss"] = []
    summary["accu"] = []
    summary["count"] = []


def setup_log(args, rank, attention_name, task):
    log_f = Path(
        os.path.join(
            args.checkpoint_dir, f"{task}__{attention_name}__{rank}_output.log"
        )
    )
    if not log_f.exists():
        log_f.parent.mkdir(parents=True, exist_ok=True)
        with open(log_f, "x") as _:
            pass

    logger = torch.multiprocessing.get_logger()
    logger.setLevel(level=logging.INFO)
    logger.addHandler(logging.FileHandler(filename=str(log_f)))
    if rank == 0:
        logger.addHandler(logging.StreamHandler(sys.stdout))
    return log_f.absolute(), logger


def eval_model(model, dataloaders, component, config, step):
    model.eval()

    for dev_step_idx, batch_dev in enumerate(dataloaders[component]):
        _ = step(
            batch_dev,
            component,
            step_idx=dev_step_idx,
            step_max=config["num_eval_steps"],
        )

        if dev_step_idx == config["num_eval_steps"]:
            break

    model.train()


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


def seed_worker(_: int):
    # Make sure that non-pytorch random generators are properly set
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def benchmark(rank, args):
    # Setup multiprocessing
    dist.init_process_group(
        init_method="file://" + args.temp_file,
        backend="NCCL",
        rank=rank,
        world_size=args.world_size,
    )
    try:
        torch.cuda.set_device(args.gpu)
    except AttributeError:
        # Single node launcher
        torch.cuda.set_device(rank)

    task = args.task
    attention_name = args.attention

    # Build the problem
    log_f_path, logger = setup_log(args, rank, attention_name, task)
    args.logger = logger
    config = load_config(args.config)

    config_task = config[f"{task}"]
    if args.sweep_parameters is not None:
        logger.info("Replacing hyperparameters")
        rewrite_hyper(config_task, args.sweep_parameters)

    config_training = config_task["training"]
    config_training["seq_len"] = config_task["model"]["common"]["seq_len"]
    model = build_model(args, config)

    torch.manual_seed(config_training.get("seed", 0))  # also sets the cuda seed
    np.random.seed(config_training.get("seed", 0))
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()

    # tensorboard
    tb_logger = SummaryWriter(args.tb_dir)

    torch.manual_seed(config_training.get("seed", 0))  # also sets the cuda seed
    np.random.seed(config_training.get("seed", 0))
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()

    # tensorboard
    tb_logger = SummaryWriter(args.tb_dir)

    # Setup the training
    device_ids = list(range(torch.cuda.device_count()))
    logger.info(f"GPU list: {device_ids}")
    model = model.cuda()
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[rank], broadcast_buffers=True, find_unused_parameters=True
    )

    (
        datasets,
        samplers,
        optimizer,
        lr_scheduler,
        amp_scaler,
    ) = build_training_setup(config_training, task, model, rank, args.world_size)

    init_t = time.time()

    # Messenger structure which will be moved around to collect metrics
    summary = {
        comp: {
            "t": 0,
            "loss": [],
            "accu": [],
            "count": [],
            "best_accu": 0,
            "component": comp,
        }
        for comp in ["train", "dev", "test"]
    }

    # Setup the dataloaders
    accumu_steps = config_task["training"]["gradient_accumulation"]
    per_gpu_batch_size = (
        config_training["batch_size"] // args.world_size // accumu_steps
    )
    logging.warning(
        "Requested batch size: {}. Given world size and grad accumulation, per-gpu batch is {}".format(
            config_training["batch_size"], per_gpu_batch_size
        )
    )

    # reset train/eval steps if using gradient accumulation
    if accumu_steps > 1:
        config_training["num_train_steps"] *= accumu_steps
        config_training["num_eval_steps"] *= accumu_steps

    epochs = math.ceil(
        config_training["num_train_steps"]
        * config_training["batch_size"]
        / len(datasets["train"])
    )

    logging.warning(
        "Requested train steps: {}. Given dataset, this translates into {} epochs".format(
            config_training["num_train_steps"], epochs
        )
    )

    logger.info(f"accumu_steps={accumu_steps}")
    model_path = str(log_f_path).replace(".log", ".model")
    g = torch.Generator()
    g.manual_seed(config_training.get("seed", 0))

    dataloaders = {
        k: DataLoader(
            datasets[k],
            sampler=samplers[k],
            batch_size=per_gpu_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=1,
            worker_init_fn=seed_worker,
            generator=g,
        )
        for k in datasets.keys()
    }

    # Our step function
    def step(
        batch: Dict[str, Any],
        component: str,
        step_idx: int,
        step_max: int,
        accumulate: bool = False,
    ):
        if step_idx > step_max:
            logger.warning(
                "Calling `step` beyond the training schedule, this is probably a mistake"
            )
            return

        t0 = time.time()
        batch_size = batch[list(batch.keys())[0]].size(0)

        for key in batch:
            batch[key] = batch[key].cuda()

        if component == "train":
            acc_context = model.no_sync() if accumulate else suppress()

            with acc_context, torch.autograd.set_detect_anomaly(args.debug):
                outputs = model(**batch)
                amp_scaler.scale(outputs["loss"]).backward()

                if not accumulate:
                    amp_scaler.step(optimizer)
                    optimizer.zero_grad()
                    amp_scaler.update()
                    lr_scheduler.step()

        else:
            with torch.no_grad():
                outputs = model(**batch)

        t1 = time.time()

        t_escape = t1 - t0
        learning_rate = optimizer.param_groups[0]["lr"]
        loss = outputs["loss"].item()
        accu = outputs["accu"].item()
        cnt = outputs["count"]
        time_since_start = time.time() - init_t
        eta = (
            datetime.timedelta(
                seconds=round(time_since_start / (step_idx + 1) * step_max)
            )
            if component == "train"
            else -1
        )

        if not step_idx % 10:
            logger.info(
                f"{component}: step={step_idx}/{step_max}, total_time={time_since_start:.1f},"
                + f" eta={eta},"
                + f" batch_time={t_escape:.3f}, bs={batch_size}, lr={learning_rate:.6f},"
                + f" loss={loss:.4f}, accu={accu:.4f}",
            )

        summary[component]["t"] += t_escape
        summary[component]["loss"].append(loss)
        summary[component]["accu"].append(accu)
        summary[component]["count"].append(cnt)

        if not accumulate:
            step_idx += 1

        return loss, step_idx

    # Start training or evaluating
    train_step_idx = 0
    if not args.skip_train:
        try:
            model.train()
            for epoch in range(epochs):
                logger.info(f"\nEpoch {epoch}")

                # Make sure that per-rank sampling is really random
                for sampler in samplers.values():
                    sampler.set_epoch(epoch)

                for i_batch, batch in enumerate(dataloaders["train"]):
                    grad_accumulate = (
                        i_batch % config_training["gradient_accumulation"] != 0
                    )

                    _, train_step_idx = step(
                        batch,
                        component="train",
                        step_idx=train_step_idx,
                        step_max=config_training["num_train_steps"],
                        accumulate=grad_accumulate,
                    )

                    if not (train_step_idx + 1) % config_training["eval_frequency"]:
                        print_summary(
                            summary["train"],
                            False,
                            train_step_idx,
                            model,
                            model_path,
                            logger,
                        )

                        eval_model(model, dataloaders, "dev", config_training, step)

                        print_summary(
                            summary["dev"],
                            True,
                            train_step_idx,
                            model,
                            model_path,
                            logger,
                            tb_logger,
                        )

                    if train_step_idx == config_training["num_train_steps"]:
                        break

        except KeyboardInterrupt as e:
            print(e)

    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    try:
        eval_model(model, dataloaders, "test", config_training, step)
    except StopIteration:
        pass

    print_summary(summary["test"], False, train_step_idx, model, model_path, logger)


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
    parser.add_argument(
        "--tb_dir",
        type=str,
        help="Path to the tensorboard directory",
        dest="tb_dir",
        default=f"/checkpoints/{os.getenv('USER')}/xformers/tb",
    )
    return parser


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    setup_log(args, "main", f"{args.attention}", f"{args.task}")

    with temp_files_ctx(num=1) as temp_files:
        args.temp_file = temp_files[0]
        torch.multiprocessing.spawn(
            benchmark, args=(args,), nprocs=args.world_size, join=True
        )
