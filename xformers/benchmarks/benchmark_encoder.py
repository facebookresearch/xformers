# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import time
from contextlib import suppress
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.model_selection import ParameterGrid

# CREDITS: Sean Naren
from torch.autograd.profiler import record_function
from tqdm import tqdm

from xformers.components import Activation
from xformers.components.attention import ATTENTION_REGISTRY
from xformers.factory.block_factory import xFormerEncoderBlock, xFormerEncoderConfig

_use_cuda = torch.cuda.is_available()
_GLOBAL_ATTENTION_RATIO = 0.1  # arbitrary


def _get_attention_query_mask(sequence_length: int, ratio: float):
    mask = torch.rand((sequence_length, 1)) < ratio
    while torch.count_nonzero(mask) / float(mask.numel()) > ratio:
        mask = torch.rand((sequence_length, 1)) < ratio

    return mask


def _get_trace_handler(name: str):
    def trace_handler(prof):
        prof.export_chrome_trace(f"profile_{name}.json")
        prof.export_stacks(f"stacks_{name}.txt", "self_cuda_time_total")

    return trace_handler


def _train_for_several_steps(
    block: xFormerEncoderBlock,
    num_steps: int,
    batch_size: int,
    sequence_length: int,
    embed_dim: int,
    autocast: bool,
    device: torch.device,
    lr: float = 0.01,
    norm_type: Optional[float] = None,
    profile: bool = False,
    att_name: str = "",
) -> Dict[str, float]:
    # use SGD with momentum instead of Adam, since Adam is scale invariant
    # and this makes it bad for tests
    optim = torch.optim.SGD(block.parameters(), lr=lr, momentum=0.9)

    if _use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time.time()

    # Optional profiler, requires a context and some setup
    profiler = (
        torch.profiler.profile(  # type: ignore
            activities=[
                torch.profiler.ProfilerActivity.CPU,  # type: ignore
                torch.profiler.ProfilerActivity.CUDA,  # type: ignore
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=1),  # type: ignore
            on_trace_ready=_get_trace_handler(
                f"{att_name}_batch_{batch_size}_seq_{sequence_length}_embed_dim_{embed_dim}"
            ),
            profile_memory=True,
            with_stack=True,
        )
        if profile
        else suppress()
    )

    # Actual vanilla training loop
    # - nonsensical data, but remove that from the compute time
    inputs = torch.rand(batch_size, sequence_length).to(device)

    with profiler as p:  # type: ignore
        for _ in range(num_steps):
            optim.zero_grad()

            with torch.cuda.amp.autocast(enabled=autocast):
                with record_function("attention_forward"):
                    output = block(inputs)

                with record_function("loss"):
                    loss = F.mse_loss(
                        inputs.unsqueeze(-1).repeat(1, 1, output.shape[-1]),
                        output,
                        reduction="sum",
                    )

            with record_function("backward"):
                loss.backward()

            if norm_type is not None:
                clip_norm = 0.3
                torch.nn.utils.clip_grad_norm_(block.parameters(), clip_norm, norm_type)
            optim.step()

            if p:
                p.step()

    if _use_cuda:
        torch.cuda.synchronize()
        max_memory = torch.cuda.max_memory_allocated() / 2**20
    else:
        max_memory = -1
    run_time = time.time() - start_time

    return {"run_time": run_time, "max_memory": round(max_memory, 1)}


def benchmark_model(num_warmup: int, num_steps: int, **kwargs) -> Dict[str, float]:
    # Run warm-up first
    warm_up_args = {**kwargs}
    warm_up_args["profile"] = False
    _train_for_several_steps(num_steps=num_warmup, **warm_up_args)

    return _train_for_several_steps(num_steps=num_steps, **kwargs)


def test_xformer_encoder_block(
    attention_name: str,
    feedforward_name: str,
    heads: int,
    attn_dropout: float,
    residual_dropout: float,
    causal: bool,
    activation: Activation,
    autocast: bool,
    batch_size: int,
    sequence_length: int,
    embed_dim: int,
    dropout: float,
    num_steps: int,
    num_warmup: int,
    device: torch.device,
    profile: bool,
) -> Dict[str, float]:

    block = instantiate_xformer(
        activation=activation,
        attention_name=attention_name,
        attn_dropout=attn_dropout,
        causal=causal,
        feedforward_name=feedforward_name,
        heads=heads,
        residual_dropout=residual_dropout,
        sequence_length=sequence_length,
        embed_dim=embed_dim,
        dropout=dropout,
    ).to(device)

    print(
        "Testing:",
        block,
        batch_size,
        sequence_length,
        embed_dim,
        autocast,
        device,
        attention_name,
    )

    return benchmark_model(
        num_steps=num_steps,
        num_warmup=num_warmup,
        block=block,
        batch_size=batch_size,
        sequence_length=sequence_length,
        embed_dim=embed_dim,
        autocast=autocast,
        device=device,
        profile=profile,
        att_name=attention_name,
    )


def instantiate_xformer(
    activation: Activation,
    attention_name: str,
    attn_dropout: float,
    causal: bool,
    feedforward_name: str,
    heads: int,
    residual_dropout: float,
    sequence_length: int,
    embed_dim: int,
    dropout: float,
) -> xFormerEncoderBlock:

    block_size = 16

    attention_config = {
        "name": attention_name,
        "dropout": attn_dropout,
        "causal": causal,
        "seq_len": sequence_length,
        "attention_query_mask": _get_attention_query_mask(
            sequence_length, _GLOBAL_ATTENTION_RATIO
        ),
        "num_heads": heads,
        "dim_head": embed_dim / heads,
        "layout": torch.eye(
            sequence_length // block_size,
            sequence_length // block_size,
            dtype=torch.long,
        )
        .unsqueeze(0)
        .expand(heads, -1, -1),
        "block_size": block_size,
    }

    multi_head_config = {
        "num_heads": heads,
        "dim_model": embed_dim,
        "residual_dropout": residual_dropout,
        "attention": attention_config,
    }

    feedforward_config = {
        "name": feedforward_name,
        "dim_model": embed_dim,
        "dropout": dropout,
        "activation": activation,
        "hidden_layer_multiplier": 4,
    }

    position_embedding_config = {
        "name": "sine",
        "dim_model": embed_dim,
        "seq_len": sequence_length,
    }

    block_config = xFormerEncoderConfig(
        dim_model=embed_dim,
        multi_head_config=multi_head_config,
        feedforward_config=feedforward_config,
        position_encoding_config=position_embedding_config,
    )

    block = xFormerEncoderBlock.from_config(block_config)
    return block


def plot(args, results: List[Dict[str, Any]]):
    df = pd.DataFrame(results)
    HEADS = args.heads[-1]
    AMP = args.pytorch_amp[-1]
    EMB = args.embedding_dim[-1]
    CAUSAL = args.causal[-1]
    BATCH_SIZE = args.batch_size[-1]
    ACTIVATION = args.activations[-1]

    df_filtered = df[
        (df["activation"] == ACTIVATION)
        & (df["heads"] == HEADS)
        & (df["autocast"] == AMP)
        & (df["embed_dim"] == EMB)
        & (df["causal"] == CAUSAL)
        & (df["batch_size"] == BATCH_SIZE)
    ]

    df_filtered.sort_values(
        by=["sequence_length", "max_memory"], ascending=[False, True], inplace=True
    )
    sns.barplot(
        x="sequence_length",
        y="max_memory",
        hue="attention_name",
        data=df_filtered,
        palette="Set2",
    )
    plt.xlabel("Sequence length")
    plt.ylabel("Max memory being used")
    plt.title("Memory use")
    plt.savefig("memory_vs_attention.png")
    plt.clf()

    df_filtered.sort_values(
        by=["sequence_length", "run_time"], ascending=[False, True], inplace=True
    )
    sns.barplot(
        x="sequence_length",
        y="run_time",
        hue="attention_name",
        data=df_filtered,
        palette="Set2",
    )
    plt.xlabel("Sequence length")
    plt.ylabel("Average epoch time")
    plt.title("Runtime")
    plt.savefig("runtime_vs_attention.png")


if __name__ == "__main__":
    # Get the user requests
    parser = argparse.ArgumentParser(
        "Benchmark different attention mechanisms on various sequence lengths"
    )
    parser.add_argument(
        "-a", "--attentions", nargs="+", default=list(ATTENTION_REGISTRY.keys())
    )
    parser.add_argument("-mlp", "--mlp", nargs="+", default=["MLP"])
    parser.add_argument(
        "-act", "--activations", nargs="+", default=[a.value for a in Activation]
    )
    parser.add_argument(
        "-emb", "--embedding_dim", nargs="+", default=[64, 128, 256], type=int
    )
    parser.add_argument(
        "-sl", "--sequence_length", nargs="+", default=[576, 1024], type=int
    )
    parser.add_argument("-bs", "--batch_size", nargs="+", default=[8, 16, 32], type=int)
    parser.add_argument("-heads", "--heads", nargs="+", default=[8, 16], type=int)

    parser.add_argument("-fp16", "--pytorch_amp", nargs="+", default=[True], type=bool)
    parser.add_argument("-causal", "--causal", nargs="+", default=[False], type=bool)
    parser.add_argument("-plot", "--plot", action="store_true", default=False)
    parser.add_argument(
        "-profile",
        "--profile",
        help="Pofile the runtime and memory",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    # Setup the test configs
    constants = {
        "device": torch.device("cuda") if _use_cuda else torch.device("cpu"),
        "num_warmup": 5,
        "num_steps": 10,
        "dropout": 0.1,
        "attn_dropout": 0.1,
        "residual_dropout": 0.1,
        "profile": args.profile,
    }

    param_grid = {
        "autocast": args.pytorch_amp,
        "causal": args.causal,
        "heads": args.heads,
        "activation": args.activations,
        "attention_name": args.attentions,
        "feedforward_name": args.mlp,
        "sequence_length": args.sequence_length,
        "embed_dim": args.embedding_dim,
        "batch_size": args.batch_size,
    }

    print(
        "Testing the following parameters: \n",
        json.dumps(param_grid, sort_keys=True, indent=4),
    )

    grid = ParameterGrid(param_grid)

    grid_outputs = []

    for params in tqdm(grid, total=len(grid)):
        outputs = test_xformer_encoder_block(**constants, **params)  # type: ignore
        results = {**outputs, **params}
        grid_outputs.append(results)

    print(json.dumps(grid_outputs, sort_keys=True, indent=4))

    # Optional plots
    if args.plot:
        plot(args, grid_outputs)
