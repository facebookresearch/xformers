import json
import time
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from xformers.components import (
    ATTENTION_REGISTRY,
    Activation,
    AttentionConfig,
    MultiHeadDispatchConfig,
)
from xformers.components.feedforward import FEEDFORWARD_REGISTRY, FeedforwardConfig
from xformers.components.positional_encoding import PositionEncodingConfig
from xformers.factory.block_factory import xFormerEncoderBlock, xFormerEncoderConfig

# Credits: Sean Naren


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
) -> Dict[str, float]:
    # use SGD with momentum instead of Adam, since Adam is scale invariant
    # and this makes it bad for tests
    optim = torch.optim.SGD(block.parameters(), lr=lr, momentum=0.9)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(num_steps):
        optim.zero_grad()
        with torch.cuda.amp.autocast(enabled=autocast):
            input = torch.rand(batch_size, sequence_length, embed_dim)
            input = input.to(device)
            output = block(input)
            loss = F.mse_loss(input, output, reduction="sum")

        loss.backward()

        if norm_type is not None:
            clip_norm = 0.3
            torch.nn.utils.clip_grad_norm_(block.parameters(), clip_norm, norm_type)
        optim.step()

    torch.cuda.synchronize()
    max_memory = torch.cuda.max_memory_allocated() / 2 ** 20
    run_time = time.time() - start_time

    return {"run_time": run_time, "max_memory": max_memory}


def benchmark_model(num_warmup: int, num_steps: int, **kwargs) -> Dict[str, float]:
    # Run warm-up first
    _train_for_several_steps(num_steps=num_warmup, **kwargs)

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
    )

    block.to(device)

    return benchmark_model(
        num_steps=num_steps,
        num_warmup=num_warmup,
        block=block,
        batch_size=batch_size,
        sequence_length=sequence_length,
        embed_dim=embed_dim,
        autocast=autocast,
        device=device,
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

    attention_config = {
        "name": attention_name,
        "dropout": attn_dropout,
        "causal": causal,
        "window_size": sequence_length // 8,
        "from_seq_dim": sequence_length,
    }

    multi_head_config = {
        "n_heads": heads,
        "from_seq_dim": sequence_length,
        "dim_model": embed_dim,
        "residual_dropout": residual_dropout,
    }

    feedforward_config = {
        "name": feedforward_name,
        "dim_latent": embed_dim,
        "dropout": dropout,
        "activation": activation,
        "hidden_layer_multiplier": 4,
    }

    position_encoding_config = {
        "name": "sine",
        "dim_model": embed_dim,
        "seq_len": sequence_length,
    }

    block_config = xFormerEncoderConfig(
        dim_model=embed_dim,
        attention_config=AttentionConfig(**attention_config),
        multi_head_config=MultiHeadDispatchConfig(**multi_head_config),
        feedforward_config=FeedforwardConfig(**feedforward_config),
        position_encoding_config=PositionEncodingConfig(**position_encoding_config),
    )

    block = xFormerEncoderBlock.from_config(block_config)
    return block


if __name__ == "__main__":
    constants = {
        "device": torch.device("cuda"),
        "num_warmup": 5,
        "num_steps": 10,
        "dropout": 0.0,
        "attn_dropout": 0.0,
        "residual_dropout": 0.0,
    }

    param_grid = {
        "autocast": [False, True],
        "causal": [False, True],
        "heads": [8, 16],
        "activation": [a.value for a in Activation],
        "attention_name": ATTENTION_REGISTRY.keys(),
        "feedforward_name": FEEDFORWARD_REGISTRY.keys(),
        "sequence_length": [128, 512, 768],
        "embed_dim": [64, 128, 512],
        "batch_size": [8, 16, 32],
    }

    grid = ParameterGrid(param_grid)

    grid_outputs = []

    for params in tqdm(grid, total=len(grid)):
        outputs = test_xformer_encoder_block(**constants, **params)  # type: ignore
        results = {**outputs, **params}
        grid_outputs.append(results)

    print(json.dumps(grid_outputs))
