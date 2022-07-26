# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import random
import time
from typing import Any, Dict, List, Tuple

import torch
import triton
from torch.cuda.amp import autocast

from xformers.benchmarks.utils import TestCase, pretty_print
from xformers.factory.model_factory import xFormer, xFormerConfig

VOCAB = 8


def _data(device, batch, seq, emb, vocab=VOCAB):
    # The dummy task is basically to classify sequences, either pure zeroes or some noise
    input_a = torch.zeros((batch, seq, emb), device=device)
    input_b = (torch.rand((batch, seq, emb), device=device) * vocab).abs()

    target_a = torch.zeros((batch, seq), device=device)
    target_b = torch.ones((batch, seq), device=device)

    if random.random() > 0.5:
        return torch.cat([input_a, input_b], dim=0), torch.cat(
            [target_a, target_b], dim=0
        )

    return torch.cat([input_b, input_a], dim=0), torch.cat([target_b, target_a], dim=0)


def reset_seeds():
    torch.manual_seed(0)
    random.seed(0)


def step(
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    batch: int,
    seq: int,
    emb: int,
    device,
):
    model.train()
    optim.zero_grad()
    batch, target = _data(device, batch, seq, emb)

    try:
        outputs = model(batch)
    except TypeError:
        # Pytorch encoder exposes target explicitly
        outputs = model(batch, tgt=batch)

    loss = torch.norm(torch.mean(outputs, dim=-1) - target)
    loss.backward()

    # Clip grad and error out if we're producing NaNs
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, norm_type=2.0, error_if_nonfinite=True)  # type: ignore
    optim.step()

    return loss.item()


def evaluate(model: torch.nn.Module, batch: int, seq: int, emb: int, device):
    reset_seeds()
    batch, target = _data(device, batch, seq, emb)
    model.eval()
    try:
        outputs = model(batch)
    except TypeError:
        # Pytorch decoder exposes target explicitly
        outputs = model(batch, tgt=batch)

    return torch.norm(torch.mean(outputs, dim=-1) - target).item()


def train(model, optimizer, name, steps, batch: int, seq: int, emb: int, device):
    # Dummy training, just checking that both options give the same results
    # Same seed for everyone
    start = time.time()
    for _ in range(steps):
        _ = step(model, optimizer, batch, seq, emb, device)

    torch.cuda.synchronize()
    print("Trained {} in {:.3}s".format(name, time.time() - start))


def bench_pytorch_encoder(
    shapes: List[Tuple[int, int, int]],
    activation: str,
    n_heads: int,
    dropout: float = 0.1,
    layers: int = 2,
    device: torch.device = torch.device("cuda"),
    steps: int = 20,
    use_amp: bool = True,
):
    results_time: Dict[str, Any] = {}
    results_memory: Dict[str, Any] = {}

    for shape in shapes:
        batch, seq, emb = shape

        # Build both a xFormers and Pytorch model
        reset_seeds()

        model_xformers = xFormer.from_config(
            xFormerConfig(
                [
                    {
                        "block_type": "encoder",
                        "dim_model": emb,
                        "num_layers": layers,
                        "residual_norm_style": "post",
                        "multi_head_config": {
                            "num_heads": n_heads,
                            "residual_dropout": dropout,
                            "use_separate_proj_weight": True,
                            "bias": True,
                            "attention": {
                                "name": "scaled_dot_product",
                                "dropout": dropout,
                                "causal": False,
                                "seq_len": seq,
                            },
                            "dim_model": emb,
                        },
                        "feedforward_config": {
                            "name": "FusedMLP",
                            "dropout": dropout,
                            "activation": activation,
                            "hidden_layer_multiplier": 4,
                            "dim_model": emb,
                        },
                    },
                ]
            )
        ).to(device)
        print(model_xformers)

        reset_seeds()
        model_pytorch = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=emb,
                nhead=n_heads,
                dim_feedforward=4 * emb,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=1e-05,
                batch_first=True,  # (batch, seq, feature)
                device=device,
            ),
            num_layers=layers,
        )
        print(model_pytorch)

        optim_xformers = torch.optim.Adam(model_xformers.parameters(), lr=1e-3)
        optim_pytorch = torch.optim.Adam(model_pytorch.parameters(), lr=1e-3)

        def run_training(model, optimizer, label):
            with autocast(enabled=use_amp):
                eval_start = evaluate(model, batch, seq, emb, device)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

                train(model, optimizer, label, steps, batch, seq, emb, device)
                max_memory = torch.cuda.max_memory_allocated() // 2**20
                print(f"Peak memory use: {max_memory}MB")

                eval_stop = evaluate(model, batch, seq, emb, device)
                print(f"Trained from {eval_start} to {eval_stop}\n")
                return eval_start, eval_stop, max_memory

        # Save the memory being used by both
        memory: Dict[str, Any] = {"pytorch": [], "xformers": []}

        def torch_train():
            _, _, max_memory = run_training(model_pytorch, optim_pytorch, "pytorch")
            memory["pytorch"].append(max_memory)

        def xformers_train():
            _, _, max_memory = run_training(model_xformers, optim_xformers, "xformers")
            memory["xformers"].append(max_memory)

        for testcase in [
            TestCase(
                xformers_train,
                "xformers",
            ),
            TestCase(
                torch_train,
                "pytorch",
            ),
        ]:
            time, _, _ = triton.testing.do_bench(lambda: testcase.function())
            key = "emb {} - heads {}".format(emb, n_heads)
            if key not in results_time:
                results_time[key] = {}
                results_memory[key] = {}

            results_time[key][testcase.name] = f"{time/1000:.1f}"

            median_memory = sorted(memory[testcase.name])[
                len(memory[testcase.name]) // 2
            ]
            results_memory[key][testcase.name] = median_memory

    pretty_print(
        results_time,
        title="\n--- Transformer training benchmark - runtime ---",
        units="s",
    )
    pretty_print(
        results_memory,
        title="\n--- Transformer training benchmark - memory use ---",
        units="MB",
    )


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    bench_pytorch_encoder(
        shapes=[(16, 128, 128), (2, 1024, 1024), (1, 1024, 2048)],
        activation="gelu",
        n_heads=8,
        dropout=0.1,
        layers=2,
        device=device,
        steps=20,
        use_amp=True,
    )
