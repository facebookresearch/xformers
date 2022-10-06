# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import random

import pytest
import torch

from xformers import _is_triton_available

if _is_triton_available():
    from xformers.benchmarks.benchmark_pytorch_transformer import evaluate, train
    from xformers.factory.model_factory import xFormer, xFormerConfig

    BATCH = 20
    SEQ = 32
    EMB = 8
    VOCAB = 8
    HEADS = 4
    DROP = 0.1
    LAYERS = 2
    ACTIVATION = "relu"

    _test_config_encoder = {
        "block_type": "encoder",
        "dim_model": EMB,
        "num_layers": LAYERS,
        "residual_norm_style": "post",
        "multi_head_config": {
            "num_heads": HEADS,
            "residual_dropout": DROP,
            "bias": True,
            "attention": {
                "name": "scaled_dot_product",
                "dropout": DROP,
                "seq_len": SEQ,
            },
            "dim_model": EMB,
        },
        "feedforward_config": {
            "name": "MLP",
            "dropout": DROP,
            "activation": ACTIVATION,
            "hidden_layer_multiplier": 4,
            "dim_model": EMB,
        },
    }

    _test_config_decoder = {
        "block_type": "decoder",
        "dim_model": EMB,
        "num_layers": LAYERS,
        "residual_norm_style": "post",
        "multi_head_config_masked": {
            "num_heads": HEADS,
            "residual_dropout": DROP,
            "dim_model": EMB,
            "bias": True,
            "attention": {
                "name": "scaled_dot_product",
                "dropout": DROP,
                "seq_len": SEQ,
            },
        },
        "multi_head_config_cross": {
            "num_heads": HEADS,
            "residual_dropout": DROP,
            "dim_model": EMB,
            "bias": True,
            "attention": {
                "name": "scaled_dot_product",
                "dropout": DROP,
                "seq_len": SEQ,
            },
        },
        "feedforward_config": {
            "name": "MLP",
            "dropout": DROP,
            "activation": ACTIVATION,
            "hidden_layer_multiplier": 4,
            "dim_model": EMB,
        },
    }

    _test_config = [_test_config_encoder, _test_config_decoder]

    def reset_seeds():
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        random.seed(42)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="This test requires a gpu"
    )
    def test_pytorch_encoder_parity(device=torch.device("cuda")):
        # Build both a xFormers and Pytorch model
        reset_seeds()
        model_xformers = xFormer.from_config(xFormerConfig([_test_config_encoder])).to(
            device
        )
        print(model_xformers)

        model_pytorch = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=EMB,
                nhead=HEADS,
                dim_feedforward=4 * EMB,
                dropout=DROP,
                activation=ACTIVATION,
                batch_first=True,  # (batch, seq, feature)
                device=device,
            ),
            num_layers=LAYERS,
        )
        print(model_pytorch)

        optim_xformers = torch.optim.SGD(
            model_xformers.parameters(), lr=1e-3, momentum=0.9
        )
        optim_pytorch = torch.optim.SGD(
            model_pytorch.parameters(), lr=1e-3, momentum=0.9
        )

        # Check that both models can be trained to comparable results
        eval_start_xformer = evaluate(model_xformers, BATCH, SEQ, EMB, device)
        eval_start_pytorch = evaluate(model_pytorch, BATCH, SEQ, EMB, device)
        print("starting point: ", eval_start_pytorch, eval_start_xformer)
        train(model_pytorch, optim_pytorch, "pytorch", 500, BATCH, SEQ, EMB, device)
        train(model_xformers, optim_xformers, "xformers", 500, BATCH, SEQ, EMB, device)

        # Check that we can classify this dummy example
        # Arbitrary threshold
        eval_stop_xformer = evaluate(model_xformers, BATCH, SEQ, EMB, device)
        eval_stop_pytorch = evaluate(model_pytorch, BATCH, SEQ, EMB, device)
        print("end point: ", eval_stop_pytorch, eval_stop_xformer)

        fit_ratio_xformer = eval_start_xformer / eval_stop_xformer
        fit_ratio_pytorch = eval_start_pytorch / eval_stop_pytorch
        print("fit ratios: ", fit_ratio_pytorch, fit_ratio_xformer)

        # Catch a broken training
        assert fit_ratio_xformer > 120
        assert fit_ratio_pytorch > 120

        # Catch a significant difference in between the two
        assert (
            abs(eval_start_xformer - eval_start_pytorch) < 1e-6
        )  # initial eval is about 25, arbitrary limits
        assert (
            abs(eval_stop_xformer - eval_stop_pytorch) < 1e-1
        )  # final eval is about 0.2, arbitrary limits

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="This test requires a gpu"
    )
    def test_pytorch_tranformer_parity(device=torch.device("cuda")):
        # Build both a xFormers and Pytorch model
        reset_seeds()
        model_xformers = xFormer.from_config(xFormerConfig(_test_config)).to(device)
        print(model_xformers)

        model_pytorch = torch.nn.Transformer(
            d_model=EMB,
            nhead=HEADS,
            num_encoder_layers=LAYERS,
            num_decoder_layers=LAYERS,
            dim_feedforward=4 * EMB,
            dropout=DROP,
            activation=ACTIVATION,
            batch_first=True,  # (batch, seq, feature)
            device=device,
        )
        print(model_pytorch)

        optim_xformers = torch.optim.SGD(
            model_xformers.parameters(), lr=1e-3, momentum=0.9
        )
        optim_pytorch = torch.optim.SGD(
            model_pytorch.parameters(), lr=1e-3, momentum=0.9
        )

        # Check that both models can be trained to comparable results
        eval_start_xformer = evaluate(model_xformers, BATCH, SEQ, EMB, device)
        eval_start_pytorch = evaluate(model_pytorch, BATCH, SEQ, EMB, device)
        print("starting point: ", eval_start_pytorch, eval_start_xformer)
        train(model_xformers, optim_xformers, "xformers", 100, BATCH, SEQ, EMB, device)
        train(model_pytorch, optim_pytorch, "pytorch", 100, BATCH, SEQ, EMB, device)

        # Check that we can classify this dummy example
        # Arbitrary threshold
        eval_stop_xformer = evaluate(model_xformers, BATCH, SEQ, EMB, device)
        eval_stop_pytorch = evaluate(model_pytorch, BATCH, SEQ, EMB, device)
        print("end point: ", eval_stop_pytorch, eval_stop_xformer)

        fit_ratio_xformer = eval_start_xformer / eval_stop_xformer
        fit_ratio_pytorch = eval_start_pytorch / eval_stop_pytorch

        print(fit_ratio_pytorch, fit_ratio_xformer)

        assert fit_ratio_xformer > 50
        assert fit_ratio_pytorch > 50
