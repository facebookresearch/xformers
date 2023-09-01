# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# A MinGPT + Lightning + xFormers example Code from Sean Naren (@seannaren)
# This is an hommage to https://github.com/karpathy/minGPT

import math
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.utilities import rank_zero_info
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler

from xformers.factory.model_factory import xFormer, xFormerConfig


class GPT(pl.LightningModule):
    """the full GPT language model, with a context size of block_size"""

    def __init__(
        self,
        vocab_size,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        learning_rate=6e-4,
        n_embd=512,
        block_size=128,
        n_layer=8,
        n_head=8,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        mlp_pdrop=0.1,
        attention="scaled_dot_product",
        hidden_layer_multiplier=4,
        warmup_tokens=20,
        final_tokens=1000,
    ):
        super().__init__()

        # auto creates self.hparams from the method signature
        self.save_hyperparameters()

        # A list of the encoder or decoder blocks which constitute the Transformer.
        xformer_config = [
            {
                "reversible": False,  # Turn on to test the effect of using reversible layers
                "block_type": "encoder",
                "num_layers": self.hparams.n_layer,
                "dim_model": self.hparams.n_embd,
                "residual_norm_style": "post",
                "position_encoding_config": {
                    "name": "vocab",
                    "seq_len": self.hparams.block_size,
                    "vocab_size": self.hparams.vocab_size,
                },
                "multi_head_config": {
                    "num_heads": self.hparams.n_head,
                    "residual_dropout": self.hparams.resid_pdrop,
                    "use_rotary_embeddings": True,
                    "attention": {
                        "name": self.hparams.attention,
                        "dropout": self.hparams.attn_pdrop,
                        "causal": True,
                        "seq_len": self.hparams.block_size,
                        "num_rules": self.hparams.n_head,
                    },
                },
                "feedforward_config": {
                    "name": "FusedMLP",  # Use MLP if Triton is not available
                    "dropout": self.hparams.mlp_pdrop,
                    "activation": "gelu",
                    "hidden_layer_multiplier": self.hparams.hidden_layer_multiplier,
                },
            }
        ]

        config = xFormerConfig(xformer_config)
        config.weight_init = "small"
        self.model = xFormer.from_config(config)

        # decoder head
        self.ln_f = nn.LayerNorm(self.hparams.n_embd)
        self.head = nn.Linear(self.hparams.n_embd, self.hparams.vocab_size, bias=False)

        self.block_size = self.hparams.block_size
        self.apply(self._init_weights)

        self._tokens_seen = 0

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # Reset the token counter
        self._tokens_seen = 0

    def get_block_size(self):
        return self.block_size

    def configure_optimizers(self):
        # Create the optimizer and the training schedule:
        # - Handle the per-param weight decay
        no_decay = ["bias", "LayerNorm.weight"]
        params_decay = [
            p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)
        ]
        params_nodecay = [
            p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)
        ]
        optim_groups = [
            {"params": params_decay, "weight_decay": self.hparams.weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]

        # - Start with a warm up, ramp up then cosine
        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.hparams.learning_rate, betas=self.hparams.betas
        )

        def update_lr(*_):
            config = self.hparams

            if self._tokens_seen < config.warmup_tokens:
                # linear warmup
                lr_mult = float(self._tokens_seen) / float(max(1, config.warmup_tokens))
                lr_mult = max(lr_mult, 1e-2)  # could be that we've not seen any yet
            else:
                # cosine learning rate decay
                progress = float(self._tokens_seen - config.warmup_tokens) / float(
                    max(1, config.final_tokens - config.warmup_tokens)
                )
                lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

            return lr_mult

        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=[update_lr, update_lr],
            ),
            "name": "learning_rate",
            "interval": "step",  # The unit of the scheduler's step size
            "frequency": 1,  # The frequency of the scheduler
        }
        return [optimizer], [lr_scheduler]

    def forward(self, src):
        # predict the next tokens (in latent space)
        prediction = self.model(src)

        # translate the predictions into tokens
        prediction = self.ln_f(prediction)
        logits = self.head(prediction)

        return logits

    def training_step(self, batch, _):
        src, targets = batch

        # Update the tokens we've seen (tracked for LR scheduling)
        self._tokens_seen += (src >= 0).numel()

        # same action as inference
        logits = self(src)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        self.logger.log_metrics(
            {
                "train_loss": loss.mean(),
                "learning_rate": self.lr_schedulers().get_last_lr()[0],
            },
            step=trainer.global_step,
        )

        return loss


class CharDataset(Dataset):
    def __init__(self, data, block_size):
        chars = list(set(data))
        data_size, vocab_size = len(data), len(chars)
        rank_zero_info("data has %d characters, %d unique." % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, i):
        chunk = self.data[i : i + self.block_size + 1]
        dix = [self.stoi[s] for s in chunk]

        # src and target are off by one, we want the model to predict the next word
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

    def to_tokens(self, message, device):
        return torch.tensor([self.stoi[s] for s in message], dtype=torch.long)[
            None, ...
        ].to(device)

    def from_tokens(self, tokens):
        return "".join([self.itos[int(i)] for i in tokens])


@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()

    # CREDITS: https://github.com/karpathy/minGPT/blob/master/mingpt/utils.py
    def top_k_logits(logits, k):
        v, _ = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[:, [-1]]] = -float("Inf")
        return out

    for _ in range(steps):
        x_cond = (
            x if x.size(1) <= block_size else x[:, -block_size:]
        )  # crop context if needed
        logits = model(x_cond)

        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature

        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)

        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)

        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)

        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x[0]  # escape the batch dimension


if __name__ == "__main__":
    seed_everything(42)

    # Adjust batch depending on the available memory on your machine.
    # You can also use reversible layers to save memory
    REF_BATCH = 512
    BATCH = 128

    WORKERS = 4
    EPOCHS = 1
    BLOCK = 128
    WARMUP = 20

    if not os.path.exists("input.txt"):
        os.system(
            "wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        )

    text = open("input.txt", "r").read()
    train_dataset = CharDataset(
        text, BLOCK
    )  # one line of poem is roughly 50 characters
    random_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        sampler=random_sampler,
        batch_size=BATCH,
        num_workers=WORKERS,
        pin_memory=True,
    )

    model = GPT(
        vocab_size=train_dataset.vocab_size,
        block_size=train_dataset.block_size,
        attention="scaled_dot_product",
        warmup_tokens=REF_BATCH * WARMUP,
        final_tokens=EPOCHS * len(train_dataset) * BLOCK,
    )
    print(model)

    trainer = Trainer(
        gpusdevices=1,
        accelerator="gpu",
        max_epochs=EPOCHS,
        precision=16,
        log_every_n_steps=1,
        accumulate_grad_batches=REF_BATCH // BATCH,
    )

    trainer.fit(model, train_loader)

    # Sample from the model, let it predict a paragraph
    context = "Friends of my soul"  # prime with something
    x = train_dataset.to_tokens(context, model.device)
    y = sample(model, x, steps=1000, temperature=1.0, sample=True, top_k=10)

    print(train_dataset.from_tokens(y))
