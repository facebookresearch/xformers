# A minGPT-like example using components from xformers.
#
# In this example we use the xformers components directly to
# construct a transformer model, rather than configuring a
# fully-constructed model. This also directly runs the
# training loop without relying on a separate trainer.
#
# based on:
# https://github.com/karpathy/minGPT/blob/master/projects/chargpt/chargpt.py
# and
# https://github.com/facebookresearch/xformers/blob/main/examples/microGPT.py


import os
import time

import torch
import torch.nn.functional as F
import torch.utils
from torch import nn
from torch.utils.data import DataLoader, Dataset

import xformers
from xformers.components import MultiHeadDispatch, positional_embedding
from xformers.components.attention import ScaledDotProduct, attention_patterns


# copied from
# https://github.com/karpathy/minGPT/blob/master/projects/chargpt/chargpt.py
class CharDataset(Dataset):
    """
    Emits batches of characters
    """

    def __init__(self, seq_len, data):
        self.seq_len = seq_len

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print("data has %d characters, %d unique." % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = vocab_size
        self.data = data

    def get_vocab_size(self):
        return self.vocab_size

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx : idx + self.seq_len + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


# A block of a GPT-style "decoder only" transformer
class GPTBlock(nn.Module):
    def __init__(self, ndim, heads, attention, dropout=0.1):
        super().__init__()
        self.ndim = ndim
        self.mhd = MultiHeadDispatch(ndim, heads, attention)
        self.fc1 = nn.Linear(ndim, ndim * 4)
        self.fc2 = nn.Linear(ndim * 4, ndim)
        self.ln1 = nn.LayerNorm(ndim)
        self.ln2 = nn.LayerNorm(ndim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seq = x.shape[1]

        x = self.ln1(x)

        mask = attention_patterns.causal_1d_pattern(seq).to(x.device)
        x = x + self.mhd(x, att_mask=mask)

        residual = x
        x = self.ln2(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = residual + self.dropout(self.fc2(x))

        return x


def _init_weights(root_module):
    for module in root_module.children():
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, GPTBlock):
            _init_weights(module)
        # initializes their own weights
        elif isinstance(module, positional_embedding.PositionEmbedding):
            pass
        elif isinstance(module, xformers.components.MultiHeadDispatch):
            pass


class CharGPT(nn.Module):
    def __init__(self, seq_len, vocab, ndim, heads, layers=6):
        super().__init__()
        self.seq_len = seq_len
        self.ndim = ndim

        self.embed = positional_embedding.VocabEmbedding(ndim, seq_len, vocab)

        self.ff = nn.Linear(ndim, ndim)
        self.mh_layers = []
        attention = ScaledDotProduct()
        for i in range(layers):
            self.mh_layers.append(GPTBlock(ndim, heads, attention))
        self.mh_layers = nn.ModuleList(self.mh_layers)
        self.lm_head = nn.Linear(ndim, vocab, bias=False)
        _init_weights(self)

    # x => batch x seq_len x n_dim
    def forward(self, x):
        x = self.embed(x)

        # already handled in mhd when `SparseDotProduct().requires_input_projection=True`
        # k, q, v = x.split(x.shape[-1] // 3, dim=-1)

        for layer in self.mh_layers:
            x = layer(x)
        x = self.lm_head(x)
        return x

    def generate(self, prefix, max_new_tokens, temperature=1.0, top_k=10):
        result = prefix
        for _ in range(max_new_tokens):
            prefix = result[-self.seq_len :]
            logits = self.forward(prefix.unsqueeze(0))

            # get logits from last value in sequence nad scale
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_val = torch.multinomial(probs[0], num_samples=1)

            result = torch.cat((result, next_val), dim=-1)
        return result


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    # m1 on mac
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    else:
        return "cpu"


if __name__ == "__main__":
    device = get_device()

    seq_len = 128
    batchsize = 64
    heads = 6
    n_dim = heads * 33

    if not os.path.exists("input.txt"):
        os.system(
            "curl https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o input.txt" # noqa: E501
        )
    text = open("input.txt", "r").read()
    dataset = CharDataset(seq_len, text)

    sampler = torch.utils.data.RandomSampler(
        dataset, replacement=True, num_samples=int(1e10)
    )
    loader = DataLoader(
        dataset, sampler=sampler, shuffle=False, batch_size=batchsize, num_workers=4
    )

    model = CharGPT(seq_len, dataset.get_vocab_size(), n_dim, heads)
    model.to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    params_decay = []
    params_nodecay = []
    for n, p in model.named_parameters():
        if not any(nd in n for nd in no_decay):
            params_decay.append(p)
        else:
            params_nodecay.append(p)
    optim_groups = [
        {"params": params_decay, "weight_decay": 0.1},
        {"params": params_nodecay, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=1e-4, betas=(0.9, 0.95))

    model.train()

    steps = -1
    for x, y in loader:
        steps += 1
        step_start = time.time()
        prev_step = steps
        x = x.to(device)
        y = y.to(device)
        logits = model(x)

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if steps % 100 == 0:
            iter_time_ms = (time.time() - step_start) * 1000
            print(f"step {steps}; loss {loss.item():.4f} ; time {iter_time_ms:.2f}ms")
        if steps % 500 == 0:
            with torch.no_grad():
                context = "O God, O God!"
                x = torch.tensor(
                    [dataset.stoi[s] for s in context], dtype=torch.long, device=device
                )
                result = model.generate(x, max_new_tokens=500)
                print("=================")
                print("".join([dataset.itos[i.item()] for i in result]))
                print("=================")
                print()
