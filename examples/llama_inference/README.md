# Llama inference

This example showcases how to use xformers kernels and cuda graphs to generate efficiently from large language models. The generation code works with both Llama2 and Code Llama (2023) models, but be aware that generating from large models will require more than a single GPU and a nightly build of PyTorch.

Example runs:
```console
$ python -m generate --ckpt_dir models/CodeLlama-7b-Instruct/
loaded SentencePiece model: #words: 32016 - bos id: 1 - eos id: 2
loaded model in 12.36 seconds
> [INST]abc[/INST] 
I'm not sure I understand what you are saying with "abc". Could you explain?
---------------
> [INST]can you write a hello world program in C#[/INST]
 Certainly! Here is a simple "Hello, World!" program in C#:
` ` `
using System;

class HelloWorld
{
    static void Main(string[] args)
    {
        Console.WriteLine("Hello, World!");
    }
}

[...]

$ # run a Code Llama model with model parallelism > 1, assuming you have two GPUs
$ torchrun --nnodes 1 --nproc-per-node 2 -m generate --ckpt_dir models/CodeLlama-13b-Instruct/
[...]
```
