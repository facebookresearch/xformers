# xFormers benchmarks

This directory contains standalone benchmark entrypoints for xFormers operators.

## Canonical long-context attention boundary

Use the memory-efficient attention benchmark with the named long-context preset:

```bash
python -m xformers.benchmarks.benchmark_mem_eff_attention \
  --preset long-context-boundary \
  --omit-backward \
  --omit-baselines \
  --label long_context_boundary
```

This command fixes the benchmark configuration to:

- device: CUDA
- dtype: `torch.float16`
- batch size: `1`
- sequence length: `8192`
- query heads: `8`
- key/value heads: `8`
- head dimension: `128`
- attention bias: `LowerTriangularMask`
- dropout: `0.0`

Report the `runtime_us` value from the `optimized` row in the emitted CSV as the
primary comparison metric. By default the benchmark writes results to:

```text
~/.cache/xformers/benchmarks/mem_eff_attention_fw/long_context_boundary.<gpu>.csv
```

Optional secondary metrics:

- `mem_use_mb` from the same CSV
- `algorithm` from the same row, to show which attention kernel handled the run

Interpretation note: this boundary is meant to stress long-context
attention/KV-related memory traffic. It is not a model-quality benchmark.

## Other benchmark entrypoints

- `benchmark_mem_eff_attention.py`: generic memory-efficient attention benchmark
- `benchmark_attn_decoding.py`: decoder-focused attention benchmark
- `benchmark_tiled_matmul.py`: tiled matmul benchmark

For ROCm-specific invocation notes, see `readme_benchmark_on_rocm.txt`.
