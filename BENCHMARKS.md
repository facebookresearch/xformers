
# Benchmarks: how to and some results

## Benchmark a full encoder block

Sweeping over different attention settings to log max memory use and runtime can for instance be done by invoking
`python3 xformers/benchmarks/benchmark_encoder.py`. Specifying a subset to test is done through command line arguments, for instance `python3 xformers/benchmarks/benchmark_encoder.py --causal True --attentions random --activations gelu -fp16 True`.

Please note that:

- These numbers are dependent of hyperparameters (dimensions chosen for Linformer, sparsity of the pattern), they are mostly an illustration
- The sparse attention patterns tested here are just presets, as explained in the linked notebook generating any new sparse attention pattern should be relatively easy, while keeping the benefits of optimized computations.

Some examples, generated with `python3 xformers/benchmarks/benchmark_encoder.py --activations gelu --plot -emb 256 -bs 32 -heads 16`

![Memory use for different attentions](docs/plots/memory_vs_attention.png)  ![Runtime for different attentions](docs/plots/runtime_vs_attention.png)

## Benchmark the core sparse attention mechanisms

`python3 xformers./benchmarks/benchmark_core.py` will measure the speed of the core sparse attention mechanism. The current numbers are as follows (times in microseconds (us)):

|                        | **matmul_with_mask**  |                        | **softmax**           |                        | **bmm**               |                        |
| ---------------------- | --------------------- | ---------------------- | --------------------- | ---------------------- | --------------------- | ---------------------- |
|                        | **B=8, M=256, K=128** | **B=8, M=1024, K=256** | **B=8, M=256, K=128** | **B=8, M=1024, K=256** | **B=8, M=256, K=128** | **B=8, M=1024, K=256** |
| dense                  | 62.3                  | 510.3                  | 12.8                  | 141.9                  | 31.0                  | 590.7                  |
| dense with masking     | 84.2                  | 805.3                  | -                     | -                      | -                     | -                      |
| sparsity pytorch: 0.50 | 392.4                 | 6197.4                 | 1140.9                | 8081.4                 | 577.0                 | 13830.2                |
| sparsity pytorch: 0.80 | 336.2                 | 4437.3                 | 515.0                 | 3494.8                 | 254.4                 | 5944.0                 |
| sparsity pytorch: 0.90 | 244.1                 | 3017.4                 | 367.3                 | 1932.6                 | 162.0                 | 3063.0                 |
| sparsity pytorch: 0.95 | 193.2                 | 1899.5                 | 293.6                 | 1078.9                 | 161.6                 | 1692.3                 |
| sparsity pytorch: 0.99 | 195.6                 | 695.0                  | 252.1                 | 342.4                  | 161.9                 | 433.4                  |
| sparsity sputnik: 0.50 | 77.9                  | 1695.9                 | 32.8                  | 164.7                  | 64.6                  | 1640.5                 |
| sparsity sputnik: 0.80 | 43.8                  | 793.0                  | 32.9                  | 50.8                   | 39.6                  | 703.3                  |
| sparsity sputnik: 0.90 | 43.6                  | 435.5                  | 33.0                  | 33.5                   | 39.6                  | 391.4                  |
| sparsity sputnik: 0.95 | 43.2                  | 258.6                  | 32.5                  | 32.7                   | 39.7                  | 223.6                  |
| sparsity sputnik: 0.99 | 43.5                  | 145.4                  | 33.2                  | 32.7                   | 39.7                  | 77.4                   |

##  Triton layers

### Fused softmax

You can reproduce these numbers locally by running `python3 xformers/benchmarks/benchmark_triton_softmax.py`. The units are GB/s. These results are for a nVidia V100, and PyTorch 1.9.


| Float16               | B=8, M=384, K=128 | B=8, M=784, K=512 | B=4, M=2048, K=384 | B=4, M=3136, K=1024 | B=2, M=1024, K=2048 | B=2, M=2048, K=4096 | B=2, M=4096, K=4096 |
| --------------------- | ----------------- | ----------------- | ------------------ | ------------------- | ------------------- | ------------------- | ------------------- |
| pytorch - fw          | 170.7             | 501.8             | 512.0              | 597.3               | 399.6               | 524.3               | 553.0               |
| triton  - fw          | 153.6             | 522.7             | 512.0              | 716.8               | 606.8               | 736.4               | 775.6               |
| pytorch - log - fw    | 192.0             | 545.4             | 534.3              | 669.0               | 496.5               | 601.2               | 615.4               |
| triton  - log - fw    | 153.6             | 570.2             | 558.5              | 748.9               | 682.7               | 780.2               | 799.2               |
| pytorch - fw+bw       | 71.4              | 170.7             | 168.3              | 205.6               | 164.7               | 196.5               | 203.5               |
| triton  - fw+bw       | 69.8              | 218.2             | 211.9              | 264.8               | 224.4               | 271.4               | 284.3               |
| pytorch - log - fw+bw | 78.8              | 207.3             | 204.8              | 255.3               | 206.1               | 247.3               | 255.5               |
| triton  - log - fw+bw | 71.4              | 220.1             | 213.7              | 266.9               | 229.1               | 273.6               | 285.6               |

| Float32               | B=8, M=384, K=128 | B=8, M=784, K=512 | B=4, M=2048, K=384 | B=4, M=3136, K=1024 | B=2, M=1024, K=2048 | B=2, M=2048, K=4096 | B=2, M=4096, K=4096 |
| --------------------- | ----------------- | ----------------- | ------------------ | ------------------- | ------------------- | ------------------- | ------------------- |
| pytorch - fw          | 341.3             | 660.2             | 682.7              | 760.2               | 555.4               | 636.3               | 650.5               |
| triton  - fw          | 307.2             | 678.1             | 682.7              | 784.0               | 712.3               | 789.6               | 809.1               |
| pytorch - log - fw    | 384.0             | 696.9             | 702.2              | 777.9               | 537.2               | 541.6               | 543.9               |
| triton  - log - fw    | 307.2             | 696.9             | 702.2              | 796.4               | 744.7               | 799.2               | 814.1               |
| pytorch - fw+bw       | 133.6             | 203.1             | 204.0              | 229.9               | 193.9               | 211.1               | 215.3               |
| triton  - fw+bw       | 136.5             | 254.7             | 257.3              | 290.9               | 263.2               | 294.5               | 301.0               |
| pytorch - log - fw+bw | 149.9             | 252.1             | 252.1              | 289.6               | 234.1               | 251.6               | 254.5               |
| triton  - log - fw+bw | 136.5             | 257.3             | 258.7              | 291.7               | 265.3               | 295.2               | 301.3               |

### Fused linear layer

You can reproduce these numbers locally by running `python3 xformers/benchmarks/benchmark_triton_fused_linear_layer.py`. The units are TFlops/s. These results are for a nVidia V100, and PyTorch 1.9.
**As of September 2021, these Triton kernels are not competitive with PyTorch for Float32 computations**.

#### Squared ReLU

|                                          | B=8, M=256, K=512 | B=8, M=512, K=1024 | B=4, M=1024, K=1024 | B=2, M=2048, K=2048 | B=2, M=4096, K=4096 |
| ---------------------------------------- | ----------------- | ------------------ | ------------------- | ------------------- | ------------------- |
| pytorch - squared_relu -  bias - fw      | 6.3               | 12.4               | 12.3                | 17.1                | 19.0                |
| triton  - squared_relu -  bias - fw      | 13.8              | 18.9               | 18.9                | 21.9                | 21.7                |
| pytorch - squared_relu -  bias - fw+bw   | 4.0               | 7.6                | 7.7                 | 10.7                | 12.6                |
| triton  - squared_relu -  bias - fw+bw   | 8.4               | 13.5               | 13.3                | 15.9                | 16.8                |
| pytorch - squared_relu - no bias - fw    | 8.8               | 14.1               | 14.1                | 18.9                | 20.2                |
| triton  - squared_relu - no bias - fw    | 14.0              | 19.6               | 19.4                | 22.2                | 22.1                |
| pytorch - squared_relu - no bias - fw+bw | 4.6               | 8.3                | 8.3                 | 11.2                | 13.0                |
| triton  - squared_relu - no bias - fw+bw | 8.4               | 13.6               | 13.4                | 16.1                | 14.9                |

#### ReLU

|                                  | B=8, M=256, K=512 | B=8, M=512, K=1024 | B=4, M=1024, K=1024 | B=2, M=2048, K=2048 | B=2, M=4096, K=4096 |
| -------------------------------- | ----------------- | ------------------ | ------------------- | ------------------- | ------------------- |
| pytorch - relu -  bias - fw      | 8.6               | 13.9               | 13.9                | 18.7                | 19.9                |
| triton  - relu -  bias - fw      | 13.6              | 19.1               | 18.7                | 21.8                | 21.7                |
| pytorch - relu -  bias - fw+bw   | 5.3               | 9.6                | 9.6                 | 12.2                | 13.7                |
| triton  - relu -  bias - fw+bw   | 9.1               | 14.2               | 14.1                | 16.4                | 17.3                |
| pytorch - relu - no bias - fw    | 11.2              | 16.8               | 16.7                | 20.9                | 21.3                |
| triton  - relu - no bias - fw    | 14.0              | 19.3               | 19.5                | 22.2                | 22.0                |
| pytorch - relu - no bias - fw+bw | 6.3               | 10.6               | 10.6                | 12.9                | 14.1                |
| triton  - relu - no bias - fw+bw | 9.2               | 14.3               | 14.3                | 16.5                | 17.4                |


#### Leaky ReLU


|                                        | B=8, M=256, K=512 | B=8, M=512, K=1024 | B=4, M=1024, K=1024 | B=2, M=2048, K=2048 | B=2, M=4096, K=4096 |
| -------------------------------------- | ----------------- | ------------------ | ------------------- | ------------------- | ------------------- |
| pytorch - leaky_relu -  bias - fw      | 8.6               | 13.9               | 13.9                | 18.5                | 19.9                |
| triton  - leaky_relu -  bias - fw      | 13.6              | 19.0               | 18.9                | 21.7                | 21.7                |
| pytorch - leaky_relu -  bias - fw+bw   | 5.2               | 9.6                | 9.5                 | 12.2                | 13.7                |
| triton  - leaky_relu -  bias - fw+bw   | 9.0               | 14.2               | 14.0                | 16.7                | 17.7                |
| pytorch - leaky_relu - no bias - fw    | 11.2              | 16.7               | 16.7                | 20.8                | 21.1                |
| triton  - leaky_relu - no bias - fw    | 14.0              | 19.3               | 19.7                | 22.4                | 22.0                |
| pytorch - leaky_relu - no bias - fw+bw | 6.3               | 10.5               | 10.5                | 13.0                | 14.2                |
| triton  - leaky_relu - no bias - fw+bw | 9.0               | 14.3               | 14.1                | 16.9                | 17.8                |

#### GeLU

**As of September 2021, these Triton kernels are not competitive with PyTorch for the GeLU activation**.

|                                  | B=8, M=256, K=512 | B=8, M=512, K=1024 | B=4, M=1024, K=1024 | B=2, M=2048, K=2048 | B=2, M=4096, K=4096 |
| -------------------------------- | ----------------- | ------------------ | ------------------- | ------------------- | ------------------- |
| pytorch - gelu -  bias - fw      | 8.1               | 13.3               | 13.4                | 18.2                | 19.4                |
| triton  - gelu -  bias - fw      | 10.0              | 9.0                | 9.0                 | 16.3                | 15.7                |
| pytorch - gelu -  bias - fw+bw   | 5.0               | 9.2                | 9.2                 | 11.8                | 13.3                |
| triton  - gelu -  bias - fw+bw   | 3.8               | 5.5                | 5.5                 | 5.6                 | 4.6                 |
| pytorch - gelu - no bias - fw    | 10.4              | 15.6               | 15.9                | 20.4                | 20.8                |
| triton  - gelu - no bias - fw    | 11.4              | 11.9               | 11.9                | 17.8                | 15.9                |
| pytorch - gelu - no bias - fw+bw | 6.0               | 10.1               | 10.1                | 12.5                | 13.8                |
| triton  - gelu - no bias - fw+bw | 3.8               | 6.0                | 6.0                 | 5.7                 | 4.6                 |

#### No activation

|                                  | B=8, M=256, K=512 | B=8, M=512, K=1024 | B=4, M=1024, K=1024 | B=2, M=2048, K=2048 | B=2, M=4096, K=4096 |
| -------------------------------- | ----------------- | ------------------ | ------------------- | ------------------- | ------------------- |
| pytorch - None -  bias - fw      | 10.8              | 16.3               | 16.2                | 20.8                | 21.0                |
| triton  - None -  bias - fw      | 13.8              | 19.0               | 18.6                | 21.7                | 21.8                |
| pytorch - None -  bias - fw+bw   | 6.2               | 10.8               | 10.8                | 12.9                | 14.2                |
| triton  - None -  bias - fw+bw   | 9.8               | 15.2               | 15.1                | 17.6                | 19.0                |
| pytorch - None - no bias - fw    | 15.2              | 20.5               | 20.1                | 23.7                | 22.6                |
| triton  - None - no bias - fw    | 14.0              | 19.5               | 19.5                | 22.1                | 21.8                |
| pytorch - None - no bias - fw+bw | 7.5               | 12.1               | 12.1                | 13.8                | 14.5                |
| triton  - None - no bias - fw+bw | 9.9               | 15.4               | 15.2                | 17.8                | 19.1                |


## LRA

The code for this benchmark has been adapted from [this repository](https://github.com/mlpen/Nystromformer/tree/main/LRA). [A dedicated README is available here](xformers/benchmarks/LRA/README.md)

__Some results:__

| Attention                   | ListOps  | Text      | Retrieval | Image     | Pathfinder | *Avg*     | *Est. Gflops* | *Peak mem (mb)* |
| --------------------------- | -------- | --------- | --------- | --------- | ---------- | --------- | ------------- | --------------- |
| _Chance_                    | _10_     | _50_      | _50_      | _10_      | _50_       | _34_      | _0_           | _0_             |
| Standard                    | **37.5** | 62.66     | 79.24     | 38.69     | **70.37**  | **57.69** | 1.21          | 2291            |
| Nystromformer-128           | 36.29    | 63.24     | 78.18     | **42.86** | 67.49      | 57.61     | 0.62          | 383             |
| Favor-256 (redraw)          | 19.56    | 62.76     | **81.1**  | 36.09     | 67.23      | 53.35     | 0.49          | 445             |
| FourierMix                  | 36.29    | 60.72     | 76.41     | 36.53     | 54.07      | 52.8      | **0.17**      | **87**          |
| Linformer-seq/4 (no redraw) | 36.69    | 57.39     | 76.41     | 35.57     | 65.12      | 54.2      | 0.67          | 719             |
| Lambda                      | 19.76    | 62.47     | 79.11     | 35.04     | 49.74      | 49.224    | x             | 1023            |
| Orthoformer-32              | 27.42    | **63.96** | 77.96     | 34.5      | 67.11      | 54.19     | 0.187         | 155             |

- Contrary to the initial LRA proposal, __we use the same model architecture for all tasks (2 layers).__
- The training schedule for ListOps has been lengthened, while keeping it the fastest of all tasks, which reduces the seed dependence in the final accuracy figure.
- Estimated flops and peak memory are on the ListOps task, using 4 GPUs. Note that LRA is not completely well defined, in that hyperparameters and model architectures can vary (should the same architecture be used everywhere ? Similar hyperparams ?). This could be improved in the future, but in the meantime one should probably not read too much into small differences for some tasks, probably not meaningful.

_Note_: The estimated flops currently miss accounting for many operators, and are almost certainly an undercount. See issue [#154](https://github.com/fairinternal/xformers/issues/154)
