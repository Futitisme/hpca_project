# GPU Logistic Regression – Code Explanation (`gpu_logreg.cu`)

This document explains the structure, design choices, and execution flow of the CUDA-based logistic regression implementation. The goal of this code is to provide a GPU-accelerated implementation that is algorithmically identical to the CPU baseline.

---

## High-Level Overview

The program implements logistic regression trained with mini-batch gradient descent. The computation is offloaded to the GPU using:

- **CUDA kernels** for elementwise operations (sigmoid, loss, updates)
- **cuBLAS** for high-performance matrix–vector multiplications
- **CUDA events** for precise timing of compute-only regions

The code supports:
- Training
- Optional BCE loss logging per epoch
- Accuracy evaluation
- Inference latency and throughput benchmarking

All numerical choices (precision, initialization, clamping) are matched to the CPU implementation.

---

## Data Loading and Preparation

### NumPy File Handling (`cnpy`)

Training and evaluation data are loaded from `.npy` files using cnpy. Since NumPy arrays may come in different dtypes, the loader:

- Converts all feature values to `float32`
- Converts labels to `int8_t` with values `{0,1}`

This ensures a consistent and GPU-friendly representation.

### Memory Layout Conversion

- Input data is loaded as **row-major (N × D)**
- It is then converted to **column-major (D × N)**

This layout is chosen because **cuBLAS expects column-major matrices**, allowing the use of `cublasSgemv` without extra transpositions.

---

## Core GPU Computation

### Forward Pass

For each mini-batch, logits are computed as:

```
logits = X_batchᵀ · w
```

This operation dominates runtime and is implemented using:

- `cublasSgemv` for optimized GPU matrix–vector multiplication

### Sigmoid and Gradient Computation

Custom CUDA kernels handle:

- Sigmoid activation with numerical clamping (`[-20, 20]`)
- Error term computation: `sigmoid(logit + b) − y`
- Optional BCE loss calculation

These kernels are lightweight and fully parallelized over batch elements.

### Gradient Reduction and Parameter Updates

- Weight gradients are computed via another `cublasSgemv`
- Gradients are averaged by batch size
- Weights and bias are updated using simple CUDA kernels

This mirrors the exact update logic of the CPU version.

---

## Mini-Batch Handling

Each epoch:

1. Indices are shuffled on the host using a fixed seed
2. Shuffled indices are copied to the device
3. A custom kernel gathers mini-batches into contiguous GPU buffers

This keeps the behavior across CPU and GPU runs.

---

## Loss Computation (Optional)

When `--log-loss` is enabled:

- Per-sample Binary Cross-Entropy loss is computed on the GPU
- A parallel reduction sums the loss
- The average loss per epoch is reported

This path is excluded from benchmark-only runs to avoid contaminating timing results.

---

## Timing and Benchmarking

### Training Timing

- Training time per epoch is measured using **CUDA events**
- Only compute time is measured (no data loading or host-device transfers)
- Reported metrics:
  - Time per epoch (ms)
  - Training throughput (samples/sec)

### Inference Benchmark

With `--infer-benchmark`, the code measures:

- Forward-pass latency only (matrix–vector multiply)
- No host copies inside the timed region

Reported metrics:
- Average inference time per run
- Latency per sample
- Inference throughput (samples/sec)

---

## Accuracy Evaluation

Accuracy is computed in a correctness-oriented path:

- Logits are computed on the GPU
- Results are copied to the CPU
- Sigmoid + thresholding is applied on the host

This is intentionally not optimized for speed, as it is excluded from performance benchmarks.

---

## Design Principles

The implementation follows several key principles:

- **Algorithmic parity** with CPU version
- **Deterministic behavior** using fixed seeds
- **Separation of compute and measurement**
- **cuBLAS for heavy linear algebra, custom kernels for simple ops**

As a result, observed performance differences show hardware and architectural effects, not changes in the learning algorithm.




