# GPU Logistic Regression Model (CUDA)

This folder contains all files required to run the **GPU (CUDA) implementation of logistic regression**, designed to be **algorithmically equivalent** to the CPU version and suitable for **fair performance comparison** in an HPC context.

The implementation follows the same optimization algorithm, numerical precision, initialization strategy, and convergence behavior as the CPU baseline, while exploiting GPU parallelism via CUDA and cuBLAS.

---

## Files

| File | Description |
|------|-------------|
| `gpu_logreg.cu` | Main CUDA program with training, evaluation, and benchmarking |
| `cnpy/cnpy.cpp` | NumPy `.npy` file loader implementation |
| `cnpy/cnpy.h` | Header for NumPy loader |
| `README.md` | This file |

---

## Build

The GPU implementation requires:
- NVIDIA CUDA Toolkit
- cuBLAS
- zlib (for `.npy` loading via `cnpy`)

Example build command:

```bash
nvcc -O3 -std=c++17 gpu_logreg.cu cnpy/cnpy.cpp -Icnpy -lz -lcublas -o gpu_logreg
```

Ensure that `nvcc` is available and that your GPU supports CUDA.

---

## Run Experiments

The program expects input data in NumPy `.npy` format. Paths can be provided explicitly via command-line arguments.

### Required data files

- `X_train_norm.npy`, `y_train.npy`
- `X_test_norm.npy`, `y_test.npy`

(Optional validation files can also be provided.)

---

## Example Commands

### Small experiment (50k samples)

```bash
./gpu_logreg \
  --x_train X_train_norm.npy --y_train y_train.npy \
  --x_test X_test_norm.npy --y_test y_test.npy \
  --max-samples 50000 \
  --epochs 5 --batch-size 512 --lr 0.1 --seed 42 \
  --log-loss
```

### Medium experiment (500k samples)

```bash
./gpu_logreg \
  --x_train X_train_norm.npy --y_train y_train.npy \
  --x_test X_test_norm.npy --y_test y_test.npy \
  --max-samples 500000 \
  --epochs 10 --batch-size 1024 --lr 0.1 --seed 42 \
  --log-loss
```

### Full dataset experiment

```bash
./gpu_logreg \
  --x_train X_train_norm.npy --y_train y_train.npy \
  --x_test X_test_norm.npy --y_test y_test.npy \
  --epochs 5 --batch-size 4096 --lr 0.1 --seed 42 \
  --log-loss
```

---

## Inference Benchmarking

The GPU implementation includes **inference latency and throughput benchmarking**.

Example:

```bash
./gpu_logreg \
  --x_train X_train_norm.npy --y_train y_train.npy \
  --x_test X_test_norm.npy --y_test y_test.npy \
  --epochs 5 --batch-size 4096 --lr 0.1 --seed 42 \
  --infer-benchmark --infer-warmup 3 --infer-repeats 10
```

This reports:
- Average inference time per batch
- Latency per sample
- Inference throughput (samples/sec)

---

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--x_train` | Training feature matrix (`.npy`) | `X_train_norm.npy` |
| `--y_train` | Training labels (`.npy`) | `y_train.npy` |
| `--x_test` | Test feature matrix | — |
| `--y_test` | Test labels | — |
| `--max-samples` | Limit number of training samples (0 = all) | `0` |
| `--batch-size` | Mini-batch size | `4096` |
| `--epochs` | Number of training epochs | `20` |
| `--lr` | Learning rate | `0.1` |
| `--seed` | Random seed (init + shuffling) | `42` |
| `--log-loss` | Print cross-entropy loss per epoch | off |
| `--benchmark` | Output CSV-style training timing only | off |
| `--infer-benchmark` | Run inference benchmark | off |
| `--infer-warmup` | Inference warm-up runs | `3` |
| `--infer-repeats` | Inference repetitions | `10` |
| `--quiet` | Suppress verbose output | off |

---

## Output

The program prints:
- Per-epoch training time and throughput
- Optional cross-entropy loss
- Final test accuracy
- Optional inference latency and throughput

Timing measurements are performed using **CUDA events** and include **training compute only**, excluding data loading and preprocessing.

---

## Notes on Algorithmic Parity

The GPU implementation is designed to match the CPU implementation exactly:

- Same logistic regression formulation
- Same float32 precision
- Same sigmoid clamping
- Same weight initialization (normal distribution with fixed seed)
- Same mini-batch gradient descent
- Same per-epoch shuffling strategy

As a result, observed performance differences between CPU and GPU executions reflect **hardware and architectural characteristics**, not algorithmic changes.

