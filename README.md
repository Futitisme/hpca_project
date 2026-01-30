# CPU Logistic Regression Model

This folder contains all files needed to run the CPU implementation of logistic regression.

## Files

| File | Description |
|------|-------------|
| `experiment.cpp` | Main program with training, evaluation, and benchmarking |
| `logistic_regression.h` | Core math functions (sigmoid, gradient, predict) |
| `npy_loader.h` | Loader for numpy .npy files |
| `Makefile` | Build configuration |
| `CPU_MODEL_EXPLANATION.md` | Algorithm and implementation details |

## Build

```bash
make
```

## Run Experiments

Data files must be in `../data/` relative to this folder (or specify with `--data-dir`).

Required data files:
- `X_train_norm.npy`, `y_train.npy`
- `X_val_norm.npy`, `y_val.npy`
- `X_test_norm.npy`, `y_test.npy`

### Example Commands

```bash
# Small experiment (50K samples)
./experiment --N 50000 --batch-size 512 --epochs 5 --lr 0.01 --threads 1 --output exp1

# Full dataset experiment
./experiment --batch-size 4096 --epochs 5 --lr 0.01 --threads 1 --output exp_full

# With custom data directory
./experiment --batch-size 4096 --epochs 5 --lr 0.01 --data-dir /path/to/data --output exp
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--N` | Number of training samples (0 = all) | 0 |
| `--batch-size` | Mini-batch size | 4096 |
| `--epochs` | Number of training epochs | 20 |
| `--lr` | Learning rate | 0.1 |
| `--threads` | Number of OpenMP threads | 1 |
| `--data-dir` | Path to data directory | ../data |
| `--output` | Output file prefix | experiment |
| `--quiet` | Suppress verbose output | false |

## Output

The program generates:
- `{output}_summary.csv` - Final metrics
- `{output}_epochs_t{threads}.csv` - Per-epoch metrics
