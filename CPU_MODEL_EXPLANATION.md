# CPU Logistic Regression Model - Technical Overview

## What is Logistic Regression

Logistic regression is a binary classification algorithm. Given input features X, it predicts the probability that a sample belongs to class 1 (malicious traffic) vs class 0 (benign traffic).

The model learns a weight vector w (one weight per feature) and a bias term b.


## Core Mathematical Operations

### 1. Linear Combination

For each sample, we compute a weighted sum of features:

```
z = b + sum(x[j] * w[j]) for j = 0 to D-1
```

where D = 46 (number of features in our dataset).


### 2. Sigmoid Activation

The linear output z is transformed into a probability using the sigmoid function:

```cpp
float sigmoid(float z) {
    if (z > 20.0f) return 1.0f;
    if (z < -20.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-z));
}
```

This maps any real number to the range (0, 1). The clamping at +/-20 prevents numerical overflow.


### 3. Prediction

For a batch of N samples:

```cpp
for (size_t i = 0; i < N; ++i) {
    float z = b;
    for (size_t j = 0; j < D; ++j) {
        z += X[i * D + j] * w[j];
    }
    probs[i] = sigmoid(z);
}
```

The final class label is determined by threshold: if prob > 0.5, predict class 1, else class 0.


## Training Algorithm

We use mini-batch gradient descent to learn the optimal weights.

### Loss Function: Binary Cross-Entropy

```
loss = -mean( y * log(p) + (1-y) * log(1-p) )
```

where y is the true label (0 or 1) and p is the predicted probability.


### Gradient Computation

For each sample in a mini-batch, we compute:

```
error = predicted_prob - true_label
grad_w[j] += error * x[j]   (for each feature j)
grad_b += error
```

Then average over the batch:

```cpp
for (size_t i = 0; i < batch_size; ++i) {
    float error = probs[i] - y_batch[i];
    for (size_t j = 0; j < D; ++j) {
        grad_w[j] += error * X_batch[i * D + j];
    }
    grad_b += error;
}
// Average gradients
for (size_t j = 0; j < D; ++j) {
    grad_w[j] /= batch_size;
}
grad_b /= batch_size;
```


### Parameter Update

After computing gradients, we update weights using gradient descent:

```cpp
for (size_t j = 0; j < D; ++j) {
    w[j] -= learning_rate * grad_w[j];
}
b -= learning_rate * grad_b;
```


## Training Loop Structure

One epoch processes all training samples in mini-batches:

```
for each epoch:
    shuffle training indices
    for each mini-batch:
        1. Extract batch of samples
        2. Compute predictions (forward pass)
        3. Compute gradients (backward pass)
        4. Update weights
    end
    Evaluate on validation set
end
```


## Data Layout

The feature matrix X is stored in row-major order:

```
X[N][D] where:
  - N = number of samples
  - D = 46 features
  - X[i * D + j] accesses feature j of sample i
```

Labels y are stored as int8: 0 = benign, 1 = malicious.


## Computational Complexity

Per mini-batch of size B with D features:
- Forward pass: O(B * D) multiplications and additions
- Gradient computation: O(B * D) multiplications and additions
- Weight update: O(D) operations

Total per epoch with N samples: O(N * D)


## Memory Access Pattern

The inner loop accesses:
- X sequentially (good cache behavior for row-major layout)
- w repeatedly (fits in L1 cache since D=46, only 184 bytes)

This makes the computation memory-bound for large N, as each sample requires loading 46 floats from main memory.


## CPU Implementation Details

The CPU implementation of logistic regression was developed in plain C++ to serve as the baseline for performance comparison with the GPU version. All computations are performed in single-precision floating point (float32), and the same initialization strategy, learning rate, sigmoid clamping, and mini-batch gradient descent logic are used as in the GPU implementation to ensure algorithmic parity.

The feature matrix is stored in row-major format (X[N][D]) which provides sequential memory access when iterating over features within a sample. This layout is cache-friendly for the CPU architecture since each sample's features are stored contiguously. The weight vector w has only 46 elements (184 bytes), which fits entirely in L1 cache and is reused across all samples in a batch.

All core operations are implemented as explicit loops without external library dependencies. The forward pass computes the dot product of each sample with the weight vector, adds the bias, and applies the sigmoid function. Gradient computation accumulates partial gradients across all samples in a mini-batch using a single pass through the data, then averages the result. Weight updates are applied immediately after processing each mini-batch.

To maintain algorithmic parity with the GPU version, training samples are shuffled at the beginning of each epoch using a fixed random seed (default 42). Mini-batches are then extracted by copying the relevant samples into a contiguous batch buffer based on the shuffled indices, rather than accessing them in-place with indirect indexing.

An optional OpenMP-parallelized gradient computation is available for multi-threaded execution. The parallel version uses thread-local gradient buffers to avoid race conditions, with a reduction step to combine results. Each thread processes a subset of samples in the batch, and the final gradient is averaged over the full batch size.

Training time measurements are taken using high-resolution chrono timers and include only the compute phase (gradient computation and weight updates), excluding data loading and preprocessing. The first epoch is treated as warm-up and excluded from timing statistics. Inference performance is measured separately using repeated forward passes after a warm-up phase to obtain stable latency and throughput estimates.


## Key Implementation Files

- `logistic_regression.h` - Core math functions (sigmoid, predict, gradient, OpenMP gradient)
- `experiment.cpp` - Training loop with timing and metrics
- `npy_loader.h` - Loads preprocessed numpy arrays


## Experiment Parameters Used

| Parameter | Value |
|-----------|-------|
| Features (D) | 46 |
| Learning rate | 0.01 |
| Batch sizes | 512, 1024, 4096 |
| Epochs | 5-32 |
| Dataset sizes (N) | 50K to 5.5M |
