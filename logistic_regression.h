#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include <vector>
#include <cstdint>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

// Core mathematical functions for Logistic Regression
// All computations use float32
// Plain C++ loops - no vectorization

// Numerically stable sigmoid function
inline float sigmoid(float z) {
    // Clamp z to prevent overflow
    if (z > 20.0f) return 1.0f;
    if (z < -20.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-z));
}

// Predict probabilities for a batch of samples
// X: feature matrix (N x D), row-major layout
// w: weight vector (D,)
// b: bias scalar
// Returns: probability vector (N,)
inline void predict_proba(
    const float* X,      // Input: N x D matrix (row-major)
    const float* w,      // Input: D weights
    float b,             // Input: bias
    float* probs,        // Output: N probabilities
    size_t N,            // Number of samples
    size_t D             // Number of features
) {
    for (size_t i = 0; i < N; ++i) {
        float z = std::isfinite(b) ? b : 0.0f;  // Start with bias
        for (size_t j = 0; j < D; ++j) {
            float x_val = X[i * D + j];
            float w_val = w[j];
            if (std::isfinite(x_val) && std::isfinite(w_val)) {
                z += x_val * w_val;
            }
        }
        probs[i] = sigmoid(z);
    }
}

// Binary cross-entropy loss
// y: ground truth labels {0, 1}
// y_hat: predicted probabilities
// Returns: scalar loss value
inline float binary_cross_entropy(
    const int8_t* y,     // Input: N labels {0, 1}
    const float* y_hat,  // Input: N probabilities
    size_t N             // Number of samples
) {
    double loss = 0.0;  // Use double for accumulation to avoid overflow
    const float epsilon = 1e-7f;  // Prevent log(0)
    
    for (size_t i = 0; i < N; ++i) {
        float y_val = static_cast<float>(y[i]);
        float p = y_hat[i];
        
        // Handle NaN/Inf in predictions
        if (!std::isfinite(p)) {
            p = 0.5f;
        }
        
        // Clamp probabilities to [epsilon, 1-epsilon]
        if (p < epsilon) p = epsilon;
        if (p > 1.0f - epsilon) p = 1.0f - epsilon;
        
        double term = -(y_val * log(p) + (1.0 - y_val) * log(1.0 - p));
        
        // Skip NaN terms
        if (std::isfinite(term)) {
            loss += term;
        }
    }
    
    return static_cast<float>(loss / static_cast<double>(N));
}

// Compute gradients for a mini-batch
// X_batch: feature matrix (batch_size x D), row-major
// y_batch: labels (batch_size,)
// w: current weights (D,)
// b: current bias
// Returns: gradients for w and b
inline void compute_gradient(
    const float* X_batch,    // Input: batch_size x D matrix
    const int8_t* y_batch,   // Input: batch_size labels
    const float* w,          // Input: D weights
    float b,                 // Input: bias
    float* grad_w,           // Output: D gradient for weights
    float* grad_b,           // Output: gradient for bias
    size_t batch_size,       // Batch size
    size_t D                 // Number of features
) {
    // Initialize gradients to zero
    for (size_t j = 0; j < D; ++j) {
        grad_w[j] = 0.0f;
    }
    *grad_b = 0.0f;
    
    // Compute predictions for the batch
    std::vector<float> probs(batch_size);
    predict_proba(X_batch, w, b, probs.data(), batch_size, D);
    
    // Compute gradients
    for (size_t i = 0; i < batch_size; ++i) {
        float y_val = static_cast<float>(y_batch[i]);
        float error = probs[i] - y_val;
        
        // Gradient w.r.t. weights
        for (size_t j = 0; j < D; ++j) {
            grad_w[j] += error * X_batch[i * D + j];
        }
        
        // Gradient w.r.t. bias
        *grad_b += error;
    }
    
    // Average over batch
    float inv_batch = 1.0f / static_cast<float>(batch_size);
    for (size_t j = 0; j < D; ++j) {
        grad_w[j] *= inv_batch;
    }
    *grad_b *= inv_batch;
}

// Compute gradients for a mini-batch (OpenMP parallelized version)
// Parallelizes over samples within the batch
// Uses OpenMP reduction for thread-safe gradient accumulation
#ifdef _OPENMP
inline void compute_gradient_omp(
    const float* X_batch,    // Input: batch_size x D matrix
    const int8_t* y_batch,   // Input: batch_size labels
    const float* w,          // Input: D weights
    float b,                 // Input: bias
    float* grad_w,           // Output: D gradient for weights
    float* grad_b,           // Output: gradient for bias
    size_t batch_size,       // Batch size
    size_t D                 // Number of features
) {
    // Initialize gradients to zero
    for (size_t j = 0; j < D; ++j) {
        grad_w[j] = 0.0f;
    }
    *grad_b = 0.0f;
    
    // Compute predictions for the batch
    std::vector<float> probs(batch_size);
    predict_proba(X_batch, w, b, probs.data(), batch_size, D);
    
    // Parallel gradient computation over samples
    // Use reduction for grad_b, and manual accumulation for grad_w array
    float grad_b_sum = 0.0f;
    
    #pragma omp parallel reduction(+:grad_b_sum)
    {
        // Private gradient buffer for weights (per thread)
        std::vector<float> grad_w_local(D, 0.0f);
        
        // Parallel loop over samples
        #pragma omp for schedule(static)
        for (size_t i = 0; i < batch_size; ++i) {
            float y_val = static_cast<float>(y_batch[i]);
            float error = probs[i] - y_val;
            
            // Accumulate gradients in thread-local buffer
            for (size_t j = 0; j < D; ++j) {
                grad_w_local[j] += error * X_batch[i * D + j];
            }
            grad_b_sum += error;
        }
        
        // Combine thread-local grad_w into global (use critical section for array reduction)
        #pragma omp critical
        {
            for (size_t j = 0; j < D; ++j) {
                grad_w[j] += grad_w_local[j];
            }
        }
    }
    
    *grad_b = grad_b_sum;
    
    // Average over batch
    float inv_batch = 1.0f / static_cast<float>(batch_size);
    for (size_t j = 0; j < D; ++j) {
        grad_w[j] *= inv_batch;
    }
    *grad_b *= inv_batch;
}
#endif // _OPENMP

// Predict labels with explicit branching (for branch prediction experiments)
// X: feature matrix (N x D), row-major
// w: weight vector (D,)
// b: bias scalar
// threshold: decision threshold (default 0.5)
// Returns: predicted labels {0, 1}
inline void predict_label(
    const float* X,      // Input: N x D matrix
    const float* w,      // Input: D weights
    float b,             // Input: bias
    int8_t* labels,      // Output: N labels
    size_t N,            // Number of samples
    size_t D,            // Number of features
    float threshold = 0.5f  // Decision threshold
) {
    std::vector<float> probs(N);
    predict_proba(X, w, b, probs.data(), N, D);
    
    // Explicit branching for branch prediction analysis
    for (size_t i = 0; i < N; ++i) {
        if (probs[i] > threshold) {
            labels[i] = 1;
        } else {
            labels[i] = 0;
        }
    }
}

#endif // LOGISTIC_REGRESSION_H
