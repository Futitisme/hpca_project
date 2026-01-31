// gpu_logreg.cu
// Logistic Regression (binary) with mini-batch gradient descent on GPU using CUDA.
// CPU-parity version adds:
//   - BCE loss per epoch (--log-loss)
//   - inference latency/throughput benchmarking (--infer-benchmark)
#include <cnpy.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do {                                     \
  cudaError_t err = (call);                                       \
  if (err != cudaSuccess) {                                       \
    throw std::runtime_error(std::string("CUDA error: ") +        \
      cudaGetErrorString(err) + " @ " + __FILE__ + ":" +          \
      std::to_string(__LINE__));                                  \
  }                                                               \
} while (0)

#define CUBLAS_CHECK(call) do {                                   \
  cublasStatus_t st = (call);                                     \
  if (st != CUBLAS_STATUS_SUCCESS) {                              \
    throw std::runtime_error(std::string("cuBLAS error @ ") +     \
      __FILE__ + ":" + std::to_string(__LINE__));                 \
  }                                                               \
} while (0)

static inline int64_t prod(const std::vector<size_t>& s) {
  int64_t p = 1;
  for (size_t v : s) p *= (int64_t)v;
  return p;
}

// Convert loaded npy data to float32 vector (row-major)
static std::vector<float> load_npy_as_f32_rowmajor(
  const std::string& path,
  std::vector<size_t>& shape_out
) {
  cnpy::NpyArray arr = cnpy::npy_load(path);
  shape_out = arr.shape;
  const int64_t n = prod(shape_out);
  std::vector<float> out(n);
  if (arr.word_size == sizeof(float)) {
    const float* p = arr.data<float>();
    std::memcpy(out.data(), p, n * sizeof(float));
  } else if (arr.word_size == sizeof(double)) {
    const double* p = arr.data<double>();
    for (int64_t i = 0; i < n; i++) out[i] = (float)p[i];
  } else if (arr.word_size == sizeof(int64_t)) {
    const int64_t* p = arr.data<int64_t>();
    for (int64_t i = 0; i < n; i++) out[i] = (float)p[i];
  } else if (arr.word_size == sizeof(int32_t)) {
    const int32_t* p = arr.data<int32_t>();
    for (int64_t i = 0; i < n; i++) out[i] = (float)p[i];
  } else if (arr.word_size == sizeof(uint8_t)) {
    const uint8_t* p = arr.data<uint8_t>();
    for (int64_t i = 0; i < n; i++) out[i] = (float)p[i];
  } else if (arr.word_size == sizeof(int8_t)) {
    const int8_t* p = arr.data<int8_t>();
    for (int64_t i = 0; i < n; i++) out[i] = (float)p[i];
  } else {
    throw std::runtime_error("Unsupported dtype word_size in " + path);
  }

  return out;
}

// Load labels as int8_t {0,1}
static std::vector<int8_t> load_npy_labels_as_i8(
  const std::string& path,
  std::vector<size_t>& shape_out
) {
  cnpy::NpyArray arr = cnpy::npy_load(path);
  shape_out = arr.shape;
  const int64_t n = prod(shape_out);
  std::vector<int8_t> out(n);
  auto to01 = [](double v) -> int8_t { return (v >= 0.5) ? int8_t{1} : int8_t{0}; };
  if (arr.word_size == sizeof(uint8_t)) {
    const uint8_t* p = arr.data<uint8_t>();
    for (int64_t i = 0; i < n; i++) out[i] = (p[i] ? 1 : 0);
  } else if (arr.word_size == sizeof(int8_t)) {
    const int8_t* p = arr.data<int8_t>();
    for (int64_t i = 0; i < n; i++) out[i] = (p[i] ? 1 : 0);
  } else if (arr.word_size == sizeof(int32_t)) {
    const int32_t* p = arr.data<int32_t>();
    for (int64_t i = 0; i < n; i++) out[i] = (p[i] ? 1 : 0);
  } else if (arr.word_size == sizeof(int64_t)) {
    const int64_t* p = arr.data<int64_t>();
    for (int64_t i = 0; i < n; i++) out[i] = (p[i] ? 1 : 0);
  } else if (arr.word_size == sizeof(float)) {
    const float* p = arr.data<float>();
    for (int64_t i = 0; i < n; i++) out[i] = to01(p[i]);
  } else if (arr.word_size == sizeof(double)) {
    const double* p = arr.data<double>();
    for (int64_t i = 0; i < n; i++) out[i] = to01(p[i]);
  } else {
    throw std::runtime_error("Unsupported label dtype word_size in " + path);
  }

  return out;
}

// X row-major (N x D) -> X col-major (D x N)
static std::vector<float> rowmajor_ND_to_colmajor_DN(const std::vector<float>& X_row, int N, int D) {
  std::vector<float> X_col((int64_t)N * D);
  for (int i = 0; i < N; i++) {
    const float* src = &X_row[(int64_t)i * D];
    for (int j = 0; j < D; j++) {
      X_col[(int64_t)j + (int64_t)i * D] = src[j];
    }
  }
  return X_col;
}

// --- GPU math kernels (CPU-parity) ---

__device__ __forceinline__ float sigmoid_clamped(float z) {
  if (z > 20.0f) return 1.0f;
  if (z < -20.0f) return 0.0f;
  return 1.0f / (1.0f + expf(-z));
}

// diff[i] = sigmoid(logit[i] + b) - y[i]
__global__ void sigmoid_and_diff_bias_kernel(
  const float* logits,
  const int8_t* y,
  float* diff,
  float b,
  int B
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < B) {
    float s = sigmoid_clamped(logits[idx] + b);
    diff[idx] = s - (float)y[idx];
  }
}

// BCE loss per sample: -( y log(p) + (1-y) log(1-p) )
__global__ void bce_loss_kernel(
  const float* logits,
  const int8_t* y,
  float* loss_out,
  float b,
  int B
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < B) {
    float p = sigmoid_clamped(logits[idx] + b);
    // match typical CPU epsilon protection
    const float eps = 1e-8f;
    if (p < eps) p = eps;
    if (p > 1.0f - eps) p = 1.0f - eps;

    float yt = (float)y[idx];
    loss_out[idx] = -(yt * logf(p) + (1.0f - yt) * logf(1.0f - p));
  }
}

__global__ void scale_grad_kernel(float* grad_w, float invB, int D) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < D) grad_w[j] *= invB;
}

__global__ void update_w_kernel(float* w, const float* grad_w, float lr, int D) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < D) w[j] -= lr * grad_w[j];
}

// Gather a mini-batch from X(DxN col-major) and y(N) using shuffled indices.
// Output X_batch(DxB col-major) and y_batch(B).
__global__ void gather_batch_DxB(
  const float* X_DxN,
  const int8_t* y_N,
  const int* indices_N,
  int start,
  int B,
  float* X_DxB,
  int8_t* y_B,
  int D
) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int total = D * B;
  if (tid < total) {
    int j = tid % D;
    int i = tid / D;
    int idx = indices_N[start + i];
    X_DxB[j + (int64_t)i * D] = X_DxN[j + (int64_t)idx * D];
  }

  if (tid < B) {
    int idx = indices_N[start + tid];
    y_B[tid] = y_N[idx];
  }
}

__global__ void reduce_sum_kernel(const float* x, float* out, int n) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;

  float v = (i < n) ? x[i] : 0.0f;
  sdata[tid] = v;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  if (tid == 0) out[blockIdx.x] = sdata[0];
}

static float device_sum(const float* d_x, int n) {
  const int threads = 256;
  int blocks = (n + threads - 1) / threads;

  float* d_part = nullptr;
  CUDA_CHECK(cudaMalloc(&d_part, blocks * sizeof(float)));

  reduce_sum_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_x, d_part, n);
  CUDA_CHECK(cudaGetLastError());

  std::vector<float> h_part(blocks);
  CUDA_CHECK(cudaMemcpy(h_part.data(), d_part, blocks * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_part));

  double sum = 0.0;
  for (float v : h_part) sum += v;
  return (float)sum;
}

// Host-side weight init to match CPU (Normal(0,0.01), seed)
static void initialize_weights_host(std::vector<float>& w, int seed) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(0.0f, 0.01f);
  for (auto& v : w) v = dist(gen);
}

struct Args {
  std::string x_train = "X_train_norm.npy";
  std::string y_train = "y_train.npy";
  std::string x_val   = "";
  std::string y_val   = "";
  std::string x_test  = "";
  std::string y_test  = "";
  int epochs = 20;
  int batch_size  = 4096;
  float lr   = 0.1f;
  int seed = 42;
  int max_samples = 0;     // 0 = use all
  bool verbose = true;
  bool benchmark = false;

  // new:
  bool log_loss = false;
  bool infer_benchmark = false;
  int infer_warmup = 3;
  int infer_repeats = 10;
};

static Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; i++) {
    std::string k = argv[i];
    auto need = [&](const std::string& name) {
      if (i + 1 >= argc) throw std::runtime_error("Missing value for " + name);
      return std::string(argv[++i]);
    };

    if (k == "--x_train") a.x_train = need(k);
    else if (k == "--y_train") a.y_train = need(k);
    else if (k == "--x_val") a.x_val = need(k);
    else if (k == "--y_val") a.y_val = need(k);
    else if (k == "--x_test") a.x_test = need(k);
    else if (k == "--y_test") a.y_test = need(k);
    else if (k == "--epochs") a.epochs = std::stoi(need(k));
    else if (k == "--batch-size" || k == "--batch") a.batch_size = std::stoi(need(k));
    else if (k == "--lr") a.lr = std::stof(need(k));
    else if (k == "--seed") a.seed = std::stoi(need(k));
    else if (k == "--max-samples") a.max_samples = std::stoi(need(k));
    else if (k == "--quiet") a.verbose = false;
    else if (k == "--benchmark") { a.benchmark = true; a.verbose = false; }

    // new:
    else if (k == "--log-loss") a.log_loss = true;
    else if (k == "--infer-benchmark") a.infer_benchmark = true;
    else if (k == "--infer-warmup") a.infer_warmup = std::stoi(need(k));
    else if (k == "--infer-repeats") a.infer_repeats = std::stoi(need(k));
    else {
      throw std::runtime_error("Unknown arg: " + k);
    }
  }
  return a;
}

// Evaluate accuracy for dataset (X_col is D x N col-major, y is N int8).
// Note: this is correctness-oriented, not a latency benchmark (it copies to CPU).
static float eval_accuracy(
  cublasHandle_t handle,
  const float* d_X_col,
  const int8_t* d_y,
  int N, int D,
  const float* d_w,
  float b,
  int batch_size
) {
  float* d_logits = nullptr;
  CUDA_CHECK(cudaMalloc(&d_logits, (int64_t)batch_size * sizeof(float)));
  std::vector<float> h_logits;
  h_logits.reserve(batch_size);
  int correct = 0;
  int total = 0;
  const float one = 1.0f, zero = 0.0f;
  auto sigmoid_host = [](float z) -> float {
    if (z > 20.0f) return 1.0f;
    if (z < -20.0f) return 0.0f;
    return 1.0f / (1.0f + std::exp(-z));
  };

  for (int start = 0; start < N; start += batch_size) {
    int B = std::min(batch_size, N - start);
    const float* d_X_batch = d_X_col + (int64_t)start * D;
    CUBLAS_CHECK(cublasSgemv(
      handle,
      CUBLAS_OP_T,
      D, B,
      &one,
      d_X_batch, D,
      d_w, 1,
      &zero,
      d_logits, 1
    ));

    h_logits.resize(B);
    CUDA_CHECK(cudaMemcpy(h_logits.data(), d_logits, (int64_t)B * sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<int8_t> h_y(B);
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y + start, (int64_t)B * sizeof(int8_t), cudaMemcpyDeviceToHost));
    for (int i = 0; i < B; i++) {
      float p = sigmoid_host(h_logits[i] + b);
      int pred = (p > 0.5f) ? 1 : 0;
      int yt = (h_y[i] ? 1 : 0);
      correct += (pred == yt);
    }
    total += B;
  }

  CUDA_CHECK(cudaFree(d_logits));
  return total ? (float)correct / (float)total : 0.0f;
}

// Inference benchmark: times forward pass only (X^T w), no host copies inside timing.
static void run_inference_benchmark(
  cublasHandle_t handle,
  const float* d_X_col, // D x N col-major
  int N, int D,
  const float* d_w,
  float b,
  int batch_size,
  int warmup,
  int repeats
) {
  if (N <= 0) throw std::runtime_error("Inference N must be > 0");
  if (batch_size <= 0) throw std::runtime_error("Inference batch_size must be > 0");
  warmup = std::max(0, warmup);
  repeats = std::max(1, repeats);
  const int Bmax = std::min(batch_size, N);
  float* d_logits = nullptr;
  CUDA_CHECK(cudaMalloc(&d_logits, (int64_t)Bmax * sizeof(float)));
  const float one = 1.0f, zero = 0.0f;
  auto do_one_pass = [&]() {
    for (int start = 0; start < N; start += batch_size) {
      int B = std::min(batch_size, N - start);
      const float* d_X_batch = d_X_col + (int64_t)start * D;

      // logits = X_batch^T * w
      CUBLAS_CHECK(cublasSgemv(
        handle,
        CUBLAS_OP_T,
        D, B,
        &one,
        d_X_batch, D,
        d_w, 1,
        &zero,
        d_logits, 1
      ));

      // Include bias + sigmoid? For pure matmul timing, no.
      // If you want full "predict proba", you'd add a kernel here.
      (void)b;
    }
  };

  // Warmup
  for (int i = 0; i < warmup; i++) {
    do_one_pass();
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));
  CUDA_CHECK(cudaEventRecord(ev_start));
  for (int r = 0; r < repeats; r++) {
    do_one_pass();
  }
  CUDA_CHECK(cudaEventRecord(ev_stop));
  CUDA_CHECK(cudaEventSynchronize(ev_stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
  float avg_ms = ms / (float)repeats;
  float latency_ms_per_sample = avg_ms / (float)N;
  float samples_per_sec = (avg_ms > 0.0f) ? ((float)N / (avg_ms / 1000.0f)) : 0.0f;
  std::cout << "inference_N=" << N
            << " D=" << D
            << " batch_size=" << batch_size
            << " repeats=" << repeats
            << " avg_time_ms=" << avg_ms
            << " latency_ms_per_sample=" << latency_ms_per_sample
            << " samples_per_sec=" << samples_per_sec
            << "\n";

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaFree(d_logits));
}

int main(int argc, char** argv) {
  int dev = 0;
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  std::cout << "CUDA device: " << dev << " | " << prop.name << "\n";

  try {
    Args args = parse_args(argc, argv);

    if (args.epochs <= 0) throw std::runtime_error("--epochs must be > 0");
    if (args.batch_size <= 0) throw std::runtime_error("--batch-size must be > 0");
    if (args.lr <= 0.0f) throw std::runtime_error("--lr must be > 0");
    if (args.seed < 0) throw std::runtime_error("--seed must be >= 0");
    if (args.max_samples < 0) throw std::runtime_error("--max-samples must be >= 0");

    // ---- Load train ----
    std::vector<size_t> xshape, yshape;
    auto X_train_row = load_npy_as_f32_rowmajor(args.x_train, xshape);
    auto y_train = load_npy_labels_as_i8(args.y_train, yshape);

    if (xshape.size() != 2) throw std::runtime_error("X_train must be 2D (N x D)");
    int N_full = (int)xshape[0];
    int D = (int)xshape[1];
    if ((int64_t)y_train.size() != N_full) throw std::runtime_error("y_train length must match X_train rows");
    if (D <= 0) throw std::runtime_error("D must be > 0");

    int N = N_full;
    if (args.max_samples > 0) N = std::min(N_full, args.max_samples);

    if (args.verbose) {
      std::cout << "Train: N=" << N << " / " << N_full << " D=" << D << "\n";
      std::cout << "Training: epochs=" << args.epochs
                << " batch_size=" << args.batch_size
                << " lr=" << args.lr
                << " seed=" << args.seed
                << " max_samples=" << args.max_samples
                << "\n";
    }

    // If subsampling, truncate host arrays to N
    if (N != N_full) {
      std::vector<float> X_trunc((int64_t)N * D);
      std::memcpy(X_trunc.data(), X_train_row.data(), (int64_t)N * D * sizeof(float));
      X_train_row.swap(X_trunc);

      std::vector<int8_t> y_trunc(N);
      std::memcpy(y_trunc.data(), y_train.data(), (int64_t)N * sizeof(int8_t));
      y_train.swap(y_trunc);
    }

    // Convert X to col-major (D x N)
    auto X_train_col = rowmajor_ND_to_colmajor_DN(X_train_row, N, D);
    X_train_row.clear();
    X_train_row.shrink_to_fit();

    // ---- Optional val/test ----
    std::vector<float> X_val_col, X_test_col;
    std::vector<int8_t> y_val, y_test;
    int Nval = 0, Ntest = 0;

    if (!args.x_val.empty() && !args.y_val.empty()) {
      std::vector<size_t> xs, ys;
      auto Xv_row = load_npy_as_f32_rowmajor(args.x_val, xs);
      y_val = load_npy_labels_as_i8(args.y_val, ys);
      if (xs.size() != 2) throw std::runtime_error("X_val must be 2D");
      Nval = (int)xs[0];
      int Dv = (int)xs[1];
      if (Dv != D) throw std::runtime_error("X_val D != X_train D");
      if ((int)y_val.size() != Nval) throw std::runtime_error("y_val length mismatch");
      X_val_col = rowmajor_ND_to_colmajor_DN(Xv_row, Nval, D);
    }

    if (!args.x_test.empty() && !args.y_test.empty()) {
      std::vector<size_t> xs, ys;
      auto Xt_row = load_npy_as_f32_rowmajor(args.x_test, xs);
      y_test = load_npy_labels_as_i8(args.y_test, ys);
      if (xs.size() != 2) throw std::runtime_error("X_test must be 2D");
      Ntest = (int)xs[0];
      int Dt = (int)xs[1];
      if (Dt != D) throw std::runtime_error("X_test D != X_train D");
      if ((int)y_test.size() != Ntest) throw std::runtime_error("y_test length mismatch");
      X_test_col = rowmajor_ND_to_colmajor_DN(Xt_row, Ntest, D);
    }

    // ---- CUDA alloc + copy ----
    float* d_X = nullptr;
    int8_t* d_y = nullptr;
    float* d_w = nullptr;

    CUDA_CHECK(cudaMalloc(&d_X, (int64_t)D * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, (int64_t)N * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_w, (int64_t)D * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_X, X_train_col.data(), (int64_t)D * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, y_train.data(), (int64_t)N * sizeof(int8_t), cudaMemcpyHostToDevice));

    // weights init: Normal(0,0.01) with seed
    std::vector<float> h_w(D);
    initialize_weights_host(h_w, args.seed);
    CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), (int64_t)D * sizeof(float), cudaMemcpyHostToDevice));

    // val/test device copies (optional)
    float *d_Xv=nullptr, *d_Xt=nullptr;
    int8_t *d_yv=nullptr, *d_yt=nullptr;
    if (Nval > 0) {
      CUDA_CHECK(cudaMalloc(&d_Xv, (int64_t)D * Nval * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&d_yv, (int64_t)Nval * sizeof(int8_t)));
      CUDA_CHECK(cudaMemcpy(d_Xv, X_val_col.data(), (int64_t)D * Nval * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_yv, y_val.data(), (int64_t)Nval * sizeof(int8_t), cudaMemcpyHostToDevice));
    }
    if (Ntest > 0) {
      CUDA_CHECK(cudaMalloc(&d_Xt, (int64_t)D * Ntest * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&d_yt, (int64_t)Ntest * sizeof(int8_t)));
      CUDA_CHECK(cudaMemcpy(d_Xt, X_test_col.data(), (int64_t)D * Ntest * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_yt, y_test.data(), (int64_t)Ntest * sizeof(int8_t), cudaMemcpyHostToDevice));
    }

    // ---- cuBLAS init ----
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // ---- Shuffling indices ----
    std::vector<int> h_indices(N);
    for (int i = 0; i < N; i++) h_indices[i] = i;

    int* d_indices = nullptr;
    CUDA_CHECK(cudaMalloc(&d_indices, (int64_t)N * sizeof(int)));

    // ---- Buffers for a batch ----
    const int Bmax = std::min(args.batch_size, N);

    float* d_Xbatch = nullptr;
    int8_t* d_ybatch = nullptr;
    float* d_logits = nullptr;
    float* d_diff   = nullptr;
    float* d_grad_w = nullptr;

    float* d_loss = nullptr; // new: for BCE
    if (args.log_loss && !args.benchmark) {
      CUDA_CHECK(cudaMalloc(&d_loss, (int64_t)Bmax * sizeof(float)));
    }

    CUDA_CHECK(cudaMalloc(&d_Xbatch, (int64_t)D * Bmax * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ybatch, (int64_t)Bmax * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_logits, (int64_t)Bmax * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_diff,   (int64_t)Bmax * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_w, (int64_t)D * sizeof(float)));

    float b = 0.0f;

    const float one = 1.0f;
    const float zero = 0.0f;

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    if (args.benchmark) {
      std::cout << "epoch,time_ms,samples_per_sec\n";
    }

    // ---- Training loop ----
    for (int epoch = 0; epoch < args.epochs; epoch++) {
      // shuffle with seed + epoch
      {
        std::mt19937 gen(args.seed + epoch);
        std::shuffle(h_indices.begin(), h_indices.end(), gen);
        CUDA_CHECK(cudaMemcpy(d_indices, h_indices.data(), (int64_t)N * sizeof(int), cudaMemcpyHostToDevice));
      }

      double epoch_loss_sum = 0.0;
      int epoch_loss_batches = 0;

      CUDA_CHECK(cudaEventRecord(ev_start));

      for (int batch_start = 0; batch_start < N; batch_start += args.batch_size) {
        int B = std::min(args.batch_size, N - batch_start);
        float invB = 1.0f / (float)B;

        // gather batch
        {
          int total = D * B;
          int threads = 256;
          int blocks = (total + threads - 1) / threads;
          gather_batch_DxB<<<blocks, threads>>>(
            d_X, d_y, d_indices,
            batch_start, B,
            d_Xbatch, d_ybatch,
            D
          );
          CUDA_CHECK(cudaGetLastError());
        }

        // logits = X_batch^T * w
        CUBLAS_CHECK(cublasSgemv(
          handle,
          CUBLAS_OP_T,
          D, B,
          &one,
          d_Xbatch, D,
          d_w, 1,
          &zero,
          d_logits, 1
        ));

        // optional BCE loss (uses logits before diff update; still valid)
        if (d_loss) {
          int threads = 256;
          int blocks = (B + threads - 1) / threads;
          bce_loss_kernel<<<blocks, threads>>>(d_logits, d_ybatch, d_loss, b, B);
          CUDA_CHECK(cudaGetLastError());
          float batch_loss_sum = device_sum(d_loss, B);
          epoch_loss_sum += (double)(batch_loss_sum * invB);
          epoch_loss_batches++;
        }

        // diff = sigmoid(logits + b) - y
        {
          int threads = 256;
          int blocks = (B + threads - 1) / threads;
          sigmoid_and_diff_bias_kernel<<<blocks, threads>>>(d_logits, d_ybatch, d_diff, b, B);
          CUDA_CHECK(cudaGetLastError());
        }

        // grad_w = X_batch * diff
        CUBLAS_CHECK(cublasSgemv(
          handle,
          CUBLAS_OP_N,
          D, B,
          &one,
          d_Xbatch, D,
          d_diff, 1,
          &zero,
          d_grad_w, 1
        ));

        // average grad + update
        {
          int threads = 256;
          int blocks = (D + threads - 1) / threads;
          scale_grad_kernel<<<blocks, threads>>>(d_grad_w, invB, D);
          CUDA_CHECK(cudaGetLastError());

          update_w_kernel<<<blocks, threads>>>(d_w, d_grad_w, args.lr, D);
          CUDA_CHECK(cudaGetLastError());
        }

        // bias update: b -= lr * mean(diff)
        float grad_b = device_sum(d_diff, B) * invB;
        b -= args.lr * grad_b;
      }

      CUDA_CHECK(cudaEventRecord(ev_stop));
      CUDA_CHECK(cudaEventSynchronize(ev_stop));

      float ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
      float seconds = ms / 1000.0f;
      float samples_per_sec = (seconds > 0.0f) ? (float)N / seconds : 0.0f;

      if (args.benchmark) {
        std::cout << (epoch + 1) << "," << ms << "," << samples_per_sec << "\n";
        continue;
      }

      if (args.verbose) {
        std::cout << "Epoch " << (epoch + 1) << "/" << args.epochs
                  << " | time_ms=" << ms
                  << " | samples_per_sec=" << samples_per_sec
                  << " | b=" << b;

        if (epoch_loss_batches > 0) {
          double avg_loss = epoch_loss_sum / (double)epoch_loss_batches;
          std::cout << " | loss=" << avg_loss;
        }

        if (Nval > 0) {
          float acc = eval_accuracy(handle, d_Xv, d_yv, Nval, D, d_w, b, args.batch_size);
          std::cout << " | val_acc=" << acc;
        }
        std::cout << "\n";
      }
    }

    if (!args.benchmark && Ntest > 0) {
      float acc = eval_accuracy(handle, d_Xt, d_yt, Ntest, D, d_w, b, args.batch_size);
      std::cout << "Test accuracy: " << acc << "\n";
    }

    // ---- Inference benchmark (forward pass compute only) ----
    if (!args.benchmark && args.infer_benchmark) {
      if (Ntest <= 0 || d_Xt == nullptr) {
        throw std::runtime_error("--infer-benchmark requires --x_test/--y_test (to know N_infer, D)");
      }
      run_inference_benchmark(
        handle,
        d_Xt,
        Ntest, D,
        d_w, b,
        args.batch_size,
        args.infer_warmup,
        args.infer_repeats
      );
    }
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUBLAS_CHECK(cublasDestroy(handle));
    if (d_loss) CUDA_CHECK(cudaFree(d_loss));
    CUDA_CHECK(cudaFree(d_Xbatch));
    CUDA_CHECK(cudaFree(d_ybatch));
    CUDA_CHECK(cudaFree(d_logits));
    CUDA_CHECK(cudaFree(d_diff));
    CUDA_CHECK(cudaFree(d_grad_w));
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_w));
    if (d_Xv) CUDA_CHECK(cudaFree(d_Xv));
    if (d_yv) CUDA_CHECK(cudaFree(d_yv));
    if (d_Xt) CUDA_CHECK(cudaFree(d_Xt));
    if (d_yt) CUDA_CHECK(cudaFree(d_yt));
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Fatal: " << e.what() << "\n";
    return 1;
  }
}
