/**
 * Comprehensive CPU Logistic Regression Experiment
 * Measures all HPCA metrics:
 *   1. Time per epoch (ms)
 *   2. Training throughput (samples/sec)
 *   3. Inference latency (ms per sample) and throughput (samples/sec)
 *   4. Accuracy (train/val/test)
 *   5. Loss (cross-entropy) per epoch
 *   6. OpenMP scalability (speedup vs threads)
 */

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "npy_loader.h"
#include "logistic_regression.h"
#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================================
// Configuration
// ============================================================================
struct ExperimentConfig {
    size_t num_epochs = 20;
    size_t batch_size = 4096;
    float learning_rate = 0.1f;
    size_t max_samples = 0;  // 0 = use all samples
    int random_seed = 42;
    int num_threads = 1;
    std::string data_dir = "../data";
    std::string output_prefix = "experiment";
    bool verbose = true;
};

// ============================================================================
// Epoch Result Structure
// ============================================================================
struct EpochResult {
    int epoch;
    double train_loss;
    double val_loss;
    double train_accuracy;
    double val_accuracy;
    double epoch_time_ms;
    double samples_per_sec;
};

// ============================================================================
// Final Results Structure
// ============================================================================
struct FinalResults {
    // Config
    size_t N;
    size_t D;
    size_t batch_size;
    size_t epochs;
    int num_threads;
    float learning_rate;
    
    // Training metrics
    double avg_epoch_time_ms;
    double avg_training_throughput;
    double final_train_loss;
    double final_val_loss;
    double final_test_loss;
    double final_train_accuracy;
    double final_val_accuracy;
    double final_test_accuracy;
    
    // Inference metrics
    double inference_latency_ms;  // ms per sample
    double inference_throughput;  // samples/sec
    
    // Scalability
    double speedup;
    double efficiency;
};

// ============================================================================
// Helper Functions
// ============================================================================

void initialize_weights(float* w, size_t D, int seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 0.01f);
    for (size_t j = 0; j < D; ++j) {
        w[j] = dist(gen);
    }
}

void shuffle_indices(std::vector<size_t>& indices, int seed) {
    std::mt19937 gen(seed);
    std::shuffle(indices.begin(), indices.end(), gen);
}

// Compute accuracy
float compute_accuracy(const int8_t* y_true, const int8_t* y_pred, size_t N) {
    size_t correct = 0;
    for (size_t i = 0; i < N; ++i) {
        if (y_true[i] == y_pred[i]) {
            correct++;
        }
    }
    return static_cast<float>(correct) / static_cast<float>(N);
}

// Evaluate model on dataset
void evaluate_model(
    const float* X, const int8_t* y, size_t N, size_t D,
    const float* w, float b,
    float* out_loss, float* out_accuracy
) {
    std::vector<float> probs(N);
    std::vector<int8_t> preds(N);
    
    predict_proba(X, w, b, probs.data(), N, D);
    predict_label(X, w, b, preds.data(), N, D, 0.5f);
    
    *out_loss = binary_cross_entropy(y, probs.data(), N);
    *out_accuracy = compute_accuracy(y, preds.data(), N);
}

// ============================================================================
// Training Function with Full Logging
// ============================================================================
std::vector<EpochResult> train_with_logging(
    const float* X_train, const int8_t* y_train, size_t N_train,
    const float* X_val, const int8_t* y_val, size_t N_val,
    size_t D,
    const ExperimentConfig& config,
    float* w, float* b
) {
    std::vector<EpochResult> results;
    
    // Initialize weights
    initialize_weights(w, D, config.random_seed);
    *b = 0.0f;
    
    // Set OpenMP threads
    #ifdef _OPENMP
    if (config.num_threads > 0) {
        omp_set_num_threads(config.num_threads);
    }
    #endif
    
    // Create indices for shuffling
    std::vector<size_t> indices(N_train);
    for (size_t i = 0; i < N_train; ++i) {
        indices[i] = i;
    }
    
    // Allocate buffers
    std::vector<float> X_batch(config.batch_size * D);
    std::vector<int8_t> y_batch(config.batch_size);
    std::vector<float> grad_w(D);
    float grad_b;
    
    if (config.verbose) {
        std::cout << "\n" << std::string(100, '=') << "\n";
        std::cout << std::setw(6) << "Epoch" << " | "
                  << std::setw(12) << "Train Loss" << " | "
                  << std::setw(12) << "Val Loss" << " | "
                  << std::setw(12) << "Train Acc %" << " | "
                  << std::setw(12) << "Val Acc %" << " | "
                  << std::setw(12) << "Time (ms)" << " | "
                  << std::setw(15) << "Throughput" << "\n";
        std::cout << std::string(100, '-') << "\n";
    }
    
    // Training loop
    for (size_t epoch = 0; epoch < config.num_epochs; ++epoch) {
        EpochResult res;
        res.epoch = static_cast<int>(epoch + 1);
        
        // Shuffle indices
        shuffle_indices(indices, config.random_seed + static_cast<int>(epoch));
        
        auto epoch_start = std::chrono::high_resolution_clock::now();
        
        // Process mini-batches
        for (size_t batch_start = 0; batch_start < N_train; batch_start += config.batch_size) {
            size_t current_batch_size = std::min(config.batch_size, N_train - batch_start);
            
            // Extract mini-batch
            for (size_t i = 0; i < current_batch_size; ++i) {
                size_t idx = indices[batch_start + i];
                for (size_t j = 0; j < D; ++j) {
                    X_batch[i * D + j] = X_train[idx * D + j];
                }
                y_batch[i] = y_train[idx];
            }
            
            // Compute gradients
            #ifdef _OPENMP
            if (config.num_threads > 1) {
                compute_gradient_omp(
                    X_batch.data(), y_batch.data(), w, *b,
                    grad_w.data(), &grad_b, current_batch_size, D
                );
            } else {
                compute_gradient(
                    X_batch.data(), y_batch.data(), w, *b,
                    grad_w.data(), &grad_b, current_batch_size, D
                );
            }
            #else
            compute_gradient(
                X_batch.data(), y_batch.data(), w, *b,
                grad_w.data(), &grad_b, current_batch_size, D
            );
            #endif
            
            // Update parameters
            for (size_t j = 0; j < D; ++j) {
                w[j] -= config.learning_rate * grad_w[j];
            }
            *b -= config.learning_rate * grad_b;
        }
        
        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::microseconds>(
            epoch_end - epoch_start);
        
        res.epoch_time_ms = epoch_duration.count() / 1000.0;
        res.samples_per_sec = (N_train * 1000.0) / res.epoch_time_ms;
        
        // Evaluate on train and validation sets
        float train_loss, train_acc, val_loss, val_acc;
        evaluate_model(X_train, y_train, N_train, D, w, *b, &train_loss, &train_acc);
        evaluate_model(X_val, y_val, N_val, D, w, *b, &val_loss, &val_acc);
        
        res.train_loss = train_loss;
        res.val_loss = val_loss;
        res.train_accuracy = train_acc * 100.0;
        res.val_accuracy = val_acc * 100.0;
        
        results.push_back(res);
        
        if (config.verbose) {
            std::cout << std::setw(6) << res.epoch << " | "
                      << std::fixed << std::setprecision(6) << std::setw(12) << res.train_loss << " | "
                      << std::setw(12) << res.val_loss << " | "
                      << std::setprecision(2) << std::setw(12) << res.train_accuracy << " | "
                      << std::setw(12) << res.val_accuracy << " | "
                      << std::setprecision(1) << std::setw(12) << res.epoch_time_ms << " | "
                      << std::setprecision(0) << std::setw(15) << res.samples_per_sec << "\n";
        }
    }
    
    if (config.verbose) {
        std::cout << std::string(100, '=') << "\n";
    }
    
    return results;
}

// ============================================================================
// Inference Benchmark
// ============================================================================
void benchmark_inference(
    const float* X, size_t N, size_t D,
    const float* w, float b,
    double* out_latency_ms, double* out_throughput
) {
    std::vector<float> probs(N);
    std::vector<int8_t> labels(N);
    
    // Warm-up
    for (int i = 0; i < 5; ++i) {
        predict_proba(X, w, b, probs.data(), N, D);
        predict_label(X, w, b, labels.data(), N, D, 0.5f);
    }
    
    // Measure batch inference
    const int num_runs = 20;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_runs; ++i) {
        predict_proba(X, w, b, probs.data(), N, D);
        predict_label(X, w, b, labels.data(), N, D, 0.5f);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double total_time_ms = duration.count() / 1000.0;
    double avg_time_ms = total_time_ms / num_runs;
    
    *out_latency_ms = avg_time_ms / N;  // ms per sample
    *out_throughput = (N * 1000.0) / avg_time_ms;  // samples/sec
}

// ============================================================================
// Save Results to CSV
// ============================================================================
void save_epoch_results(const std::vector<EpochResult>& results, 
                        const std::string& filename, int num_threads) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    file << "epoch,train_loss,val_loss,train_accuracy,val_accuracy,epoch_time_ms,samples_per_sec,num_threads\n";
    
    for (const auto& r : results) {
        file << r.epoch << ","
             << std::fixed << std::setprecision(6) << r.train_loss << ","
             << r.val_loss << ","
             << std::setprecision(4) << r.train_accuracy << ","
             << r.val_accuracy << ","
             << std::setprecision(2) << r.epoch_time_ms << ","
             << std::setprecision(0) << r.samples_per_sec << ","
             << num_threads << "\n";
    }
    
    file.close();
}

void save_final_results(const FinalResults& r, const std::string& filename, bool append) {
    std::ofstream file;
    if (append) {
        file.open(filename, std::ios::app);
    } else {
        file.open(filename);
        // Write header
        file << "N,D,batch_size,epochs,num_threads,learning_rate,"
             << "avg_epoch_time_ms,avg_training_throughput,"
             << "train_loss,val_loss,test_loss,"
             << "train_accuracy,val_accuracy,test_accuracy,"
             << "inference_latency_ms,inference_throughput,"
             << "speedup,efficiency\n";
    }
    
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    file << r.N << "," << r.D << "," << r.batch_size << "," << r.epochs << ","
         << r.num_threads << "," << std::fixed << std::setprecision(4) << r.learning_rate << ","
         << std::setprecision(2) << r.avg_epoch_time_ms << ","
         << std::setprecision(0) << r.avg_training_throughput << ","
         << std::setprecision(6) << r.final_train_loss << ","
         << r.final_val_loss << "," << r.final_test_loss << ","
         << std::setprecision(2) << r.final_train_accuracy << ","
         << r.final_val_accuracy << "," << r.final_test_accuracy << ","
         << std::setprecision(6) << r.inference_latency_ms << ","
         << std::setprecision(0) << r.inference_throughput << ","
         << std::setprecision(3) << r.speedup << "," << r.efficiency << "\n";
    
    file.close();
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[]) {
    try {
        ExperimentConfig config;
        bool append_results = false;
        double baseline_time = 0.0;
        
        // Parse command line arguments
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
                config.num_epochs = std::stoull(argv[++i]);
            } else if (std::strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc) {
                config.batch_size = std::stoull(argv[++i]);
            } else if (std::strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
                config.learning_rate = std::stof(argv[++i]);
            } else if (std::strcmp(argv[i], "--N") == 0 && i + 1 < argc) {
                config.max_samples = std::stoull(argv[++i]);
            } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
                config.random_seed = std::stoi(argv[++i]);
            } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
                config.num_threads = std::stoi(argv[++i]);
            } else if (std::strcmp(argv[i], "--data-dir") == 0 && i + 1 < argc) {
                config.data_dir = argv[++i];
            } else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
                config.output_prefix = argv[++i];
            } else if (std::strcmp(argv[i], "--quiet") == 0) {
                config.verbose = false;
            } else if (std::strcmp(argv[i], "--append") == 0) {
                append_results = true;
            } else if (std::strcmp(argv[i], "--baseline-time") == 0 && i + 1 < argc) {
                baseline_time = std::stod(argv[++i]);
            }
        }
        
        std::cout << "\n";
        std::cout << "╔══════════════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║         CPU LOGISTIC REGRESSION - HPCA EXPERIMENT                            ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════════════════════════╝\n\n";
        
        // Print OpenMP status
        #ifdef _OPENMP
        std::cout << "[OpenMP] Enabled, using " << config.num_threads << " thread(s)\n";
        #else
        std::cout << "[OpenMP] Not available, single-threaded execution\n";
        #endif
        
        // ====================================================================
        // Load Data
        // ====================================================================
        std::cout << "\n[1/5] Loading datasets...\n";
        
        NpyArray X_train_npy = load_npy(config.data_dir + "/X_train_norm.npy");
        NpyArray y_train_npy = load_npy(config.data_dir + "/y_train.npy");
        NpyArray X_val_npy = load_npy(config.data_dir + "/X_val_norm.npy");
        NpyArray y_val_npy = load_npy(config.data_dir + "/y_val.npy");
        NpyArray X_test_npy = load_npy(config.data_dir + "/X_test_norm.npy");
        NpyArray y_test_npy = load_npy(config.data_dir + "/y_test.npy");
        
        size_t N_train_orig = X_train_npy.shape[0];
        size_t N_val = X_val_npy.shape[0];
        size_t N_test = X_test_npy.shape[0];
        size_t D = X_train_npy.shape[1];
        
        // Apply subsampling if --N is specified
        size_t N_train = N_train_orig;
        if (config.max_samples > 0 && config.max_samples < N_train_orig) {
            N_train = config.max_samples;
            std::cout << "      Train: N=" << N_train << " (subsampled from " << N_train_orig << "), D=" << D << "\n";
        } else {
            std::cout << "      Train: N=" << N_train << ", D=" << D << "\n";
        }
        std::cout << "      Val:   N=" << N_val << "\n";
        std::cout << "      Test:  N=" << N_test << "\n";
        
        // ====================================================================
        // Print Configuration
        // ====================================================================
        std::cout << "\n[2/5] Configuration:\n";
        std::cout << "      N (samples):   " << N_train << "\n";
        std::cout << "      Epochs:        " << config.num_epochs << "\n";
        std::cout << "      Batch size:    " << config.batch_size << "\n";
        std::cout << "      Learning rate: " << config.learning_rate << "\n";
        std::cout << "      Threads:       " << config.num_threads << "\n";
        std::cout << "      Random seed:   " << config.random_seed << "\n";
        
        // ====================================================================
        // Training
        // ====================================================================
        std::cout << "\n[3/5] Training model...\n";
        
        std::vector<float> w(D);
        float b = 0.0f;
        
        auto total_start = std::chrono::high_resolution_clock::now();
        
        std::vector<EpochResult> epoch_results = train_with_logging(
            X_train_npy.data_float.data(), y_train_npy.data_int8.data(), N_train,
            X_val_npy.data_float.data(), y_val_npy.data_int8.data(), N_val,
            D, config, w.data(), &b
        );
        
        auto total_end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            total_end - total_start);
        
        std::cout << "\n      Total training time: " << total_duration.count() << " ms\n";
        
        // ====================================================================
        // Test Set Evaluation
        // ====================================================================
        std::cout << "\n[4/5] Evaluating on test set...\n";
        
        float test_loss, test_acc;
        evaluate_model(X_test_npy.data_float.data(), y_test_npy.data_int8.data(),
                      N_test, D, w.data(), b, &test_loss, &test_acc);
        
        std::cout << "      Test Loss:     " << std::fixed << std::setprecision(6) << test_loss << "\n";
        std::cout << "      Test Accuracy: " << std::setprecision(2) << (test_acc * 100.0) << "%\n";
        
        // ====================================================================
        // Inference Benchmark
        // ====================================================================
        std::cout << "\n[5/5] Benchmarking inference...\n";
        
        double inf_latency_ms, inf_throughput;
        benchmark_inference(X_test_npy.data_float.data(), N_test, D,
                           w.data(), b, &inf_latency_ms, &inf_throughput);
        
        std::cout << "      Inference latency:    " << std::scientific << std::setprecision(4) 
                  << inf_latency_ms << " ms/sample\n";
        std::cout << "      Inference throughput: " << std::fixed << std::setprecision(0) 
                  << inf_throughput << " samples/sec\n";
        
        // ====================================================================
        // Compute Summary Statistics
        // ====================================================================
        double avg_epoch_time = 0.0;
        double avg_throughput = 0.0;
        
        // Skip first epoch (warm-up) if we have enough epochs
        size_t start_epoch = (epoch_results.size() > 2) ? 1 : 0;
        for (size_t i = start_epoch; i < epoch_results.size(); ++i) {
            avg_epoch_time += epoch_results[i].epoch_time_ms;
            avg_throughput += epoch_results[i].samples_per_sec;
        }
        size_t count = epoch_results.size() - start_epoch;
        avg_epoch_time /= count;
        avg_throughput /= count;
        
        // Get final metrics from last epoch
        const EpochResult& last = epoch_results.back();
        
        // Compute speedup and efficiency
        double speedup = 1.0;
        double efficiency = 1.0;
        if (baseline_time > 0.0 && config.num_threads > 1) {
            speedup = baseline_time / avg_epoch_time;
            efficiency = speedup / config.num_threads;
        }
        
        // ====================================================================
        // Print Final Summary
        // ====================================================================
        std::cout << "\n";
        std::cout << "╔══════════════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                              FINAL RESULTS                                   ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ Training Metrics                                                             ║\n";
        std::cout << "║   Time per epoch (avg):    " << std::setw(12) << std::fixed << std::setprecision(2) 
                  << avg_epoch_time << " ms" << std::setw(35) << "║\n";
        std::cout << "║   Training throughput:     " << std::setw(12) << std::setprecision(0) 
                  << avg_throughput << " samples/sec" << std::setw(25) << "║\n";
        if (speedup > 1.0) {
            std::cout << "║   Speedup:                 " << std::setw(12) << std::setprecision(3) 
                      << speedup << "x" << std::setw(37) << "║\n";
            std::cout << "║   Parallel Efficiency:     " << std::setw(12) << std::setprecision(1) 
                      << (efficiency * 100.0) << "%" << std::setw(37) << "║\n";
        }
        std::cout << "╠══════════════════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ Final Accuracy & Loss                                                        ║\n";
        std::cout << "║   Train:  Loss=" << std::setw(10) << std::setprecision(6) << last.train_loss 
                  << "  Acc=" << std::setw(6) << std::setprecision(2) << last.train_accuracy << "%"
                  << std::setw(29) << "║\n";
        std::cout << "║   Val:    Loss=" << std::setw(10) << std::setprecision(6) << last.val_loss 
                  << "  Acc=" << std::setw(6) << std::setprecision(2) << last.val_accuracy << "%"
                  << std::setw(29) << "║\n";
        std::cout << "║   Test:   Loss=" << std::setw(10) << std::setprecision(6) << test_loss 
                  << "  Acc=" << std::setw(6) << std::setprecision(2) << (test_acc * 100.0) << "%"
                  << std::setw(29) << "║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ Inference Metrics                                                            ║\n";
        std::cout << "║   Latency:    " << std::setw(12) << std::scientific << std::setprecision(4) 
                  << inf_latency_ms << " ms/sample" << std::setw(31) << "║\n";
        std::cout << "║   Throughput: " << std::setw(12) << std::fixed << std::setprecision(0) 
                  << inf_throughput << " samples/sec" << std::setw(29) << "║\n";
        std::cout << "╚══════════════════════════════════════════════════════════════════════════════╝\n";
        
        // ====================================================================
        // Save Results
        // ====================================================================
        std::string epoch_file = config.output_prefix + "_epochs_t" + std::to_string(config.num_threads) + ".csv";
        std::string summary_file = config.output_prefix + "_summary.csv";
        
        save_epoch_results(epoch_results, epoch_file, config.num_threads);
        std::cout << "\n[Output] Epoch results saved to: " << epoch_file << "\n";
        
        FinalResults final_res;
        final_res.N = N_train;
        final_res.D = D;
        final_res.batch_size = config.batch_size;
        final_res.epochs = config.num_epochs;
        final_res.num_threads = config.num_threads;
        final_res.learning_rate = config.learning_rate;
        final_res.avg_epoch_time_ms = avg_epoch_time;
        final_res.avg_training_throughput = avg_throughput;
        final_res.final_train_loss = last.train_loss;
        final_res.final_val_loss = last.val_loss;
        final_res.final_test_loss = test_loss;
        final_res.final_train_accuracy = last.train_accuracy;
        final_res.final_val_accuracy = last.val_accuracy;
        final_res.final_test_accuracy = test_acc * 100.0;
        final_res.inference_latency_ms = inf_latency_ms;
        final_res.inference_throughput = inf_throughput;
        final_res.speedup = speedup;
        final_res.efficiency = efficiency;
        
        save_final_results(final_res, summary_file, append_results);
        std::cout << "[Output] Summary saved to: " << summary_file << "\n";
        
        // Print avg_epoch_time for scripting (used as baseline for speedup calculation)
        std::cout << "\n[BASELINE_TIME] " << std::fixed << std::setprecision(4) << avg_epoch_time << "\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
