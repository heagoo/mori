// Benchmark program for dispatch-combine performance analysis
// Copyright (c) 2025. MIT License.

#include "dispatch_combine.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <chrono>

using namespace simple_dispatch_combine;

struct BenchmarkConfig {
    int hidden_dim;
    int max_tokens_per_rank;
    int num_experts_per_rank;
    int num_experts_per_token;
    int warmup_iterations;
    int benchmark_iterations;
};

struct BenchmarkResults {
    double avg_dispatch_time_ms;
    double avg_combine_time_ms;
    double dispatch_bandwidth_gbps;
    double combine_bandwidth_gbps;
    double min_dispatch_time_ms;
    double max_dispatch_time_ms;
    double min_combine_time_ms;
    double max_combine_time_ms;
};

class DispatchCombineBenchmark {
public:
    DispatchCombineBenchmark(const Config& config, const BenchmarkConfig& bench_config)
        : config_(config), bench_config_(bench_config), ctx_(nullptr), handle_(nullptr) {
        
        // Initialize context
        ctx_ = new DispatchCombineContext(config);
        ctx_->Initialize();
        
        // Initialize handle
        handle_ = new DispatchCombineHandle(ctx_);
        
        // Allocate test data
        AllocateTestData();
    }
    
    ~DispatchCombineBenchmark() {
        FreeTestData();
        if (handle_) delete handle_;
        if (ctx_) {
            ctx_->Finalize();
            delete ctx_;
        }
    }
    
    BenchmarkResults Run() {
        if (config_.rank == 0) {
            std::cout << "\n=== Running Benchmark ===" << std::endl;
            std::cout << "Configuration:" << std::endl;
            std::cout << "  World size: " << config_.world_size << std::endl;
            std::cout << "  Hidden dim: " << bench_config_.hidden_dim << std::endl;
            std::cout << "  Tokens per rank: " << bench_config_.max_tokens_per_rank << std::endl;
            std::cout << "  Experts per rank: " << bench_config_.num_experts_per_rank << std::endl;
            std::cout << "  Experts per token: " << bench_config_.num_experts_per_token << std::endl;
        }
        
        // Warmup
        Warmup();
        
        // Run benchmark
        return RunTimedIterations();
    }

private:
    void AllocateTestData() {
        int num_tokens = bench_config_.max_tokens_per_rank;
        int hidden_dim = bench_config_.hidden_dim;
        int num_experts_per_token = bench_config_.num_experts_per_token;
        int num_experts = config_.world_size * bench_config_.num_experts_per_rank;
        
        // Allocate GPU memory
        size_t tokens_size = num_tokens * hidden_dim * sizeof(float);
        size_t expert_ids_size = num_tokens * num_experts_per_token * sizeof(int32_t);
        size_t weights_size = num_tokens * num_experts_per_token * sizeof(float);
        
        hipMalloc(&d_tokens_, tokens_size);
        hipMalloc(&d_expert_ids_, expert_ids_size);
        hipMalloc(&d_weights_, weights_size);
        
        // Initialize with random data
        std::mt19937 gen(config_.rank + 42);
        std::uniform_real_distribution<float> token_dist(0.0f, 1.0f);
        std::uniform_int_distribution<int> expert_dist(0, num_experts - 1);
        std::uniform_real_distribution<float> weight_dist(0.5f, 1.0f);
        
        std::vector<float> h_tokens(num_tokens * hidden_dim);
        std::vector<int32_t> h_expert_ids(num_tokens * num_experts_per_token);
        std::vector<float> h_weights(num_tokens * num_experts_per_token);
        
        for (auto& val : h_tokens) val = token_dist(gen);
        for (auto& val : h_expert_ids) val = expert_dist(gen);
        
        // Normalize weights per token
        for (int i = 0; i < num_tokens; i++) {
            float sum = 0.0f;
            for (int j = 0; j < num_experts_per_token; j++) {
                float w = weight_dist(gen);
                h_weights[i * num_experts_per_token + j] = w;
                sum += w;
            }
            for (int j = 0; j < num_experts_per_token; j++) {
                h_weights[i * num_experts_per_token + j] /= sum;
            }
        }
        
        // Copy to device
        hipMemcpy(d_tokens_, h_tokens.data(), tokens_size, hipMemcpyHostToDevice);
        hipMemcpy(d_expert_ids_, h_expert_ids.data(), expert_ids_size, hipMemcpyHostToDevice);
        hipMemcpy(d_weights_, h_weights.data(), weights_size, hipMemcpyHostToDevice);
        
        // Prepare inference
        handle_->PrepareInference(d_tokens_, d_expert_ids_, d_weights_,
                                 bench_config_.max_tokens_per_rank);
    }
    
    void FreeTestData() {
        if (d_tokens_) hipFree(d_tokens_);
        if (d_expert_ids_) hipFree(d_expert_ids_);
        if (d_weights_) hipFree(d_weights_);
    }
    
    void Warmup() {
        if (config_.rank == 0) {
            std::cout << "\nWarming up (" << bench_config_.warmup_iterations 
                     << " iterations)..." << std::endl;
        }
        
        for (int i = 0; i < bench_config_.warmup_iterations; i++) {
            handle_->LaunchDispatch();
            handle_->LaunchCombine();
            hipDeviceSynchronize();
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    BenchmarkResults RunTimedIterations() {
        if (config_.rank == 0) {
            std::cout << "Running benchmark (" << bench_config_.benchmark_iterations
                     << " iterations)..." << std::endl;
        }
        
        hipEvent_t start, stop;
        hipEventCreate(&start);
        hipEventCreate(&stop);
        
        std::vector<float> dispatch_times;
        std::vector<float> combine_times;
        
        for (int i = 0; i < bench_config_.benchmark_iterations; i++) {
            // Time dispatch
            MPI_Barrier(MPI_COMM_WORLD);
            hipEventRecord(start);
            handle_->LaunchDispatch();
            hipEventRecord(stop);
            hipEventSynchronize(stop);
            
            float dispatch_time;
            hipEventElapsedTime(&dispatch_time, start, stop);
            dispatch_times.push_back(dispatch_time);
            
            // Time combine
            MPI_Barrier(MPI_COMM_WORLD);
            hipEventRecord(start);
            handle_->LaunchCombine();
            hipEventRecord(stop);
            hipEventSynchronize(stop);
            
            float combine_time;
            hipEventElapsedTime(&combine_time, start, stop);
            combine_times.push_back(combine_time);
        }
        
        hipEventDestroy(start);
        hipEventDestroy(stop);
        
        // Compute statistics
        BenchmarkResults results;
        results.avg_dispatch_time_ms = ComputeAverage(dispatch_times);
        results.avg_combine_time_ms = ComputeAverage(combine_times);
        results.min_dispatch_time_ms = *std::min_element(dispatch_times.begin(), dispatch_times.end());
        results.max_dispatch_time_ms = *std::max_element(dispatch_times.begin(), dispatch_times.end());
        results.min_combine_time_ms = *std::min_element(combine_times.begin(), combine_times.end());
        results.max_combine_time_ms = *std::max_element(combine_times.begin(), combine_times.end());
        
        // Compute bandwidth
        size_t data_size = bench_config_.max_tokens_per_rank * 
                          bench_config_.num_experts_per_token *
                          bench_config_.hidden_dim * sizeof(float);
        
        results.dispatch_bandwidth_gbps = (data_size / 1e9) / (results.avg_dispatch_time_ms / 1000.0);
        results.combine_bandwidth_gbps = (data_size / 1e9) / (results.avg_combine_time_ms / 1000.0);
        
        return results;
    }
    
    double ComputeAverage(const std::vector<float>& values) {
        double sum = 0.0;
        for (float v : values) sum += v;
        return sum / values.size();
    }

private:
    Config config_;
    BenchmarkConfig bench_config_;
    DispatchCombineContext* ctx_;
    DispatchCombineHandle* handle_;
    
    float* d_tokens_;
    int32_t* d_expert_ids_;
    float* d_weights_;
};

void PrintResults(const BenchmarkResults& results, int rank, int world_size) {
    // Gather results from all ranks
    std::vector<double> all_dispatch_avg(world_size);
    std::vector<double> all_combine_avg(world_size);
    std::vector<double> all_dispatch_bw(world_size);
    std::vector<double> all_combine_bw(world_size);
    
    MPI_Gather(&results.avg_dispatch_time_ms, 1, MPI_DOUBLE,
               all_dispatch_avg.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&results.avg_combine_time_ms, 1, MPI_DOUBLE,
               all_combine_avg.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&results.dispatch_bandwidth_gbps, 1, MPI_DOUBLE,
               all_dispatch_bw.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&results.combine_bandwidth_gbps, 1, MPI_DOUBLE,
               all_combine_bw.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "\n=== Benchmark Results ===" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        
        // Per-rank results
        std::cout << "\nPer-Rank Results:" << std::endl;
        std::cout << "Rank | Dispatch (ms) | Dispatch (GB/s) | Combine (ms) | Combine (GB/s)" << std::endl;
        std::cout << "-----|---------------|-----------------|--------------|---------------" << std::endl;
        
        for (int i = 0; i < world_size; i++) {
            std::cout << std::setw(4) << i << " | "
                     << std::setw(13) << all_dispatch_avg[i] << " | "
                     << std::setw(15) << all_dispatch_bw[i] << " | "
                     << std::setw(12) << all_combine_avg[i] << " | "
                     << std::setw(13) << all_combine_bw[i] << std::endl;
        }
        
        // Summary statistics
        double avg_dispatch = 0, avg_combine = 0, avg_dispatch_bw = 0, avg_combine_bw = 0;
        for (int i = 0; i < world_size; i++) {
            avg_dispatch += all_dispatch_avg[i];
            avg_combine += all_combine_avg[i];
            avg_dispatch_bw += all_dispatch_bw[i];
            avg_combine_bw += all_combine_bw[i];
        }
        avg_dispatch /= world_size;
        avg_combine /= world_size;
        avg_dispatch_bw /= world_size;
        avg_combine_bw /= world_size;
        
        std::cout << "\nSummary:" << std::endl;
        std::cout << "  Average Dispatch Time: " << avg_dispatch << " ms" << std::endl;
        std::cout << "  Average Dispatch Bandwidth: " << avg_dispatch_bw << " GB/s" << std::endl;
        std::cout << "  Average Combine Time: " << avg_combine << " ms" << std::endl;
        std::cout << "  Average Combine Bandwidth: " << avg_combine_bw << " GB/s" << std::endl;
        
        std::cout << "\n  Min/Max Dispatch: " << results.min_dispatch_time_ms 
                 << " / " << results.max_dispatch_time_ms << " ms" << std::endl;
        std::cout << "  Min/Max Combine: " << results.min_combine_time_ms
                 << " / " << results.max_combine_time_ms << " ms" << std::endl;
    }
}

int main(int argc, char** argv) {
    // Initialize MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // Parse arguments
    BenchmarkConfig bench_config;
    bench_config.hidden_dim = (argc > 1) ? std::atoi(argv[1]) : 4096;
    bench_config.max_tokens_per_rank = (argc > 2) ? std::atoi(argv[2]) : 128;
    bench_config.num_experts_per_rank = (argc > 3) ? std::atoi(argv[3]) : 8;
    bench_config.num_experts_per_token = (argc > 4) ? std::atoi(argv[4]) : 2;
    bench_config.warmup_iterations = (argc > 5) ? std::atoi(argv[5]) : 5;
    bench_config.benchmark_iterations = (argc > 6) ? std::atoi(argv[6]) : 20;
    
    // Create configuration
    Config config;
    config.world_size = world_size;
    config.rank = rank;
    config.hidden_dim = bench_config.hidden_dim;
    config.max_tokens_per_rank = bench_config.max_tokens_per_rank;
    config.num_experts_per_rank = bench_config.num_experts_per_rank;
    config.num_experts_per_token = bench_config.num_experts_per_token;
    config.gpu_per_node = 8;
    
    // Run benchmark
    DispatchCombineBenchmark benchmark(config, bench_config);
    BenchmarkResults results = benchmark.Run();
    
    // Print results
    PrintResults(results, rank, world_size);
    
    // Cleanup
    MPI_Finalize();
    return 0;
}
