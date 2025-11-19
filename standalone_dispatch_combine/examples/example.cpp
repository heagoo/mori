// Example program demonstrating dispatch-combine with P2P and RDMA
// Copyright (c) 2025. MIT License.

#include "dispatch_combine.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

using namespace simple_dispatch_combine;

// Helper function to initialize test data
void InitializeTestData(float* hidden_states, int32_t* expert_ids, float* weights,
                       int num_tokens, int hidden_dim, int num_experts_per_token,
                       int num_experts, int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> token_dist(0.0f, 1.0f);
    std::uniform_int_distribution<int> expert_dist(0, num_experts - 1);
    std::uniform_real_distribution<float> weight_dist(0.0f, 1.0f);
    
    // Initialize tokens with random values
    for (int i = 0; i < num_tokens * hidden_dim; i++) {
        hidden_states[i] = token_dist(gen);
    }
    
    // Initialize expert assignments
    for (int i = 0; i < num_tokens * num_experts_per_token; i++) {
        expert_ids[i] = expert_dist(gen);
    }
    
    // Initialize weights (normalized per token)
    for (int i = 0; i < num_tokens; i++) {
        float sum = 0.0f;
        for (int j = 0; j < num_experts_per_token; j++) {
            float w = weight_dist(gen);
            weights[i * num_experts_per_token + j] = w;
            sum += w;
        }
        // Normalize
        float inv_sum = 1.0f / sum;
        for (int j = 0; j < num_experts_per_token; j++) {
            weights[i * num_experts_per_token + j] *= inv_sum;
        }
    }
}

// Helper function to verify results (simple sanity check)
bool VerifyResults(const float* output, int num_tokens, int hidden_dim) {
    bool valid = true;
    for (int i = 0; i < num_tokens * hidden_dim; i++) {
        if (std::isnan(output[i]) || std::isinf(output[i])) {
            std::cerr << "Invalid output at index " << i << ": " << output[i] << std::endl;
            valid = false;
            break;
        }
    }
    return valid;
}

int main(int argc, char** argv) {
    // Initialize MPI (must be called once at start of program)
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // Parse command line arguments (simple version)
    int hidden_dim = 4096;
    int max_tokens_per_rank = 128;
    int num_experts_per_rank = 32;
    int num_experts_per_token = 8;
    int gpu_per_node = 8;
    int num_warmup = 5;
    int num_iterations = 10;
    
    if (argc > 1) hidden_dim = std::atoi(argv[1]);
    if (argc > 2) max_tokens_per_rank = std::atoi(argv[2]);
    if (argc > 3) num_experts_per_rank = std::atoi(argv[3]);
    if (argc > 4) num_experts_per_token = std::atoi(argv[4]);
    
    if (rank == 0) {
        std::cout << "=== Dispatch-Combine Example ===" << std::endl;
        std::cout << "World size: " << world_size << std::endl;
        std::cout << "Hidden dim: " << hidden_dim << std::endl;
        std::cout << "Max tokens per rank: " << max_tokens_per_rank << std::endl;
        std::cout << "Experts per rank: " << num_experts_per_rank << std::endl;
        std::cout << "Experts per token: " << num_experts_per_token << std::endl;
        std::cout << "GPUs per node: " << gpu_per_node << std::endl;
    }
    
    // Create configuration
    Config config;
    config.world_size = world_size;
    config.rank = rank;
    config.hidden_dim = hidden_dim;
    config.max_tokens_per_rank = max_tokens_per_rank;
    config.num_experts_per_rank = num_experts_per_rank;
    config.num_experts_per_token = num_experts_per_token;
    config.gpu_per_node = gpu_per_node;
    
    // Create context and initialize
    DispatchCombineContext ctx(config);
    ctx.Initialize();
    
    // Create handle for dispatch-combine operations
    DispatchCombineHandle handle(&ctx);
    
    // Allocate and initialize input data on GPU
    int num_tokens = max_tokens_per_rank;
    int num_experts = world_size * num_experts_per_rank;
    
    float* d_hidden_states;
    int32_t* d_expert_ids;
    float* d_weights;
    
    size_t tokens_size = num_tokens * hidden_dim * sizeof(float);
    size_t expert_ids_size = num_tokens * num_experts_per_token * sizeof(int32_t);
    size_t weights_size = num_tokens * num_experts_per_token * sizeof(float);
    
    hipMalloc(&d_hidden_states, tokens_size);
    hipMalloc(&d_expert_ids, expert_ids_size);
    hipMalloc(&d_weights, weights_size);
    
    // Initialize on host then copy to device
    std::vector<float> h_hidden_states(num_tokens * hidden_dim);
    std::vector<int32_t> h_expert_ids(num_tokens * num_experts_per_token);
    std::vector<float> h_weights(num_tokens * num_experts_per_token);
    
    InitializeTestData(h_hidden_states.data(), h_expert_ids.data(), h_weights.data(),
                      num_tokens, hidden_dim, num_experts_per_token, num_experts,
                      rank + 128);
    
    hipMemcpy(d_hidden_states, h_hidden_states.data(), tokens_size, hipMemcpyHostToDevice);
    hipMemcpy(d_expert_ids, h_expert_ids.data(), expert_ids_size, hipMemcpyHostToDevice);
    hipMemcpy(d_weights, h_weights.data(), weights_size, hipMemcpyHostToDevice);
    
    // Prepare inference
    handle.PrepareInference(d_hidden_states, d_expert_ids, d_weights, num_tokens);
    
    // Warmup iterations
    if (rank == 0) std::cout << "\nWarming up..." << std::endl;
    for (int i = 0; i < num_warmup; i++) {
        handle.LaunchDispatch();
        handle.LaunchCombine();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Timed iterations
    if (rank == 0) std::cout << "Running timed iterations..." << std::endl;
    
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    
    float total_dispatch_time = 0.0f;
    float total_combine_time = 0.0f;
    
    for (int i = 0; i < num_iterations; i++) {
        // Time dispatch
        hipEventRecord(start);
        handle.LaunchDispatch();
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        
        float dispatch_time;
        hipEventElapsedTime(&dispatch_time, start, stop);
        total_dispatch_time += dispatch_time;
        
        // Time combine
        hipEventRecord(start);
        handle.LaunchCombine();
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        
        float combine_time;
        hipEventElapsedTime(&combine_time, start, stop);
        total_combine_time += combine_time;
    }
    
    // Report results
    float avg_dispatch_time = total_dispatch_time / num_iterations;
    float avg_combine_time = total_combine_time / num_iterations;
    
    // Calculate bandwidth
    size_t dispatch_bytes = num_tokens * num_experts_per_token * hidden_dim * sizeof(float);
    float dispatch_bw = (dispatch_bytes / 1e9) / (avg_dispatch_time / 1000.0f);
    float combine_bw = (dispatch_bytes / 1e9) / (avg_combine_time / 1000.0f);
    
    std::cout << "Rank " << rank 
              << " - Dispatch: " << avg_dispatch_time << " ms"
              << " (" << dispatch_bw << " GB/s)"
              << " | Combine: " << avg_combine_time << " ms"
              << " (" << combine_bw << " GB/s)"
              << std::endl;
    
    // Verify output
    std::vector<float> h_output(num_tokens * hidden_dim);
    void* output_ptr = handle.GetOutputBuffer();
    hipMemcpy(h_output.data(), output_ptr, tokens_size, hipMemcpyDeviceToHost);
    
    bool valid = VerifyResults(h_output.data(), num_tokens, hidden_dim);
    if (rank == 0) {
        if (valid) {
            std::cout << "\n✓ Results verified successfully!" << std::endl;
        } else {
            std::cout << "\n✗ Result verification failed!" << std::endl;
        }
    }
    
    // Cleanup
    hipFree(d_hidden_states);
    hipFree(d_expert_ids);
    hipFree(d_weights);
    hipEventDestroy(start);
    hipEventDestroy(stop);
    
    ctx.Finalize();
    
    // Finalize MPI (called once at end of program)
    MPI_Finalize();
    
    return 0;
}
