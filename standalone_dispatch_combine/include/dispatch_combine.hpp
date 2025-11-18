// Simple standalone dispatch-combine implementation with P2P and RDMA
// Copyright (c) 2025. MIT License.

#pragma once

#include <hip/hip_runtime.h>
#include <mpi.h>
#include <cstdint>
#include <vector>

namespace simple_dispatch_combine {

// Configuration for dispatch-combine operations
struct Config {
    int world_size;          // Total number of processes
    int rank;                // Current process rank
    int hidden_dim;          // Hidden dimension size
    int max_tokens_per_rank; // Maximum tokens per rank
    int num_experts_per_rank; // Number of experts per rank
    int num_experts_per_token; // Number of experts per token (top-k)
    int gpu_per_node;        // Number of GPUs per node
};

// Symmetric memory object holding local and peer pointers
struct SymmetricMemory {
    void* local_ptr;                    // Local GPU memory pointer
    std::vector<uintptr_t> peer_ptrs;   // Peer GPU memory pointers (P2P or RDMA)
    std::vector<uint32_t> rkeys;        // RDMA remote keys
    uint32_t lkey;                      // RDMA local key
    size_t size;                        // Memory size
    
    SymmetricMemory() : local_ptr(nullptr), lkey(0), size(0) {}
};

// Context managing all communication resources
class DispatchCombineContext {
public:
    DispatchCombineContext(const Config& cfg);
    ~DispatchCombineContext();
    
    // Initialize MPI and communication resources
    void Initialize();
    
    // Cleanup resources
    void Finalize();
    
    // Allocate symmetric GPU memory accessible from all peers
    SymmetricMemory* AllocateSymmetricMemory(size_t size);
    
    // Free symmetric GPU memory
    void FreeSymmetricMemory(SymmetricMemory* mem);
    
    // Get configuration
    const Config& GetConfig() const { return config_; }
    
    // Check if peer is in same node (use P2P) or different node (use RDMA)
    bool IsSameNode(int peer_rank) const;
    
private:
    Config config_;
    bool initialized_;
    MPI_Comm comm_;
    
    // Track symmetric memory allocations
    std::vector<SymmetricMemory*> allocations_;
    
    // RDMA context (simplified - in real implementation would use ibverbs)
    struct RdmaContext {
        void* pd;  // Protection domain
        void* cq;  // Completion queue
        std::vector<void*> qps; // Queue pairs
    } rdma_ctx_;
    
    void InitializeMPI();
    void InitializeRDMA();
    void ExchangeP2PHandles(SymmetricMemory* mem);
    void ExchangeRDMAKeys(SymmetricMemory* mem);
};

// Handle for dispatch-combine operations
class DispatchCombineHandle {
public:
    DispatchCombineHandle(DispatchCombineContext* ctx);
    ~DispatchCombineHandle();
    
    // Prepare buffers for inference
    // tokens: [num_tokens, hidden_dim] input tokens on this rank
    // expert_ids: [num_tokens, num_experts_per_token] expert assignments
    // weights: [num_tokens, num_experts_per_token] weights for combining
    void PrepareInference(void* tokens, int32_t* expert_ids, float* weights, int num_tokens);
    
    // Dispatch phase: send tokens to appropriate expert ranks
    // Uses P2P for intra-node, RDMA for inter-node
    void LaunchDispatch(hipStream_t stream = 0);
    
    // Combine phase: receive and aggregate expert outputs
    // Uses P2P for intra-node, RDMA for inter-node
    void LaunchCombine(hipStream_t stream = 0);
    
    // Get output buffer (after combine)
    void* GetOutputBuffer() const { return output_buffer_->local_ptr; }
    
private:
    DispatchCombineContext* ctx_;
    Config config_;
    
    // User provided buffers
    void* input_tokens_;
    int32_t* expert_ids_;
    float* weights_;
    int num_tokens_;
    
    // Symmetric memory buffers
    SymmetricMemory* dispatch_buffer_;   // Buffer for dispatched tokens
    SymmetricMemory* output_buffer_;     // Buffer for combined output
    SymmetricMemory* token_counts_;      // Track token counts per rank
    
    // GPU work buffers
    int32_t* send_offsets_;    // Offsets for sending tokens to each rank
    int32_t* recv_offsets_;    // Offsets for receiving tokens from each rank
    
    void AllocateBuffers();
    void FreeBuffers();
    void ComputeSendRecvCounts();
};

} // namespace simple_dispatch_combine
