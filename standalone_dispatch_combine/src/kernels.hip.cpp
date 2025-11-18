// GPU kernels for dispatch-combine operations
// Copyright (c) 2025. MIT License.

#include "dispatch_combine.hpp"
#include <hip/hip_runtime.h>

namespace simple_dispatch_combine {

// Device function to compute which rank owns a given expert
__device__ inline int GetExpertRank(int expert_id, int num_experts_per_rank) {
    return expert_id / num_experts_per_rank;
}

// Device function to write data to peer GPU memory (P2P)
// This directly accesses peer GPU memory through IPC handles
__device__ void WriteToP2PPeer(float* dest, const float* src, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = tid; i < size; i += stride) {
        dest[i] = src[i];
    }
}

// Kernel to compute send/receive counts per rank
__global__ void ComputeSendRecvCountsKernel(
    const int32_t* expert_ids,      // [num_tokens, num_experts_per_token]
    int num_tokens,
    int num_experts_per_token,
    int num_experts_per_rank,
    int world_size,
    int32_t* send_counts,            // [world_size]
    int32_t* send_offsets)           // [world_size]
{
    // Use shared memory to accumulate counts
    extern __shared__ int32_t shared_counts[];
    
    int tid = threadIdx.x;
    
    // Initialize shared memory
    if (tid < world_size) {
        shared_counts[tid] = 0;
    }
    __syncthreads();
    
    // Count tokens going to each rank
    int total_dispatches = num_tokens * num_experts_per_token;
    for (int idx = tid; idx < total_dispatches; idx += blockDim.x) {
        int expert_id = expert_ids[idx];
        int dest_rank = GetExpertRank(expert_id, num_experts_per_rank);
        atomicAdd(&shared_counts[dest_rank], 1);
    }
    __syncthreads();
    
    // Write results to global memory
    if (tid < world_size) {
        send_counts[tid] = shared_counts[tid];
    }
}

// Dispatch kernel using P2P for intra-node communication
// This kernel directly writes to peer GPU memory
__global__ void DispatchIntraNodeKernel(
    const float* input_tokens,       // [num_tokens, hidden_dim]
    const int32_t* expert_ids,       // [num_tokens, num_experts_per_token]
    const uintptr_t* peer_ptrs,      // [world_size] - peer GPU pointers
    int num_tokens,
    int num_experts_per_token,
    int num_experts_per_rank,
    int hidden_dim,
    int rank,
    int32_t* send_offsets)           // [world_size] - current write offset for each peer
{
    int token_idx = blockIdx.x;
    int expert_slot = blockIdx.y;
    
    if (token_idx >= num_tokens || expert_slot >= num_experts_per_token) return;
    
    int expert_id = expert_ids[token_idx * num_experts_per_token + expert_slot];
    int dest_rank = GetExpertRank(expert_id, num_experts_per_rank);
    
    // Get write position in destination buffer
    int write_offset = atomicAdd(&send_offsets[dest_rank], 1);
    
    // Get destination pointer (peer GPU memory via P2P)
    float* dest_buffer = reinterpret_cast<float*>(peer_ptrs[dest_rank]);
    float* dest_token = dest_buffer + write_offset * hidden_dim;
    
    // Get source pointer
    const float* src_token = input_tokens + token_idx * hidden_dim;
    
    // Copy token data using cooperative threads
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        dest_token[i] = src_token[i];
    }
}

// Dispatch kernel using RDMA for inter-node communication
// In a full implementation, this would use GPU-initiated RDMA
// For now, we'll simulate with regular GPU copies
__global__ void DispatchInterNodeKernel(
    const float* input_tokens,       // [num_tokens, hidden_dim]
    const int32_t* expert_ids,       // [num_tokens, num_experts_per_token]
    float* staging_buffer,           // Staging buffer for RDMA
    int num_tokens,
    int num_experts_per_token,
    int num_experts_per_rank,
    int hidden_dim,
    int rank,
    int32_t* send_offsets)           // [world_size]
{
    int token_idx = blockIdx.x;
    int expert_slot = blockIdx.y;
    
    if (token_idx >= num_tokens || expert_slot >= num_experts_per_token) return;
    
    int expert_id = expert_ids[token_idx * num_experts_per_token + expert_slot];
    int dest_rank = GetExpertRank(expert_id, num_experts_per_rank);
    
    // Get write position in staging buffer
    int write_offset = atomicAdd(&send_offsets[dest_rank], 1);
    
    // Write to staging buffer
    float* dest_token = staging_buffer + write_offset * hidden_dim;
    const float* src_token = input_tokens + token_idx * hidden_dim;
    
    // Copy token data
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        dest_token[i] = src_token[i];
    }
    
    // In a full implementation with GPU-initiated RDMA:
    // - Use MLX5 or similar provider APIs to post RDMA write from GPU
    // - Write directly to remote GPU memory using rkeys
    // - Signal completion when done
}

// Combine kernel - aggregate expert outputs with weighted sum
__global__ void CombineKernel(
    const float* dispatch_buffer,    // [world_size * max_tokens, hidden_dim]
    const float* weights,            // [num_tokens, num_experts_per_token]
    const int32_t* expert_ids,       // [num_tokens, num_experts_per_token]
    float* output,                   // [num_tokens, hidden_dim]
    const int32_t* recv_offsets,     // [world_size] - where tokens from each rank start
    int num_tokens,
    int num_experts_per_token,
    int num_experts_per_rank,
    int hidden_dim,
    int rank)
{
    int token_idx = blockIdx.x;
    int dim_idx = threadIdx.x;
    
    if (token_idx >= num_tokens || dim_idx >= hidden_dim) return;
    
    float sum = 0.0f;
    float weight_sum = 0.0f;
    
    // Aggregate outputs from all experts assigned to this token
    for (int expert_slot = 0; expert_slot < num_experts_per_token; expert_slot++) {
        int expert_id = expert_ids[token_idx * num_experts_per_token + expert_slot];
        int src_rank = GetExpertRank(expert_id, num_experts_per_rank);
        float weight = weights[token_idx * num_experts_per_token + expert_slot];
        
        // Find where this token's expert output is in the dispatch buffer
        // This requires tracking which tokens were sent to which experts
        // For simplicity, we assume sequential ordering
        int buffer_offset = recv_offsets[src_rank] + token_idx;
        float expert_output = dispatch_buffer[buffer_offset * hidden_dim + dim_idx];
        
        sum += weight * expert_output;
        weight_sum += weight;
    }
    
    // Normalize by weight sum and write output
    if (weight_sum > 0.0f) {
        output[token_idx * hidden_dim + dim_idx] = sum / weight_sum;
    }
}

// Host function to launch dispatch (chooses P2P or RDMA based on topology)
extern "C" void LaunchDispatchKernels(
    const float* input_tokens,
    const int32_t* expert_ids,
    const uintptr_t* peer_ptrs,
    int num_tokens,
    int num_experts_per_token,
    int num_experts_per_rank,
    int hidden_dim,
    int rank,
    int world_size,
    int gpu_per_node,
    int32_t* send_offsets,
    hipStream_t stream)
{
    // Launch dispatch kernel with 2D grid: tokens x experts_per_token
    dim3 grid(num_tokens, num_experts_per_token);
    dim3 block(256); // 256 threads per block
    
    // For simplicity, use P2P kernel (in full implementation, would check topology)
    hipLaunchKernelGGL(DispatchIntraNodeKernel, grid, block, 0, stream,
                       input_tokens, expert_ids, peer_ptrs,
                       num_tokens, num_experts_per_token, num_experts_per_rank,
                       hidden_dim, rank, send_offsets);
}

// Host function to launch combine
extern "C" void LaunchCombineKernel(
    const float* dispatch_buffer,
    const float* weights,
    const int32_t* expert_ids,
    float* output,
    const int32_t* recv_offsets,
    int num_tokens,
    int num_experts_per_token,
    int num_experts_per_rank,
    int hidden_dim,
    int rank,
    hipStream_t stream)
{
    dim3 grid(num_tokens);
    dim3 block(256); // Use 256 threads per token
    
    hipLaunchKernelGGL(CombineKernel, grid, block, 0, stream,
                       dispatch_buffer, weights, expert_ids, output, recv_offsets,
                       num_tokens, num_experts_per_token, num_experts_per_rank,
                       hidden_dim, rank);
}

} // namespace simple_dispatch_combine
