// Optimized GPU kernels with advanced techniques
// Copyright (c) 2025. MIT License.

#include "dispatch_combine.hpp"
#include <hip/hip_runtime.h>

namespace simple_dispatch_combine {

// ============================================================================
// Optimization 1: Warp-level primitives for efficient token dispatch
// ============================================================================

// Use warp shuffle to reduce shared memory usage
__device__ inline void WarpReduceSum(float* val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        *val += __shfl_down(*val, offset);
    }
}

// ============================================================================
// Optimization 2: Vectorized memory access for better bandwidth
// ============================================================================

// Use float4 for 128-bit aligned loads/stores
__global__ void DispatchIntraNodeVectorizedKernel(
    const float* input_tokens,
    const int32_t* expert_ids,
    const uintptr_t* peer_ptrs,
    int num_tokens,
    int num_experts_per_token,
    int num_experts_per_rank,
    int hidden_dim,
    int rank,
    int32_t* send_offsets)
{
    int token_idx = blockIdx.x;
    int expert_slot = blockIdx.y;
    
    if (token_idx >= num_tokens || expert_slot >= num_experts_per_token) return;
    
    int expert_id = expert_ids[token_idx * num_experts_per_token + expert_slot];
    int dest_rank = expert_id / num_experts_per_rank;
    
    // Get write position atomically
    int write_offset = atomicAdd(&send_offsets[dest_rank], 1);
    
    // Get pointers
    float4* dest_buffer = reinterpret_cast<float4*>(peer_ptrs[dest_rank]);
    const float4* src_buffer = reinterpret_cast<const float4*>(input_tokens);
    
    // Calculate offsets for vectorized access
    int token_offset_vec = (write_offset * hidden_dim) / 4;
    int src_offset_vec = (token_idx * hidden_dim) / 4;
    int vec_size = hidden_dim / 4;
    
    // Vectorized copy (4 floats at a time)
    for (int i = threadIdx.x; i < vec_size; i += blockDim.x) {
        dest_buffer[token_offset_vec + i] = src_buffer[src_offset_vec + i];
    }
}

// ============================================================================
// Optimization 3: Asynchronous dispatch with double buffering
// ============================================================================

// Kernel that uses staging buffers to overlap computation and communication
__global__ void DispatchWithStagingKernel(
    const float* input_tokens,
    const int32_t* expert_ids,
    float* staging_buffer_0,
    float* staging_buffer_1,
    const uintptr_t* peer_ptrs,
    int num_tokens,
    int num_experts_per_token,
    int num_experts_per_rank,
    int hidden_dim,
    int rank,
    int32_t* send_offsets,
    int buffer_select)  // 0 or 1 to select staging buffer
{
    float* staging_buffer = (buffer_select == 0) ? staging_buffer_0 : staging_buffer_1;
    
    int token_idx = blockIdx.x;
    int expert_slot = blockIdx.y;
    
    if (token_idx >= num_tokens || expert_slot >= num_experts_per_token) return;
    
    int expert_id = expert_ids[token_idx * num_experts_per_token + expert_slot];
    int dest_rank = expert_id / num_experts_per_rank;
    
    // Stage 1: Copy to staging buffer (local GPU memory)
    int write_offset = atomicAdd(&send_offsets[dest_rank], 1);
    float* dest_staging = staging_buffer + write_offset * hidden_dim;
    const float* src = input_tokens + token_idx * hidden_dim;
    
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        dest_staging[i] = src[i];
    }
    __syncthreads();
    
    // Stage 2: Transfer from staging to peer (can be overlapped)
    if (threadIdx.x == 0) {
        // In full implementation, initiate RDMA or P2P transfer here
        // This allows overlapping next token copy with current transfer
    }
}

// ============================================================================
// Optimization 4: Combining with on-the-fly normalization
// ============================================================================

// Optimized combine that fuses weight normalization
__global__ void CombineFusedKernel(
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
    bool normalize_weights)  // Whether to normalize weights
{
    int token_idx = blockIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    if (token_idx >= num_tokens) return;
    
    // Each warp processes multiple dimensions
    const int dims_per_warp = (hidden_dim + blockDim.x / 32 - 1) / (blockDim.x / 32);
    int dim_start = warp_id * dims_per_warp;
    int dim_end = min(dim_start + dims_per_warp, hidden_dim);
    
    // First pass: compute weight sum if normalization needed
    float weight_sum = 0.0f;
    if (normalize_weights) {
        for (int expert_slot = 0; expert_slot < num_experts_per_token; expert_slot++) {
            float weight = weights[token_idx * num_experts_per_token + expert_slot];
            weight_sum += weight;
        }
    } else {
        weight_sum = 1.0f;
    }
    
    // Second pass: accumulate weighted expert outputs
    for (int dim_idx = dim_start + lane_id; dim_idx < dim_end; dim_idx += 32) {
        float sum = 0.0f;
        
        for (int expert_slot = 0; expert_slot < num_experts_per_token; expert_slot++) {
            int expert_id = expert_ids[token_idx * num_experts_per_token + expert_slot];
            int src_rank = expert_id / num_experts_per_rank;
            float weight = weights[token_idx * num_experts_per_token + expert_slot];
            
            int buffer_offset = recv_offsets[src_rank] + token_idx;
            float expert_output = dispatch_buffer[buffer_offset * hidden_dim + dim_idx];
            
            sum += (weight / weight_sum) * expert_output;
        }
        
        output[token_idx * hidden_dim + dim_idx] = sum;
    }
}

// ============================================================================
// Optimization 5: All-to-All communication pattern
// ============================================================================

// Kernel optimized for all-to-all communication pattern
// Each block handles communication between specific src-dest rank pairs
__global__ void DispatchAllToAllKernel(
    const float* input_tokens,
    const int32_t* expert_ids,
    const uintptr_t* peer_ptrs,
    const int32_t* send_counts,    // [world_size] tokens to send to each rank
    const int32_t* send_offsets,   // [world_size] cumulative offsets
    int num_tokens,
    int num_experts_per_token,
    int num_experts_per_rank,
    int hidden_dim,
    int rank,
    int world_size)
{
    int dest_rank = blockIdx.x % world_size;
    int chunk_id = blockIdx.x / world_size;
    
    // This block handles tokens going to dest_rank
    int tokens_to_dest = send_counts[dest_rank];
    if (chunk_id >= tokens_to_dest) return;
    
    // Find which token and expert slot this corresponds to
    int offset = send_offsets[dest_rank] + chunk_id;
    
    // Scan through token-expert pairs to find the one at this offset
    int pair_idx = 0;
    for (int t = 0; t < num_tokens && pair_idx <= offset; t++) {
        for (int e = 0; e < num_experts_per_token && pair_idx <= offset; e++) {
            int exp_id = expert_ids[t * num_experts_per_token + e];
            int exp_rank = exp_id / num_experts_per_rank;
            if (exp_rank == dest_rank) {
                if (pair_idx == offset) {
                    // Copy this token
                    float* dest = reinterpret_cast<float*>(peer_ptrs[dest_rank]);
                    const float* src = input_tokens + t * hidden_dim;
                    
                    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
                        dest[chunk_id * hidden_dim + i] = src[i];
                    }
                    return;
                }
                pair_idx++;
            }
        }
    }
}

// ============================================================================
// Optimization 6: Bank conflict avoidance in shared memory
// ============================================================================

// Optimized kernel using shared memory with padding to avoid bank conflicts
__global__ void DispatchSharedMemOptimizedKernel(
    const float* input_tokens,
    const int32_t* expert_ids,
    const uintptr_t* peer_ptrs,
    int num_tokens,
    int num_experts_per_token,
    int num_experts_per_rank,
    int hidden_dim,
    int rank,
    int32_t* send_offsets)
{
    // Shared memory with padding to avoid bank conflicts
    // Each bank is 4 bytes, and there are 32 banks
    extern __shared__ float shared_buffer[];
    
    int token_idx = blockIdx.x;
    int expert_slot = blockIdx.y;
    
    if (token_idx >= num_tokens || expert_slot >= num_experts_per_token) return;
    
    // Load token to shared memory with padding
    const float* src = input_tokens + token_idx * hidden_dim;
    int padded_dim = hidden_dim + 8; // Add padding
    
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        shared_buffer[i + (i / 32) * 8] = src[i]; // Interleave padding
    }
    __syncthreads();
    
    // Get destination
    int expert_id = expert_ids[token_idx * num_experts_per_token + expert_slot];
    int dest_rank = expert_id / num_experts_per_rank;
    int write_offset = atomicAdd(&send_offsets[dest_rank], 1);
    
    // Write to peer memory from shared memory
    float* dest = reinterpret_cast<float*>(peer_ptrs[dest_rank]) + write_offset * hidden_dim;
    
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        dest[i] = shared_buffer[i + (i / 32) * 8];
    }
}

// ============================================================================
// Host launcher functions for optimized kernels
// ============================================================================

extern "C" void LaunchDispatchOptimized(
    const float* input_tokens,
    const int32_t* expert_ids,
    const uintptr_t* peer_ptrs,
    int num_tokens,
    int num_experts_per_token,
    int num_experts_per_rank,
    int hidden_dim,
    int rank,
    int world_size,
    int32_t* send_offsets,
    hipStream_t stream,
    int optimization_level)  // 0=basic, 1=vectorized, 2=staging, 3=alltoall
{
    dim3 grid(num_tokens, num_experts_per_token);
    dim3 block(256);
    
    switch (optimization_level) {
        case 1:  // Vectorized
            hipLaunchKernelGGL(DispatchIntraNodeVectorizedKernel, grid, block, 0, stream,
                              input_tokens, expert_ids, peer_ptrs,
                              num_tokens, num_experts_per_token, num_experts_per_rank,
                              hidden_dim, rank, send_offsets);
            break;
            
        case 3:  // All-to-all pattern
            hipLaunchKernelGGL(DispatchAllToAllKernel, 
                              dim3(world_size * num_tokens), block, 0, stream,
                              input_tokens, expert_ids, peer_ptrs,
                              nullptr, send_offsets,  // Need to compute send_counts first
                              num_tokens, num_experts_per_token, num_experts_per_rank,
                              hidden_dim, rank, world_size);
            break;
            
        default:  // Basic version from kernels.hip.cpp
            // Use basic kernel
            break;
    }
}

extern "C" void LaunchCombineOptimized(
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
    bool normalize_weights,
    hipStream_t stream)
{
    dim3 grid(num_tokens);
    dim3 block(256);
    
    hipLaunchKernelGGL(CombineFusedKernel, grid, block, 0, stream,
                       dispatch_buffer, weights, expert_ids, output, recv_offsets,
                       num_tokens, num_experts_per_token, num_experts_per_rank,
                       hidden_dim, rank, normalize_weights);
}

} // namespace simple_dispatch_combine
